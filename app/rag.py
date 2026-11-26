"""
RAG (Retrieval-Augmented Generation) engine.

Each session gets its own in-memory FAISS vector store so that multiple
users can work independently without sharing document context.

Chat models are tried in order, based on which API keys are set:
  - **anthropic** - Claude (``ANTHROPIC_API_KEY``, model ``CLAUDE_MODEL``)
  - **xai** - Grok (``XAI_API_KEY``, model ``XAI_MODEL``)
  - **gemini** - Google Gemini (``GOOGLE_API_KEY``)
  - **openai** - OpenAI GPT (``OPENAI_API_KEY``)

Embeddings run locally (fastembed, BAAI/bge-small-en-v1.5), so uploading a
document costs nothing. Every LLM call is metered against ``BUDGET_USD``.

Uses the modern LangChain LCEL approach (langchain >=1.x compatible).
"""

import io
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model fallback chain
# ---------------------------------------------------------------------------
# Each entry defines a model to try in order. When one hits a rate limit
# (HTTP 429), we automatically fall through to the next.
# The chain is built dynamically from available API keys.

_GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.0-flash"]
_OPENAI_MODELS = ["gpt-4o-mini"]

# Index into the fallback chain; persists across requests so we stay on
# the working model instead of repeatedly hitting the blocked one.
_current_model_idx: int = 0
_model_cooldowns: dict[int, float] = {}  # idx -> timestamp when cooldown expires
_MODEL_COOLDOWN_SECS = 60  # wait before retrying a rate-limited model


def _build_model_chain() -> list[dict]:
    """Build the ordered fallback chain from available API keys."""
    chain: list[dict] = []
    if os.getenv("ANTHROPIC_API_KEY"):
        chain.append({"provider": "anthropic", "chat_model": os.getenv("CLAUDE_MODEL", "claude-opus-5-5")})
    if os.getenv("XAI_API_KEY"):
        chain.append({"provider": "xai", "chat_model": os.getenv("XAI_MODEL", "grok-4.20-0309-non-reasoning")})
    if os.getenv("GOOGLE_API_KEY"):
        chain += [{"provider": "gemini", "chat_model": m} for m in _GEMINI_MODELS]
    if os.getenv("OPENAI_API_KEY"):
        chain += [{"provider": "openai", "chat_model": m} for m in _OPENAI_MODELS]

    if not chain:
        raise RuntimeError(
            "No LLM provider configured. Set ANTHROPIC_API_KEY, XAI_API_KEY, "
            "GOOGLE_API_KEY or OPENAI_API_KEY in your .env file."
        )
    return chain


def _get_model_chain() -> list[dict]:
    """Cached model chain (built once)."""
    if not hasattr(_get_model_chain, "_chain"):
        _get_model_chain._chain = _build_model_chain()
    return _get_model_chain._chain


def _advance_model(current_idx: int) -> int:
    """Move to the next model in the chain. Returns new index."""
    global _current_model_idx
    chain = _get_model_chain()
    _model_cooldowns[current_idx] = time.time() + _MODEL_COOLDOWN_SECS
    next_idx = (current_idx + 1) % len(chain)
    # Skip models still in cooldown
    tried = 0
    while tried < len(chain):
        cooldown_until = _model_cooldowns.get(next_idx, 0)
        if time.time() >= cooldown_until:
            break
        next_idx = (next_idx + 1) % len(chain)
        tried += 1
    _current_model_idx = next_idx
    return next_idx


def _use_preferred_model() -> None:
    """Start each request on the first model whose cooldown has expired."""
    global _current_model_idx
    now = time.time()
    _current_model_idx = next(
        (i for i in range(len(_get_model_chain())) if _model_cooldowns.get(i, 0) <= now),
        _current_model_idx,
    )


def _should_fall_back(exc: Exception) -> bool:
    """Try the next model on rate limits and provider errors, never on budget stops."""
    return not isinstance(exc, BudgetExceeded)


def get_active_provider() -> dict:
    """Return info about the currently active model for the /api/provider endpoint."""
    chain = _get_model_chain()
    current = chain[_current_model_idx]
    return {
        "provider": current["provider"],
        "model": current["chat_model"],
        "fallback_models": len(chain),
        "current_index": _current_model_idx,
    }


# ---------------------------------------------------------------------------
# Rate limiting (sliding window, in-memory)
# ---------------------------------------------------------------------------

SESSION_RPM = int(os.getenv("SESSION_RPM", "20"))          # per-session requests/min
GLOBAL_RPD  = int(os.getenv("GLOBAL_RPD", "1000"))         # global requests/day
SESSION_TTL = int(os.getenv("SESSION_TTL_HOURS", "2")) * 3600  # seconds

_rate_windows: dict[str, deque] = {}   # session_id -> deque of timestamps
_global_window: deque = deque()        # timestamps of all requests today


def check_rate_limit(session_id: str) -> None:
    """Raise ``RuntimeError`` if the session or global rate limit is exceeded."""
    now = time.time()

    # -- per-session --
    if session_id not in _rate_windows:
        _rate_windows[session_id] = deque()
    win = _rate_windows[session_id]
    while win and win[0] < now - 60:
        win.popleft()
    if len(win) >= SESSION_RPM:
        raise RuntimeError(
            f"You're asking questions a little fast (limit: {SESSION_RPM} per minute). "
            "Please wait a moment and try again."
        )

    # -- global daily --
    day_start = now - 86400
    while _global_window and _global_window[0] < day_start:
        _global_window.popleft()
    if len(_global_window) >= GLOBAL_RPD:
        raise RuntimeError(
            "The service has reached its daily usage limit. "
            "Please try again tomorrow."
        )

    win.append(now)
    _global_window.append(now)


# ---------------------------------------------------------------------------
# Spend budget (persisted, so a restart doesn't reset it)
# ---------------------------------------------------------------------------

BUDGET_USD = float(os.getenv("BUDGET_USD", "5"))
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "1.5"))  # keeps one bad day from draining it all
BUDGET_FILE = Path(os.getenv("BUDGET_FILE", "data/budget.json"))
# Stop before the last call could overshoot: a worst-case answer costs ~$0.08
# (capped question, history, context and max_tokens below).
CALL_RESERVE_USD = 0.10

# USD per 1M tokens (input, output). Unknown models are priced high on purpose.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "grok-4.20-0309-non-reasoning": (1.25, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}
_FALLBACK_PRICE = (5.00, 25.00)


class BudgetExceeded(RuntimeError):
    pass


_budget_lock = threading.Lock()


def _load_budget() -> dict:
    try:
        return json.loads(BUDGET_FILE.read_text())
    except (OSError, ValueError):
        return {"spent_usd": 0.0, "calls": 0, "by_model": {}, "by_day": {}}


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def get_budget() -> dict:
    with _budget_lock:
        b = _load_budget()
    today = b["by_day"].get(_today(), 0.0)
    return {
        "limit_usd": BUDGET_USD,
        "spent_usd": round(b["spent_usd"], 4),
        "remaining_usd": round(max(0.0, BUDGET_USD - b["spent_usd"]), 4),
        "daily_limit_usd": DAILY_BUDGET_USD,
        "spent_today_usd": round(today, 4),
        "calls": b["calls"],
    }


def check_budget() -> None:
    """Raise ``BudgetExceeded`` if the next LLM call could go over a cap."""
    b = get_budget()
    if b["remaining_usd"] < CALL_RESERVE_USD:
        raise BudgetExceeded(
            f"The ${BUDGET_USD:.2f} demo budget has been used up, so new questions "
            "are paused. Thanks for trying QuillShift!"
        )
    if b["spent_today_usd"] > DAILY_BUDGET_USD - CALL_RESERVE_USD:
        raise BudgetExceeded(
            "Today's share of the demo budget is used up. "
            "Please come back tomorrow (resets at midnight UTC)."
        )


def record_usage(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, _FALLBACK_PRICE)
    cost = (input_tokens * pin + output_tokens * pout) / 1_000_000
    with _budget_lock:
        b = _load_budget()
        b["spent_usd"] += cost
        b["calls"] += 1
        b["by_model"][model] = b["by_model"].get(model, 0.0) + cost
        b["by_day"][_today()] = b["by_day"].get(_today(), 0.0) + cost
        BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = BUDGET_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(b, indent=1))
        tmp.replace(BUDGET_FILE)
    logging.getLogger("uvicorn.error").info("llm usage model=%s in=%d out=%d usd=%.5f total=%.4f",
                model, input_tokens, output_tokens, cost, b["spent_usd"])
    return cost


class _Meter(BaseCallbackHandler):
    """Records token usage of every finished LLM call (streaming or not)."""

    def __init__(self, model: str):
        self.model = model

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None)
                if usage:
                    record_usage(self.model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}

SNIPPET_PREVIEW_LENGTH = 150


def _get_session(session_id: str) -> dict:
    """Return (creating if needed) the state dict for *session_id*."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "vector_store": None,
            "chat_history": [],   # list of HumanMessage and AIMessage objects
            "file_names": [],
            "raw_texts": {},      # filename -> extracted plain text for preview
            "document_count": 0,
            "created_at": time.time(),
            "last_active": time.time(),
        }
    else:
        _sessions[session_id]["last_active"] = time.time()
    return _sessions[session_id]


def cleanup_sessions() -> int:
    """Remove sessions older than SESSION_TTL. Returns count of removed sessions."""
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s["last_active"] > SESSION_TTL
    ]
    for sid in expired:
        _sessions.pop(sid, None)
        _rate_windows.pop(sid, None)
    return len(expired)


# ---------------------------------------------------------------------------
# Usage stats
# ---------------------------------------------------------------------------

_usage = {"questions_answered": 0, "documents_analyzed": 0}


def get_usage_stats() -> dict:
    return {
        **_usage,
        "active_sessions": len(_sessions),
    }


# ---------------------------------------------------------------------------
# Document parsing helpers
# ---------------------------------------------------------------------------

def _parse_pdf(content: bytes) -> str:
    import PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(content))
    return "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


def _parse_docx(content: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(content))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def _parse_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "400000"))  # per session, ~120 pages
MAX_FILES = int(os.getenv("MAX_FILES", "5"))                   # per session
MAX_QUESTION_CHARS = 1000
MAX_HISTORY_MESSAGES = 6   # last 3 Q&A turns go back to the model
ANSWER_MAX_TOKENS = 2000


def parse_document(filename: str, content: bytes) -> str:
    """Parse *content* into plain text based on *filename* extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(content)
    if ext == ".docx":
        return _parse_docx(content)
    return _parse_txt(content)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are QuillShift, an expert document analysis assistant. "
    "Your role is to provide precise, well-structured answers strictly based "
    "on the provided document context. Follow these rules:\n\n"
    "1. Only answer from the context below -- never fabricate information.\n"
    "2. If the context is insufficient, clearly state what is missing.\n"
    "3. Use markdown formatting: **bold** for key terms, bullet points for "
    "lists, `code` for technical terms, and headers where appropriate.\n"
    "4. When quoting the document, use > blockquote formatting.\n"
    "5. Be thorough but concise -- aim for clarity over length.\n"
    "6. Keep a plain, professional tone. Do not use emojis.\n"
    "7. The excerpts below are retrieved fresh for each question. Earlier answers "
    "were based on other excerpts, so never retract or 'correct' an earlier answer "
    "just because its source is not in the excerpts shown now.\n\n"
    "Context:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _retrieve_docs(vector_store: FAISS, question: str) -> list[Document]:
    """Retrieve relevant documents for the given question."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    return retriever.invoke(question)


def _get_llm(*, streaming: bool = False, model_idx: int | None = None, max_tokens: int = ANSWER_MAX_TOKENS):
    """Return the LLM instance for the given (or current) model index."""
    check_budget()
    chain = _get_model_chain()
    idx = model_idx if model_idx is not None else _current_model_idx
    entry = chain[idx]
    meter = [_Meter(entry["chat_model"])]
    if entry["provider"] == "anthropic":
        # Opus 5.5 always thinks; low effort keeps answers fast and cheap.
        return ChatAnthropic(model=entry["chat_model"], max_tokens=max_tokens, effort="low",
                             streaming=streaming, callbacks=meter, max_retries=1)
    if entry["provider"] == "xai":
        return ChatOpenAI(model=entry["chat_model"], temperature=0.2, max_tokens=max_tokens,
                          api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1",
                          streaming=streaming, stream_usage=True, callbacks=meter, max_retries=1)
    if entry["provider"] == "gemini":
        return ChatGoogleGenerativeAI(
            model=entry["chat_model"],
            temperature=0.2,
            max_output_tokens=max_tokens,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            streaming=streaming,
            callbacks=meter,
        )
    return ChatOpenAI(
        model=entry["chat_model"],
        temperature=0.2,
        max_tokens=max_tokens,
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=streaming,
        stream_usage=True,
        callbacks=meter,
    )


def _build_chain(vector_store: FAISS, *, streaming: bool = False, model_idx: int | None = None):
    """Build a RAG chain from *vector_store* using LCEL."""
    llm = _get_llm(streaming=streaming, model_idx=model_idx)

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: _format_docs(
                    _retrieve_docs(vector_store, x["question"])
                )
            )
        )
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_embeddings = None
_embeddings_lock = threading.Lock()


def _get_embeddings():
    """Local CPU embeddings: free, and the same model no matter which LLM answers."""
    global _embeddings
    with _embeddings_lock:
        if _embeddings is None:
            _embeddings = FastEmbedEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                cache_dir=os.getenv("FASTEMBED_CACHE", "/app/models"),
            )
    return _embeddings


# Load the model in the background at startup so the first upload isn't slow.
threading.Thread(target=_get_embeddings, daemon=True).start()


def ingest_document(session_id: str, filename: str, content: bytes) -> None:
    """
    Parse *content*, chunk it, embed it, and add it to the session's vector
    store.
    """
    session = _get_session(session_id)
    if len(session["file_names"]) >= MAX_FILES:
        raise ValueError(f"This demo allows up to {MAX_FILES} documents per conversation. Start a new conversation to add more.")
    text = parse_document(filename, content)
    if not text.strip():
        raise ValueError(f"Could not extract any text from '{filename}'.")
    used = sum(len(t) for t in session["raw_texts"].values())
    if used + len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"'{filename}' is too long for this demo (about {MAX_TEXT_CHARS // 3000} pages of text "
            "per conversation). Try a shorter document or a few sections of it."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={"source": filename})
        for chunk in chunks
    ]

    embeddings = _get_embeddings()

    session["file_names"].append(filename)
    session["raw_texts"][filename] = text
    session["document_count"] += len(documents)

    if session["vector_store"] is None:
        session["vector_store"] = FAISS.from_documents(documents, embeddings)
    else:
        session["vector_store"].add_documents(documents)

    _usage["documents_analyzed"] += 1


def _build_sources(retrieved_docs: list[Document]) -> list[dict]:
    """Deduplicate and format source references from retrieved docs."""
    sources = []
    seen = set()
    for doc in retrieved_docs:
        source_name = doc.metadata.get("source", "Unknown")
        preview = doc.page_content[:SNIPPET_PREVIEW_LENGTH].strip()
        key = (source_name, preview)
        if key not in seen:
            seen.add(key)
            sources.append({"file": source_name, "snippet": preview})
    return sources


_EXPIRED = (
    "This conversation has expired (sessions are cleared after 2 hours of inactivity). "
    "Click **New conversation** and add your documents again."
)


def chat(session_id: str, question: str) -> dict:
    """
    Answer *question* using the RAG chain for *session_id*.

    Returns a dict with ``answer``, ``sources``, and ``model`` keys.
    Automatically falls back to the next model on rate-limit errors.
    """
    if session_id not in _sessions:
        raise ValueError(_EXPIRED)
    session = _get_session(session_id)
    if session["vector_store"] is None:
        raise ValueError(
            "No documents have been uploaded yet. Please upload a file first."
        )

    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
    check_budget()
    check_rate_limit(session_id)

    # Retrieve source documents for citations
    retrieved_docs = _retrieve_docs(session["vector_store"], question)
    sources = _build_sources(retrieved_docs)

    _use_preferred_model()
    chain_len = len(_get_model_chain())
    last_err = None

    for attempt in range(chain_len):
        idx = _current_model_idx
        model_info = _get_model_chain()[idx]
        try:
            chain = _build_chain(session["vector_store"], model_idx=idx)
            answer = chain.invoke(
                {
                    "question": question,
                    "chat_history": session["chat_history"][-MAX_HISTORY_MESSAGES:],
                }
            )

            # Persist history
            session["chat_history"].extend(
                [HumanMessage(content=question), AIMessage(content=answer)]
            )
            _usage["questions_answered"] += 1

            return {
                "answer": answer,
                "sources": sources,
                "model": model_info["chat_model"],
            }
        except Exception as exc:
            if _should_fall_back(exc) and attempt < chain_len - 1:
                logger.warning(
                    "%s failed (%s), switching to next model",
                    model_info["chat_model"], exc,
                )
                _advance_model(idx)
                last_err = exc
                continue
            raise

    raise RuntimeError("The AI service is having trouble right now. Please try again in a minute.") from last_err


async def chat_stream(session_id: str, question: str) -> AsyncIterator[dict]:
    """
    Streaming variant of ``chat``.  Yields dicts with a ``type`` key:
      - ``{"type": "status",  "message": "..."}``
      - ``{"type": "sources", "sources": [...]}``
      - ``{"type": "token",   "token": "..."}``
      - ``{"type": "done",    "answer": "...", "model": "..."}``

    Automatically retries with the next model on rate-limit errors.
    """
    if session_id not in _sessions:
        raise ValueError(_EXPIRED)
    session = _get_session(session_id)
    if session["vector_store"] is None:
        raise ValueError(
            "No documents have been uploaded yet. Please upload a file first."
        )

    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
    check_budget()
    check_rate_limit(session_id)

    # Step 1: Retrieve
    yield {"type": "status", "message": "Searching documents..."}
    retrieved_docs = _retrieve_docs(session["vector_store"], question)
    sources = _build_sources(retrieved_docs)
    yield {"type": "status", "message": f"Found {len(retrieved_docs)} relevant sections"}
    yield {"type": "sources", "sources": sources}

    # Step 2: Stream the answer (with model fallback)
    context = _format_docs(retrieved_docs)
    prompt_value = _PROMPT.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": session["chat_history"][-MAX_HISTORY_MESSAGES:],
        }
    )

    _use_preferred_model()
    chain_len = len(_get_model_chain())
    last_err = None

    for attempt in range(chain_len):
        idx = _current_model_idx
        model_info = _get_model_chain()[idx]
        yield {"type": "status", "message": f"Using {model_info['chat_model']}..."}

        try:
            llm = _get_llm(streaming=True, model_idx=idx)
            full_answer = ""
            async for chunk in llm.astream(prompt_value):
                token = chunk.text  # text only; skips Claude's (empty) thinking blocks
                if token:
                    full_answer += token
                    yield {"type": "token", "token": token}

            # Success -- persist history
            session["chat_history"].extend(
                [HumanMessage(content=question), AIMessage(content=full_answer)]
            )
            _usage["questions_answered"] += 1

            yield {"type": "done", "answer": full_answer, "model": model_info["chat_model"]}
            return
        except Exception as exc:
            if _should_fall_back(exc) and attempt < chain_len - 1:
                logger.warning(
                    "%s failed (%s), switching to next model",
                    model_info["chat_model"], exc,
                )
                _advance_model(idx)
                last_err = exc
                yield {
                    "type": "status",
                    "message": f"{model_info['chat_model']} unavailable, trying next model...",
                }
                continue
            raise

    raise RuntimeError("The AI service is having trouble right now. Please try again in a minute.") from last_err


def get_session_info(session_id: str) -> dict:
    """Return metadata about what has been ingested for *session_id*."""
    session = _get_session(session_id)
    return {
        "session_id": session_id,
        "file_names": session["file_names"],
        "document_count": session["document_count"],
        "ready": session["vector_store"] is not None,
    }


def get_document_text(session_id: str, filename: str) -> str:
    """Return the extracted plain text of a document for preview."""
    session = _get_session(session_id)
    text = session["raw_texts"].get(filename)
    if text is None:
        raise ValueError(f"Document '{filename}' not found in this session.")
    return text


def generate_suggestions(session_id: str, count: int = 4) -> list[str]:
    """Generate suggested questions based on the ingested documents."""
    session = _get_session(session_id)
    if session["vector_store"] is None:
        return []

    # Sample a broad set of chunks to give the LLM a good overview
    docs = session["vector_store"].similarity_search("summary overview key topics", k=6)
    context = _format_docs(docs)

    llm = _get_llm(streaming=False, max_tokens=600)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Based on the document excerpts below, "
         "generate exactly {count} short, specific questions a user would likely "
         "want to ask about these documents. Each question should cover a different "
         "topic from the content. Return ONLY the questions, one per line, with no "
         "numbering, bullets, or extra text.\n\nDocument excerpts:\n{context}"),
        ("human", "Generate {count} suggested questions."),
    ])

    result = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "count": count,
    })

    questions = [q.strip() for q in result.strip().splitlines() if q.strip()]
    return questions[:count]


_FOLLOWUP_MAX_ANSWER_CHARS = 600  # truncate answer to cap token usage


def generate_followups(session_id: str, count: int = 3) -> list[str]:
    """Generate follow-up question chips from the last Q&A turn.

    Token-efficient: uses only the last question + a truncated answer
    (no retrieval, no full history). Typically ~250 input tokens.
    """
    session = _get_session(session_id)
    history = session["chat_history"]
    if len(history) < 2:
        return []

    last_question = history[-2].content   # HumanMessage
    last_answer = history[-1].content     # AIMessage
    # Truncate long answers to save tokens
    if len(last_answer) > _FOLLOWUP_MAX_ANSWER_CHARS:
        last_answer = last_answer[:_FOLLOWUP_MAX_ANSWER_CHARS] + "..."

    llm = _get_llm(streaming=False, max_tokens=600)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Based on this Q&A, suggest exactly {count} short follow-up questions "
         "the user might ask next. Each must be under 10 words. "
         "One per line, no numbering or bullets.\n\n"
         "Q: {question}\nA: {answer}"),
        ("human", "Suggest {count} follow-ups."),
    ])

    result = (prompt | llm | StrOutputParser()).invoke({
        "question": last_question,
        "answer": last_answer,
        "count": count,
    })

    questions = [q.strip() for q in result.strip().splitlines() if q.strip()]
    return questions[:count]


def new_session() -> str:
    """Create a brand-new session and return its ID."""
    cleanup_sessions()  # lazy cleanup on new session creation
    session_id = str(uuid.uuid4())
    _get_session(session_id)
    return session_id


# ---------------------------------------------------------------------------
# Demo document
# ---------------------------------------------------------------------------

_DEMO_TEXT = """\
# QuillShift -- Technical Architecture Overview

## What is QuillShift?
QuillShift is a production-grade Retrieval-Augmented Generation (RAG) application \
that lets users upload documents and chat with an AI that understands their content. \
Built with FastAPI, LangChain, and FAISS, it demonstrates modern AI engineering \
practices including vector search, streaming responses, and session isolation.

## Architecture
- **Backend**: FastAPI (Python 3.12) with async streaming endpoints
- **RAG Engine**: LangChain LCEL pipelines with a FAISS vector store
- **LLM**: Anthropic Claude, with xAI Grok as an automatic fallback
- **Embeddings**: BAAI/bge-small-en-v1.5, run locally on CPU (uploads cost nothing)
- **Frontend**: Vanilla JS single-page app with light and dark themes
- **Deployment**: Docker on a small ARM server behind Cloudflare

## Key Features
1. **Multi-format document parsing** -- PDF, DOCX, TXT, MD, CSV
2. **Session isolation** -- Each user gets their own vector store
3. **Streaming responses** -- Token-by-token SSE with pipeline visibility
4. **Source citations** -- Every answer references exact document passages
5. **Spend controls** -- Every LLM call is metered against a fixed dollar budget, with a daily cap
6. **Auto-cleanup** -- Idle sessions are cleared after 2 hours

## RAG Pipeline
When a user asks a question:
1. The question is embedded using the same model as the documents
2. FAISS performs similarity search to find the top-4 most relevant chunks
3. Those chunks are injected as context into the system prompt
4. The LLM generates an answer strictly grounded in the context
5. Source citations are extracted and returned alongside the answer

## Tech Stack Details
| Component | Technology | Why |
|-----------|-----------|-----|
| Web Framework | FastAPI | Async-native, automatic OpenAPI docs, Pydantic validation |
| Vector Store | FAISS | CPU-optimized, no external DB needed, fast similarity search |
| LLM Framework | LangChain | LCEL composability, provider abstraction, streaming support |
| Containerization | Docker | Non-root user, CPU and memory caps, reproducible deploys |
| Hosting | ARM VM + Cloudflare | Origin only accepts Cloudflare traffic; app has its own login |

## About the Developer
This project was built by Harshit Raj Chaurasia as a demonstration of \
applied AI/ML engineering skills, specifically RAG system design, LLM \
integration, production deployment, and full-stack development.

Contact: Check the GitHub repository for more information.
"""


def load_demo(session_id: str) -> dict:
    """Ingest the built-in demo document into *session_id*."""
    session = _get_session(session_id)
    if session["vector_store"] is not None:
        return get_session_info(session_id)

    ingest_document(session_id, "QuillShift-Architecture.md", _DEMO_TEXT.encode())
    return get_session_info(session_id)
