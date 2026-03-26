"""
RAG (Retrieval-Augmented Generation) engine.

Each session gets its own in-memory FAISS vector store so that multiple
users can work independently without sharing document context.

Supports **two LLM providers** so end-users don't need their own API key:
  - **gemini** - Google Gemini via Google AI Studio (generous free tier).
  - **openai** - OpenAI GPT models (requires a paid key).

Set ``LLM_PROVIDER`` to ``gemini`` or ``openai`` in ``.env``, or leave it
unset and the engine will auto-detect based on which API key is present.

Uses the modern LangChain LCEL approach (langchain >=1.x compatible).
"""

import io
import os
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
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> str:
    """Return the active LLM provider name (``gemini`` or ``openai``).

    Resolution order:
      1. Explicit ``LLM_PROVIDER`` env var.
      2. Auto-detect: prefer Gemini (free tier) if ``GOOGLE_API_KEY`` is set,
         otherwise fall back to OpenAI.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in ("gemini", "openai"):
        return explicit

    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    raise RuntimeError(
        "No LLM provider configured. Set GOOGLE_API_KEY (free) or "
        "OPENAI_API_KEY in your .env file."
    )


def get_active_provider() -> str:
    """Public helper - returns the provider name for the /api/provider endpoint."""
    return _detect_provider()


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
            f"Rate limit exceeded -- max {SESSION_RPM} requests per minute. "
            "Please wait a moment."
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


def parse_document(filename: str, content: bytes) -> str:
    """Parse *content* into plain text based on *filename* extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(content)
    if ext in (".docx", ".doc"):
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
    "5. Be thorough but concise -- aim for clarity over length.\n\n"
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
        search_kwargs={"k": 4},
    )
    return retriever.invoke(question)


def _get_llm(*, streaming: bool = False):
    """Return the LLM instance for the active provider."""
    provider = _detect_provider()
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            streaming=streaming,
        )
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=streaming,
    )


def _build_chain(vector_store: FAISS, *, streaming: bool = False):
    """Build a RAG chain from *vector_store* using LCEL."""
    llm = _get_llm(streaming=streaming)

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

def _get_embeddings():
    """Return the embedding model for the active provider."""
    provider = _detect_provider()
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


def ingest_document(session_id: str, filename: str, content: bytes) -> None:
    """
    Parse *content*, chunk it, embed it, and add it to the session's vector
    store.
    """
    text = parse_document(filename, content)
    if not text.strip():
        raise ValueError(f"Could not extract any text from '{filename}'.")

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

    session = _get_session(session_id)
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


def chat(session_id: str, question: str) -> dict:
    """
    Answer *question* using the RAG chain for *session_id*.

    Returns a dict with ``answer`` and ``sources`` keys.
    Raises ``ValueError`` if no documents have been ingested yet.
    """
    session = _get_session(session_id)
    if session["vector_store"] is None:
        raise ValueError(
            "No documents have been uploaded yet. Please upload a file first."
        )

    check_rate_limit(session_id)

    # Retrieve source documents for citations
    retrieved_docs = _retrieve_docs(session["vector_store"], question)
    sources = _build_sources(retrieved_docs)

    chain = _build_chain(session["vector_store"])

    answer = chain.invoke(
        {
            "question": question,
            "chat_history": session["chat_history"],
        }
    )

    # Persist history
    session["chat_history"].extend(
        [HumanMessage(content=question), AIMessage(content=answer)]
    )
    _usage["questions_answered"] += 1

    return {"answer": answer, "sources": sources}


async def chat_stream(session_id: str, question: str) -> AsyncIterator[dict]:
    """
    Streaming variant of ``chat``.  Yields dicts with a ``type`` key:
      - ``{"type": "status",  "message": "..."}``
      - ``{"type": "sources", "sources": [...]}``
      - ``{"type": "token",   "token": "..."}``
      - ``{"type": "done",    "answer": "..."}``
    """
    session = _get_session(session_id)
    if session["vector_store"] is None:
        raise ValueError(
            "No documents have been uploaded yet. Please upload a file first."
        )

    check_rate_limit(session_id)

    # Step 1: Retrieve
    yield {"type": "status", "message": "Searching documents..."}
    retrieved_docs = _retrieve_docs(session["vector_store"], question)
    sources = _build_sources(retrieved_docs)
    yield {"type": "status", "message": f"Found {len(retrieved_docs)} relevant sections"}
    yield {"type": "sources", "sources": sources}

    # Step 2: Stream the answer
    yield {"type": "status", "message": "Generating answer..."}

    context = _format_docs(retrieved_docs)
    prompt_value = _PROMPT.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": session["chat_history"],
        }
    )

    llm = _get_llm(streaming=True)
    full_answer = ""
    async for chunk in llm.astream(prompt_value):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full_answer += token
            yield {"type": "token", "token": token}

    # Persist history
    session["chat_history"].extend(
        [HumanMessage(content=question), AIMessage(content=full_answer)]
    )
    _usage["questions_answered"] += 1

    yield {"type": "done", "answer": full_answer}


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

    llm = _get_llm(streaming=False)
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

    llm = _get_llm(streaming=False)
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
- **RAG Engine**: LangChain LCEL pipelines with FAISS vector store
- **LLM**: Google Gemini 2.0 Flash (free tier) with OpenAI fallback
- **Embeddings**: Google gemini-embedding-001 / OpenAI text-embedding-ada-002
- **Frontend**: Vanilla JS SPA with glassmorphism dark theme
- **Deployment**: Docker -> Google Cloud Run (scales to zero, $0 cost)

## Key Features
1. **Multi-format document parsing** -- PDF, DOCX, TXT, MD, CSV
2. **Session isolation** -- Each user gets their own vector store
3. **Streaming responses** -- Token-by-token SSE with pipeline visibility
4. **Source citations** -- Every answer references exact document passages
5. **Rate limiting** -- Per-session and global daily caps protect free-tier credits
6. **Auto-cleanup** -- Idle sessions are garbage collected after 2 hours

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
| Containerization | Docker | Multi-stage build, non-root user, reproducible deploys |
| Hosting | Cloud Run | Scale-to-zero, free tier generous, HTTPS included |

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
