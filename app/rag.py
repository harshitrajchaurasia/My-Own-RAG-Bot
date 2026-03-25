"""
RAG (Retrieval-Augmented Generation) engine.

Each session gets its own in-memory FAISS vector store so that multiple
users can work independently without sharing document context.

Supports **two LLM providers** so end-users don't need their own API key:
  • **gemini** – Google Gemini via Google AI Studio (generous free tier).
  • **openai** – OpenAI GPT models (requires a paid key).

Set ``LLM_PROVIDER`` to ``gemini`` or ``openai`` in ``.env``, or leave it
unset and the engine will auto-detect based on which API key is present.

Uses the modern LangChain LCEL approach (langchain >=1.x compatible).
"""

import io
import os
import uuid
from pathlib import Path
from typing import Optional

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
    """Public helper – returns the provider name for the /api/provider endpoint."""
    return _detect_provider()

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
            "document_count": 0,
        }
    return _sessions[session_id]


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
    "You are DocuMind, an expert document analysis assistant. "
    "Your role is to provide precise, well-structured answers strictly based "
    "on the provided document context. Follow these rules:\n\n"
    "1. Only answer from the context below — never fabricate information.\n"
    "2. If the context is insufficient, clearly state what is missing.\n"
    "3. Use markdown formatting: **bold** for key terms, bullet points for "
    "lists, `code` for technical terms, and headers where appropriate.\n"
    "4. When quoting the document, use > blockquote formatting.\n"
    "5. Be thorough but concise — aim for clarity over length.\n\n"
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


def _build_chain(vector_store: FAISS):
    """Build a RAG chain from *vector_store* using LCEL."""
    provider = _detect_provider()

    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    else:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

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
            model="models/embedding-001",
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
    session["document_count"] += len(documents)

    if session["vector_store"] is None:
        session["vector_store"] = FAISS.from_documents(documents, embeddings)
    else:
        session["vector_store"].add_documents(documents)


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

    # Retrieve source documents for citations
    retrieved_docs = _retrieve_docs(session["vector_store"], question)

    # Build unique source references
    sources = []
    seen = set()
    for doc in retrieved_docs:
        source_name = doc.metadata.get("source", "Unknown")
        preview = doc.page_content[:SNIPPET_PREVIEW_LENGTH].strip()
        key = (source_name, preview)
        if key not in seen:
            seen.add(key)
            sources.append({"file": source_name, "snippet": preview})

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

    return {"answer": answer, "sources": sources}


def get_session_info(session_id: str) -> dict:
    """Return metadata about what has been ingested for *session_id*."""
    session = _get_session(session_id)
    return {
        "session_id": session_id,
        "file_names": session["file_names"],
        "document_count": session["document_count"],
        "ready": session["vector_store"] is not None,
    }


def new_session() -> str:
    """Create a brand-new session and return its ID."""
    session_id = str(uuid.uuid4())
    _get_session(session_id)
    return session_id
