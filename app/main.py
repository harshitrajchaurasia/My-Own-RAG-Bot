"""
FastAPI application entry-point for the RAG Bot.

Endpoints:
  POST /api/session          - create a new session
  GET  /api/session/{id}     - get session metadata
  POST /api/upload/{id}      - upload one or more documents
  POST /api/chat/{id}        - send a message to the RAG agent
  GET  /api/chat/{id}/stream - SSE streaming variant
  POST /api/demo/{id}        - load built-in demo document
  GET  /api/health           - health check
  GET  /api/provider         - active LLM provider info
  GET  /api/stats            - usage statistics
  GET  /                     - serve the SPA frontend
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag import (
    chat,
    chat_stream,
    generate_followups,
    generate_suggestions,
    get_active_provider,
    get_document_text,
    get_session_info,
    get_usage_stats,
    ingest_document,
    load_demo,
    new_session,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QuillShift",
    description="Upload documents and chat with an AI that understands them.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".md", ".csv"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class ChatRequest(BaseModel):
    question: str


class SessionResponse(BaseModel):
    session_id: str


@app.get("/api/health", summary="Health check")
async def health():
    return {"status": "ok"}


@app.get("/api/provider", summary="Get active LLM provider info")
async def provider_info():
    try:
        provider = get_active_provider()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"provider": provider}


@app.get("/api/stats", summary="Usage statistics")
async def stats():
    return get_usage_stats()


@app.post("/api/session", response_model=SessionResponse, summary="Create a new chat session")
async def create_session():
    session_id = new_session()
    return {"session_id": session_id}


@app.get("/api/session/{session_id}", summary="Get session metadata")
async def session_info(session_id: str):
    info = get_session_info(session_id)
    return info


@app.post("/api/demo/{session_id}", summary="Load built-in demo document")
async def demo_endpoint(session_id: str):
    try:
        info = load_demo(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "demo_loaded", "session": info}


@app.get("/api/suggestions/{session_id}", summary="Generate suggested questions from documents")
async def suggestions_endpoint(session_id: str):
    try:
        questions = generate_suggestions(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"suggestions": questions}


@app.get("/api/followups/{session_id}", summary="Generate follow-up question chips after a chat turn")
async def followups_endpoint(session_id: str):
    try:
        questions = generate_followups(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"followups": questions}


@app.get("/api/document/{session_id}/{filename:path}", summary="Get extracted text of a document for preview")
async def document_text_endpoint(session_id: str, filename: str):
    try:
        text = get_document_text(session_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"filename": filename, "text": text}


@app.post("/api/upload/{session_id}", summary="Upload documents for a session")
async def upload_documents(session_id: str, files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{ext}' is not supported. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' exceeds the 20 MB size limit.",
            )

        try:
            ingest_document(session_id, file.filename or "document", content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to process '{file.filename}': {exc}") from exc

        results.append({"filename": file.filename, "status": "ingested"})

    info = get_session_info(session_id)
    return {"uploaded": results, "session": info}


@app.post("/api/chat/{session_id}", summary="Chat with the RAG agent")
async def chat_endpoint(session_id: str, body: ChatRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = chat(session_id, body.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the answer: {exc}") from exc

    return {"answer": result["answer"], "sources": result["sources"]}


@app.post("/api/chat/{session_id}/stream", summary="Chat with SSE streaming")
async def chat_stream_endpoint(session_id: str, body: ChatRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_generator():
        try:
            async for event in chat_stream(session_id, body.question):
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except RuntimeError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'An error occurred: {exc}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Serve the SPA frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    return FileResponse(str(STATIC_DIR / "index.html"))
