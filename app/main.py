"""
FastAPI application entry-point for the RAG Bot.

Endpoints:
  POST /api/session          – create a new session
  GET  /api/session/{id}     – get session metadata
  POST /api/upload/{id}      – upload one or more documents
  POST /api/chat/{id}        – send a message to the RAG agent
  GET  /                     – serve the SPA frontend
"""

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag import chat, get_session_info, ingest_document, new_session

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Bot",
    description="Upload your documents and chat with an AI agent that understands them.",
    version="1.0.0",
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


@app.post("/api/session", response_model=SessionResponse, summary="Create a new chat session")
async def create_session():
    session_id = new_session()
    return {"session_id": session_id}


@app.get("/api/session/{session_id}", summary="Get session metadata")
async def session_info(session_id: str):
    info = get_session_info(session_id)
    return info


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
        answer = chat(session_id, body.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the answer: {exc}") from exc

    return {"answer": answer}


# ---------------------------------------------------------------------------
# Serve the SPA frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    return FileResponse(str(STATIC_DIR / "index.html"))
