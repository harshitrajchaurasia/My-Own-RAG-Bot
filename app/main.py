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
  GET  /api/budget           - LLM spend vs the demo budget
  GET  /                     - serve the SPA frontend
"""

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from app.rag import (
    BudgetExceeded,
    chat,
    chat_stream,
    generate_followups,
    generate_suggestions,
    get_active_provider,
    get_budget,
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
    docs_url="/api/docs" if os.getenv("ENABLE_DOCS") else None,
    redoc_url=None,
)


# -- Security headers middleware -------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


# -- Request body size limit middleware ------------------------------------

MAX_BODY_SIZE = 25 * 1024 * 1024  # 25 MB global cap


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with a body larger than MAX_BODY_SIZE."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large."},
            )
        return await call_next(request)


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware)

# -- CORS ------------------------------------------------------------------
# Allow the Cloud Run URL and localhost for dev. The wildcard is NOT used.
_allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://quillshift-*.run.app,http://localhost:8000,http://localhost:8080",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"https://quillshift-.*\.run\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".csv"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB; embeddings are local, so this guards CPU/RAM, not spend


AI_DOWN = "The AI service is having trouble right now. Please try again in a minute."


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
        get_active_provider()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return get_active_provider()


@app.get("/api/stats", summary="Usage statistics")
async def stats():
    return get_usage_stats()


@app.get("/api/budget", summary="LLM spend against the demo budget")
async def budget():
    return get_budget()


@app.post("/api/session", response_model=SessionResponse, summary="Create a new chat session")
async def create_session():
    session_id = new_session()
    return {"session_id": session_id}


@app.get("/api/session/{session_id}", summary="Get session metadata")
async def session_info(session_id: str):
    info = get_session_info(session_id)
    return info


@app.post("/api/demo/{session_id}", summary="Load built-in demo document")
def demo_endpoint(session_id: str):
    try:
        info = load_demo(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "demo_loaded", "session": info}


@app.get("/api/suggestions/{session_id}", summary="Generate suggested questions from documents")
def suggestions_endpoint(session_id: str):
    try:
        questions = generate_suggestions(session_id)
    except BudgetExceeded:
        questions = []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"suggestions": questions}


@app.get("/api/followups/{session_id}", summary="Generate follow-up question chips after a chat turn")
def followups_endpoint(session_id: str):
    try:
        questions = generate_followups(session_id)
    except BudgetExceeded:
        questions = []
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

    # Check every file before ingesting any, so a bad file doesn't leave a half-done upload.
    contents = []
    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext == ".doc":
            raise HTTPException(status_code=400, detail=f"'{file.filename}' is an old Word .doc file. Please save it as .docx or PDF and try again.")
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{ext or 'unknown'}' is not supported. Allowed types: PDF, DOCX, TXT, MD, CSV.",
            )
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is larger than the 5 MB limit for this demo.",
            )
        contents.append(content)

    results = []
    for file, content in zip(files, contents):
        # Browsers send quotes in multipart filenames as %22.
        file.filename = (file.filename or "document").replace("%22", '"')
        try:
            await asyncio.to_thread(ingest_document, session_id, file.filename or "document", content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("ingest failed")
            raise HTTPException(status_code=422, detail=f"Couldn't read '{file.filename}'. It may be damaged or password-protected.") from exc

        results.append({"filename": file.filename, "status": "ingested"})

    info = get_session_info(session_id)
    return {"uploaded": results, "session": info}


@app.post("/api/chat/{session_id}", summary="Chat with the RAG agent")
def chat_endpoint(session_id: str, body: ChatRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = chat(session_id, body.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("chat failed")
        raise HTTPException(status_code=503, detail=AI_DOWN) from exc

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
            log.exception("chat stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': AI_DOWN})}\n\n"

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
# Login (hosted demo). Empty LOGIN_PASS = no login, for local dev.
# ---------------------------------------------------------------------------

import hashlib
import hmac
import logging
import time
from collections import defaultdict, deque

from fastapi import Form
from fastapi.responses import RedirectResponse

log = logging.getLogger("uvicorn.error")  # shows up in `docker logs quillshift`
LOGIN_USER = os.getenv("LOGIN_USER", "")
LOGIN_PASS = os.getenv("LOGIN_PASS", "")
COOKIE = "qs_session"
SESSION_TTL = 7 * 86400
# Key derives from the password, so changing it signs everyone out.
_KEY = hashlib.sha256(f"quillshift:{LOGIN_USER}:{LOGIN_PASS}".encode()).digest()
PUBLIC_PATHS = {"/login", "/api/health"}

# ponytail: in-memory per-IP limiter, fine for one container; move to Redis if it scales out.
_attempts: dict[str, deque] = defaultdict(deque)
MAX_ATTEMPTS, MAX_GLOBAL, WINDOW = 5, 30, 15 * 60  # failed tries per IP / across all IPs


def _prune(now: float) -> int:
    """Drop expired failures; return how many remain across all IPs."""
    total = 0
    for ip in list(_attempts):
        hits = _attempts[ip]
        while hits and hits[0] < now - WINDOW:
            hits.popleft()
        if hits:
            total += len(hits)
        else:
            del _attempts[ip]
    return total


def _sign(expiry: int) -> str:
    mac = hmac.new(_KEY, str(expiry).encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{mac}"


def _valid(token: str | None) -> bool:
    try:
        expiry, _ = (token or "").split(".", 1)
        return int(expiry) > time.time() and hmac.compare_digest(token, _sign(int(expiry)))
    except ValueError:
        return False


def _client_ip(request: Request) -> str:
    # Caddy sets X-Forwarded-For to CF-Connecting-IP; origin only admits Cloudflare.
    return request.headers.get("x-forwarded-for", "").split(",")[0].strip() or request.client.host


class LoginMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not LOGIN_PASS or path in PUBLIC_PATHS or path.startswith("/static/") \
                or _valid(request.cookies.get(COOKIE)):
            return await call_next(request)
        if path.startswith("/api/"):
            return JSONResponse(status_code=401, content={"detail": "Sign in required."})
        return RedirectResponse("/login", status_code=303)


app.add_middleware(LoginMiddleware)


@app.get("/login", include_in_schema=False)
async def login_page(request: Request):
    if not LOGIN_PASS or _valid(request.cookies.get(COOKIE)):
        return RedirectResponse("/", status_code=303)
    return FileResponse(str(STATIC_DIR / "login.html"))


@app.post("/login", include_in_schema=False)
async def login(request: Request, username: str = Form(""), password: str = Form("")):
    ip, now = _client_ip(request), time.time()
    total = _prune(now)
    if len(_attempts.get(ip, ())) >= MAX_ATTEMPTS or total >= MAX_GLOBAL:
        log.warning("login locked out ip=%s", ip)
        return RedirectResponse("/login?e=wait", status_code=303)
    ok = hmac.compare_digest(username.encode(), LOGIN_USER.encode()) & \
        hmac.compare_digest(password.encode(), LOGIN_PASS.encode())
    if not ok:
        _attempts[ip].append(now)
        log.warning("login failed ip=%s", ip)  # never log the typed username/password
        return RedirectResponse("/login?e=bad", status_code=303)
    _attempts.pop(ip, None)
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(COOKIE, _sign(int(now) + SESSION_TTL), max_age=SESSION_TTL,
                    httponly=True, secure=True, samesite="lax")
    return resp


@app.get("/logout", include_in_schema=False)
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE)
    return resp


# ---------------------------------------------------------------------------
# Serve the SPA frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    return FileResponse(str(STATIC_DIR / "index.html"))
