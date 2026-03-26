# QuillShift -- Project Instructions

## What This Project Is

QuillShift is a production-grade RAG (Retrieval-Augmented Generation) chatbot. Users upload documents and chat with an AI that answers strictly from the uploaded content. Built as a portfolio project.

## Tech Stack

- **Backend**: Python 3.12, FastAPI, LangChain LCEL, FAISS (in-memory per session)
- **LLM**: Google Gemini 2.0 Flash (free tier) with OpenAI fallback
- **Frontend**: Vanilla JS SPA -- no frameworks, no build step
- **Deployment**: Docker -> Google Cloud Run (scales to zero = $0)

## Code Conventions

- Python: use type hints, docstrings on public functions, f-strings
- Async endpoints in FastAPI; sync helpers in `rag.py` are fine
- LangChain LCEL style (pipe operators `|`), not legacy chains
- Frontend: vanilla JS only -- no React/Vue/npm. All in `app/static/`
- CSS: custom properties in `:root`, glassmorphism dark theme, no CSS frameworks
- No external databases -- all state is in-memory (FAISS + dicts)

## File Layout

```
app/
  __init__.py          # Exports FastAPI `app`
  main.py              # All API endpoints
  rag.py               # RAG engine: providers, sessions, rate limiting, streaming
  static/
    index.html         # HTML shell
    app.js             # Frontend SPA logic
    style.css          # All styles
deploy.sh              # Cloud Run deployment script
docker-compose.yml     # Local dev
Dockerfile             # Multi-stage build
requirements.txt       # Python deps
.env.example           # Config template
```

## Key Design Rules

1. **No API key required for end users** -- the server provides the LLM key
2. **Session isolation** -- each user gets their own FAISS vector store
3. **Rate limiting** -- sliding window per-session (20/min) and global daily (1000/day)
4. **Streaming** -- SSE via `chat_stream()` async generator, not WebSockets
5. **Demo mode** -- built-in document so users can try without uploading
6. **Provider auto-detection** -- prefers Gemini if `GOOGLE_API_KEY` is set

## Running Locally

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Deploying

```bash
chmod +x deploy.sh
./deploy.sh
```

## When Making Changes

- Keep the frontend vanilla JS -- no build tools or frameworks
- Don't add external databases -- keep FAISS in-memory
- Don't expose API keys in frontend code
- Rate limiting logic lives in `rag.py`, not `main.py`
- All new endpoints go in `main.py`; all RAG logic in `rag.py`
- Test with `GET /api/health` to verify the server is running
