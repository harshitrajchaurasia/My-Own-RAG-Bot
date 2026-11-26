# QuillShift - Intelligent Document Chat

A production-ready **Retrieval-Augmented Generation (RAG)** application that lets users upload documents and have intelligent conversations about their contents. Built to showcase modern RAG architecture with a polished, non-generic interface.

**FastAPI** | **LangChain LCEL** | **FAISS** | **Claude** | **Grok** | **Vanilla JS**

**[Live demo](https://quillshift.saakura.dev)** (it spends API credits, so it needs a login - [ask me for one](https://harshit.saakura.dev/projects/#login)) - part of my [side projects](https://harshit.saakura.dev/projects/)

---

## Key Features

| Feature | Details |
|---|---|
| **Multi-format Upload** | PDF, DOCX, DOC, TXT, Markdown, CSV - drag & drop or browse (5 MB per file) |
| **Vector Search** | FAISS similarity search with local `bge-small-en-v1.5` embeddings, so uploads cost nothing |
| **Model Fallback** | Claude answers by default; if it errors or is rate-limited, Grok takes over automatically |
| **Conversational Memory** | The last 3 Q&A turns go back to the model for natural follow-ups |
| **Spend Controls** | Every LLM call is metered in dollars against `BUDGET_USD` (plus a daily cap); the UI shows how much is used |
| **Source Citations** | Every answer references the exact document passages used |
| **Distinctive UI** | Custom warm-palette theme with light/dark modes - not a generic AI chatbot |
| **Session Isolation** | Per-session FAISS stores - users never see each other's data |
| **Responsive Design** | Works on desktop and mobile with collapsible sidebar |
| **Docker-ready** | One-command deployment via Docker Compose |

---

## Architecture

```
+-------------------------------------------------------------+
|                        Browser (SPA)                         |
|   Vanilla JS - Custom Markdown renderer - Warm palette UI    |
+----------------------------+--------------------------------+
                             |  REST API
+----------------------------v--------------------------------+
|                    FastAPI Server                             |
|   /api/session - /api/upload - /api/chat - /api/health       |
+----------------------------+--------------------------------+
                             |
+----------------------------v--------------------------------+
|                  LangChain RAG Pipeline                       |
|                                                              |
|   +------+   +-------+   +---------+   +----------+         |
|   |Parse |-->| Chunk  |-->| Embed   |-->|  FAISS   |         |
|   |(PDF, |   |(1000t, |   |(ada-002)|   |(per-user)|         |
|   |DOCX) |   |150 ovl)|   |         |   |          |         |
|   +------+   +-------+   +---------+   +----+-----+         |
|                                              |               |
|   +------------------------------------------v-----+         |
|   |  Query -> Top-4 similarity -> GPT-3.5 + History |         |
|   |  Returns: answer + source citations              |         |
|   +--------------------------------------------------+         |
+-------------------------------------------------------------+
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- An [Anthropic](https://console.anthropic.com/) and/or [xAI](https://console.x.ai/) API key (Gemini and OpenAI also work)

### 1. Clone & install

```bash
git clone https://github.com/harshitrajchaurasia/My-Own-RAG-Bot.git
cd My-Own-RAG-Bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env -> set ANTHROPIC_API_KEY and/or XAI_API_KEY
```

### 3. Run

```bash
bash run.sh
# or: uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**

---

## Docker

```bash
cp .env.example .env   # set ANTHROPIC_API_KEY and/or XAI_API_KEY
docker compose up --build
```

---

## Deploy to Google Cloud Run (Free Tier)

Google Cloud Run's free tier includes **2 million requests/month** and **360,000 GB-seconds** of compute -- more than enough for a portfolio project.

```bash
# 1. Build & push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/quillshift

# 2. Deploy
gcloud run deploy quillshift \
  --image gcr.io/YOUR_PROJECT/quillshift \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ANTHROPIC_API_KEY=sk-ant-... \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2
```

> **Cost note:** With `--min-instances 0`, Cloud Run scales to zero when idle -- you only pay for actual requests. LLM costs are usage-based and capped by `BUDGET_USD`. The spend file lives in `BUDGET_FILE`; on Cloud Run mount a volume there, or the meter resets with each instance.

---

## REST API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/session` | Create a new session |
| `GET` | `/api/session/{id}` | Get session metadata |
| `POST` | `/api/upload/{id}` | Upload documents (multipart) |
| `POST` | `/api/chat/{id}` | Chat - returns `{answer, sources}` |
| `GET` | `/api/budget` | Dollars spent vs the budget |

Interactive docs: **http://localhost:8000/docs**

---

## Project Structure

```
My-Own-RAG-Bot/
  app/
    main.py          # FastAPI app, REST endpoints, health check
    rag.py           # RAG pipeline: parse -> chunk -> embed -> retrieve -> answer
    static/
      index.html     # SPA shell
      login.html     # Sign-in page, used when LOGIN_USER/LOGIN_PASS are set
      style.css      # Custom theme with light/dark modes
      app.js         # Client: sessions, upload, chat, markdown, citations
  requirements.txt
  Dockerfile           # Multi-stage build, non-root user
  docker-compose.yml
  run.sh
  .env.example
  README.md
```

---

## How It Works

1. **Upload** - User drops files into the sidebar
2. **Parse** - PyPDF2 / python-docx / plain text extraction
3. **Chunk** - RecursiveCharacterTextSplitter (1000 tokens, 150 overlap)
4. **Embed** - `BAAI/bge-small-en-v1.5` runs locally on CPU (fastembed)
5. **Store** - FAISS in-memory vector store (per session)
6. **Query** - User question -> embedding -> top-4 similarity search
7. **Generate** - Claude (Grok as fallback) answers with retrieved context + recent history; token usage is priced and added to the budget
8. **Cite** - Source document snippets returned alongside the answer

---

## License

[MIT](LICENSE)
