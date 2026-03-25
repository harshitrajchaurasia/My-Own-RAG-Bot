# My-Own-RAG-Bot

A **Retrieval-Augmented Generation (RAG)** chatbot that lets users upload their own documents (PDF, DOCX, TXT, Markdown, CSV) and instantly chat with an AI agent that understands the file contents.

Built with **FastAPI** · **LangChain** · **FAISS** · **OpenAI GPT-3.5-Turbo**  
Deployable on **Google Cloud Run**, **Docker**, or any Python host.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📂 File Upload | PDF, DOCX, DOC, TXT, MD, CSV (up to 20 MB each) |
| 🔍 Smart Retrieval | FAISS vector search with OpenAI embeddings |
| 💬 Conversational AI | GPT-3.5-Turbo with conversation memory |
| 🔒 Session Isolation | Each browser session has its own private vector store |
| 🌙 Dark UI | Responsive dark-mode chat interface |
| 🐳 Docker-ready | One-command startup with Docker Compose |

---

## 🚀 Quick Start (Local)

### 1 — Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 2 — Install dependencies

```bash
git clone https://github.com/harshitrajchaurasia/My-Own-RAG-Bot.git
cd My-Own-RAG-Bot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Configure the API key

```bash
cp .env.example .env
# Edit .env and set:  OPENAI_API_KEY=sk-...
```

### 4 — Run the server

```bash
bash run.sh
# or directly:
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## 🐳 Docker Compose

```bash
cp .env.example .env          # set OPENAI_API_KEY
docker compose up --build
```

Open **http://localhost:8000**.

---

## ☁️ Deploy to Google Cloud Run

```bash
# 1. Build & push the image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/ragbot

# 2. Deploy
gcloud run deploy ragbot \
  --image gcr.io/YOUR_PROJECT/ragbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=sk-...
```

---

## 📡 REST API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/session` | Create a new session |
| `GET` | `/api/session/{id}` | Get session metadata |
| `POST` | `/api/upload/{id}` | Upload one or more files (multipart) |
| `POST` | `/api/chat/{id}` | Send a question, receive an answer |

Interactive docs available at **http://localhost:8000/docs** (Swagger UI).

---

## 🗂 Project Structure

```
My-Own-RAG-Bot/
├── app/
│   ├── main.py          # FastAPI app & REST endpoints
│   ├── rag.py           # RAG pipeline (parse → chunk → embed → retrieve → answer)
│   └── static/
│       ├── index.html   # Single-page chat UI
│       ├── style.css    # Dark-mode responsive styles
│       └── app.js       # Frontend logic (session, upload, chat)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.sh               # Local convenience start script
├── .env.example
└── README.md
```

---

## 🔧 How It Works

```
User uploads file
       │
       ▼
  Parse text  (PyPDF2 / python-docx / plain text)
       │
       ▼
  Chunk text  (RecursiveCharacterTextSplitter, 1000 tokens, 150 overlap)
       │
       ▼
  Embed chunks  (OpenAI text-embedding-ada-002)
       │
       ▼
  Store in FAISS  (per-session, in-memory)
       │
  User asks question
       │
       ▼
  Embed question → similarity search → top-4 chunks
       │
       ▼
  GPT-3.5-Turbo answers with retrieved context + conversation history
```

---

## 📄 License

[MIT](LICENSE)
