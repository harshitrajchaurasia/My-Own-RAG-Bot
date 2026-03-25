#!/usr/bin/env bash
# run.sh — convenience script to start the RAG Bot server locally
set -euo pipefail

if [ ! -f .env ]; then
  echo "⚠️  No .env file found. Copying .env.example → .env"
  cp .env.example .env
  echo "   Edit .env and add your OPENAI_API_KEY, then re-run this script."
  exit 1
fi

export $(grep -v '^#' .env | xargs)

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌  OPENAI_API_KEY is not set in your .env file."
  exit 1
fi

echo "✅  Starting RAG Bot on http://localhost:8000"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
