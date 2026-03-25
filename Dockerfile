# ── Build stage ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Create directory for temporary uploads (not persisted)
RUN mkdir -p /tmp/rag_uploads

EXPOSE 8000

# Use a non-root user for security
RUN adduser --disabled-password --gecos "" ragbot
USER ragbot

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
