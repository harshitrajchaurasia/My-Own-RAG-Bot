# -- Build stage -----------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# -- Runtime stage ---------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Bake the local embedding model into the image (no download at runtime)
ENV FASTEMBED_CACHE=/app/models BUDGET_FILE=/data/budget.json
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models')"

# Copy application source
COPY app/ ./app/

# Create directory for temporary uploads (not persisted)
RUN mkdir -p /tmp/rag_uploads

EXPOSE 8080

# Use a non-root user for security
RUN adduser --disabled-password --gecos "" ragbot && mkdir -p /data && chown ragbot /data
USER ragbot

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
