# Base image
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps separately to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY main.py ./

# Create non-root user with explicit UID
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER 1000

EXPOSE 8000

# Default environment (can be overridden at runtime)
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    QDRANT_HOST=localhost \
    QDRANT_PORT=6333 \
    QDRANT_COLLECTION=rust_docs \
    EMBEDDING_DIM=128 \
    CACHE_TTL_SECONDS=300

# Health hints (optional)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import sys,urllib.request; u=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); sys.exit(0 if u.status==200 else 1)" || exit 1

# Run the app with uvicorn; allow APP_PORT override
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${APP_PORT:-8000}"]
