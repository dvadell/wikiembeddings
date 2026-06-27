FROM python:3.12-slim

# -----------------------------
# System setup (minimal)
# -----------------------------
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install uv
# -----------------------------
RUN pip install --no-cache-dir uv

# -----------------------------
# Dependency layer (cached)
# -----------------------------
COPY pyproject.toml uv.lock ./

# Force venv inside project
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

RUN uv sync --frozen --no-install-project

# -----------------------------
# Application layer
# -----------------------------
COPY app/ ./app/
COPY wiki_faiss.index.stub /app/wiki_faiss.index
COPY wiki_titles.txt.stub /app/wiki_titles.txt

# -----------------------------
# Writable data directory for docker-compose named volume.
# Docker copies image content into a named volume on first creation;
# `/data` must be owned by `app` or the non-root USER cannot write.
# -----------------------------
RUN mkdir -p /data && chown app:app /data

RUN useradd -m app && chown -R app:app /app

USER app

# -----------------------------
# Environment
# -----------------------------
ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_NAME="all-MiniLM-L6-v2" \
    PORT=8000 \
    WORKERS=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# -----------------------------
# Healthcheck
# -----------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# -----------------------------
# Runtime
# -----------------------------
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0"]
