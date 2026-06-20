FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv && \
    uv sync --frozen

# ── runtime stage -----------------------------------------------------------
FROM python:3.12-alpine AS runtime

RUN addgroup -S app && adduser -S app -G app

WORKDIR /app

# Only curl needed at runtime (HEALTHCHECK).  numpy/faiss-cpu ship
# manylinux wheels that bundle BLAS; ONNX has musl-compatible wheels.
# openblas-libs and libstdc++ are Debian names — Alpine rejects them.
RUN apk add --no-cache curl

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv && \
    uv sync --frozen

# Stub data files (mounted via volumes in docker-compose; stubs keep the image
# self-contained for non-volume deployments).
COPY wiki_faiss.index.stub /app/wiki_faiss.index
COPY wiki_titles.txt.stub  /app/wiki_titles.txt

# Application source — changes more often than data files; layer-cache friendly.
COPY app/ ./app/
RUN chown -R app:app /app

USER app

# Health check (PRD §8.1 / SC2) — poll every 30 s.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Runtime defaults (override via docker-compose / --env-file).
ENV MODEL_NAME="all-MiniLM-L6-v2" \
    PORT=8000 \
    WORKERS=1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0"]
