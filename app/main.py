"""Wikipedia Semantic Search — Persistent Server.

Loads the FAISS index and sentence-transformer model once at startup,
then serves queries over HTTP. Each search takes ~10ms instead of ~10s.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import faiss
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

from app.config import (
    DEFAULT_K,
    DEFAULT_NPROBE,
    FAISS_INDEX,
    MODEL_NAME,
    PORT,
    TITLES_FILE,
    WORKERS,
)

logger = logging.getLogger(__name__)

# Global state — loaded once at startup
state: dict[str, Any] = {}


def _validate_path(path: str) -> str:
    """Return *path* if it exists, else sys.exit with a message."""
    if not os.path.exists(path):
        sys.exit(f"'{path}' not found. Run wiki_search.py --build-index first.")
    return path


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:  # noqa: D401
    """Load everything into memory before accepting requests."""
    try:
        _verify_faiss_installed()
    except SystemExit as exc:
        logger.error("Startup failed: %s", exc)
        raise

    _validate_path(FAISS_INDEX)
    _validate_path(TITLES_FILE)

    logger.info("Loading titles …")
    t0 = time.time()
    with open(TITLES_FILE, "r", encoding="utf-8") as f:
        state["titles"] = f.read().splitlines()
    logger.info("  %s titles loaded in %.1fs", len(state["titles"]), time.time() - t0)

    logger.info("Loading FAISS index …")
    t0 = time.time()
    state["index"] = faiss.read_index(FAISS_INDEX)
    state["index"].nprobe = DEFAULT_NPROBE
    logger.info("  Index loaded in %.1fs", time.time() - t0)

    logger.info("Loading model '%s' …", MODEL_NAME)
    t0 = time.time()
    state["model"] = SentenceTransformer(MODEL_NAME)
    logger.info("  Model loaded in %.1fs", time.time() - t0)

    logger.info("\nServer ready. Listening on http://localhost:%d\n", PORT)
    yield
    state.clear()
    logger.info("State cleared on shutdown.")


def _verify_faiss_installed() -> None:
    """Ensure FAISS is importable; sys.exit with an actionable message otherwise."""
    try:
        import faiss as _  # noqa: F401 (top-level import side-effect)
    except ImportError:
        sys.exit("FAISS not installed. Run: pip install faiss-cpu")


app = FastAPI(
    title="Wikipedia Semantic Search",
    description="Finds the closest Wikipedia titles to a natural language question.",
    lifespan=lifespan,
)


@app.get("/search")
def search(  # noqa: D402
    q: str = Query(..., description="Your question or search phrase"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results"),
    nprobe: int = Query(
        DEFAULT_NPROBE, ge=1, le=4096, description="FAISS cells to search (higher = more accurate)"
    ),
) -> JSONResponse:
    model = state["model"]
    index = state["index"]
    titles = state["titles"]

    index.nprobe = nprobe

    t0 = time.time()
    q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
    q_emb /= np.linalg.norm(q_emb, keepdims=True)

    scores, indices = index.search(q_emb, k)
    elapsed_ms = (time.time() - t0) * 1000

    results = [
        {"rank": i + 1, "title": titles[idx], "score": round(float(score), 4)}
        for i, (score, idx) in enumerate(zip(scores[0], indices[0]))
    ]

    return JSONResponse(
        {
            "query": q,
            "results": results,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "titles_loaded": len(state.get("titles", []))}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=WORKERS)
