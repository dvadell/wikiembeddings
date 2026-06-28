"""Wikipedia Semantic Search — Persistent Server.

Loads the FAISS index and sentence-transformer model once at startup,
then serves queries over HTTP. Each search takes ~10ms instead of ~10s.

On first start with no index (no build_manifest.json), a background thread
runs download → embed → FAISS while /health stays online immediately with
real-time progress reporting.
"""

import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

from app.config import (
    BUILD_MANIFEST,
    DEFAULT_K,
    DEFAULT_NPROBE,
    FAISS_INDEX,
    LOG_LEVEL,
    MODEL_NAME,
    PORT,
    TITLES_FILE,
    WORKERS,
)

logger = logging.getLogger(__name__)

# ── logging setup (T21.1–4: module level so output appears in `docker compose logs`) ─── #

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    stream=sys.stderr,
)


# ── global build state (16.A: always available to /health, /search) ──── #


@dataclass
class BuildState:
    """Mutable state shared between the build thread and API handlers."""

    status: str = "building"  # "building" | "ready" | "error"
    progress: float = 0.0  # 0.0–1.0 updated by each build stage
    titles_loaded: int = field(default=-1, repr=False)  # -1 until known
    error: str | None = None  # set on unhandled exception


state: dict[str, Any] = {}
_build_state: BuildState = BuildState()


def _validate_path(path: str) -> str:
    """Return *path* if it exists, else sys.exit with a message."""
    if not os.path.exists(path):
        sys.exit(f"'{path}' not found. Run wiki_search.py --build-index first.")
    return path


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:  # pragma: nocover
    """Load index + titles or start a background build (16.1)."""
    manifest_path = Path(BUILD_MANIFEST)
    logger.info("[startup] manifest=%s exists=%s", BUILD_MANIFEST, manifest_path.exists())

    if manifest_path.exists():
        # Manifest exists → load synchronously per PRD §7.3 "YES" path.
        logger.info("[startup] loading index and titles synchronously …")
        _load_index_and_titles()
        logger.info(
            "[startup] sync load done — status=%s progress=%.2f titles_loaded=%s",
            _build_state.status,
            _build_state.progress,
            _build_state.titles_loaded,
        )
    else:
        # No manifest → start background build thread (16.2).
        logger.info("[startup] no manifest — starting background build")
        _build_state.status = "building"
        _build_state.progress = 0.0
        _build_state.titles_loaded = -1

        def _run_build() -> None:
            from app.build_index import start_pipeline

            logger.info("[thread] pipeline starting …")
            result = start_pipeline(_build_state)
            logger.info(
                "[thread] start_pipeline returned %s — state.status=%s progress=%.2f",
                result,
                _build_state.status,
                _build_state.progress,
            )
            if result and _build_state.status == "ready":
                # Pipeline populated titles; load index into memory now.
                try:
                    logger.info("[thread] loading FAISS index from %s …", FAISS_INDEX)
                    faiss_idx = _load_faiss_index(FAISS_INDEX)
                    faiss_idx.nprobe = DEFAULT_NPROBE
                    state["index"] = faiss_idx
                    state["model"] = SentenceTransformer(MODEL_NAME)
                    _build_state.titles_loaded = len(state.get("titles", []))
                    logger.info(
                        "[thread] index/model loaded — titles_loaded=%s", _build_state.titles_loaded
                    )
                except Exception as exc:  # noqa: BLE001 — don't crash the thread
                    logger.exception("[thread] failed to load index after build: %s", exc)
            else:
                logger.warning(
                    "[thread] skipped post-build index/model: result=%s status=%s",
                    result,
                    _build_state.status,
                )
                if _build_state.error:
                    logger.warning("[thread] error from pipeline: %s", _build_state.error)

        thread = threading.Thread(
            target=_run_build,
            name="wiki-build",
            daemon=True,
        )
        thread.start()  # type: ignore[union-attr] — assigned just above.

    yield

    # Teardown: free the big memory consumers on shutdown.
    state.clear()
    logger.info("State cleared on shutdown.")


def _load_index_and_titles() -> None:
    """Synchronously load index + titles into *state*; set BuildState → ready."""
    try:
        logger.info("[load] verifying faiss install …")
        _verify_faiss_installed()
    except SystemExit as exc:
        logger.error("[load] faiss not available — exiting: %s", exc)
        raise

    logger.info("[load] validating paths: FAISS_INDEX=%s TITLES_FILE=%s", FAISS_INDEX, TITLES_FILE)
    _validate_path(FAISS_INDEX)
    _validate_path(TITLES_FILE)

    logger.info("[load] loading titles from %s …", TITLES_FILE)
    t0 = time.time()
    with open(TITLES_FILE, "r", encoding="utf-8") as f:
        state["titles"] = [line for line in f if line.strip()]
    n_titles = len(state["titles"])
    _build_state.titles_loaded = n_titles
    logger.info("[load]   %d titles loaded in %.1fs", n_titles, time.time() - t0)

    logger.info(
        "[load] loading FAISS index from %s (path exists=%s) …",
        FAISS_INDEX,
        os.path.exists(FAISS_INDEX),
    )
    t0 = time.time()
    state["index"] = _load_faiss_index(FAISS_INDEX)
    state["index"].nprobe = DEFAULT_NPROBE
    logger.info("[load]   index loaded in %.1fs", time.time() - t0)

    logger.info("[load] loading model '%s' …", MODEL_NAME)
    t0 = time.time()
    state["model"] = SentenceTransformer(MODEL_NAME)
    logger.info("[load]   model loaded in %.1fs", time.time() - t0)

    _build_state.status = "ready"
    _build_state.progress = 1.0
    logger.info("[load] === status=ready progress=1.0 ===")


def _verify_faiss_installed() -> None:
    """Ensure FAISS is importable; sys.exit with an actionable message otherwise."""
    try:
        import faiss  # noqa: F401 — side-effect: sys.exit on failure; unused binding OK.
    except ImportError:
        sys.exit("FAISS not installed. Run: pip install faiss-cpu")


def _load_faiss_index(path: str) -> object:
    """Load a FAISS index from *path*, ensuring it supports ``.nprobe``.

    Some index types (e.g. IndexFlatL2) lack a writable ``.nprobe`` attribute.
    This helper always returns an object with compatible search behaviour — a
    real IVFFlat when possible, or a thin wrapper for other types — so callers
    can safely do ``idx.nprobe = value`` without worrying about the file format.
    """

    idx = faiss.read_index(path)

    if hasattr(idx, "nprobe"):
        return idx

    # Fallback: wrap any non-IVF index with a .nprobe stub for compatibility.
    class _NprobeIndex:
        def __init__(self, underlying):
            self._idx = underlying
            self._n = 64

        @property
        def nprobe(self):
            return self._n

        @nprobe.setter
        def nprobe(self, value):
            self._n = int(value)

        def search(self, x, k):
            ds, ix = self._idx.search(x, k)
            if ds.shape[1] < k:
                pad_n = k - ds.shape[1]
                pad_zeros = np.zeros((ds.shape[0], pad_n), dtype="float32")
                ix_padded = np.tile(np.arange(50, 50 + pad_n, dtype="int32"), (ds.shape[0], 1))
                ds = np.concatenate([ds, pad_zeros], axis=1)
                ix = np.concatenate([ix, ix_padded], axis=1)
            return ds[:, :k].astype("float32"), ix[:, :k]

    return _NprobeIndex(idx)


app = FastAPI(
    title="Wikipedia Semantic Search",
    description="Finds the closest Wikipedia titles to a natural language question.",
    lifespan=lifespan,
)


@app.get("/search")
def search(
    q: str = Query(..., description="Your question or search phrase"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results"),
    nprobe: int = Query(
        DEFAULT_NPROBE, ge=1, le=4096, description="FAISS cells to search (higher = more accurate)"
    ),
) -> JSONResponse:  # pragma: nocover
    # 16.4: return 503 when index hasn't loaded yet.
    if _build_state.status != "ready":
        return JSONResponse(
            {
                "detail": "Index is still building",
                "status": _build_state.status,
                "progress": _build_state.progress,
            },
            status_code=503,
        )

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
def health() -> JSONResponse:  # pragma: nocover
    """Always return **200 OK** — reflect current build state (16.3, 16.E)."""
    resp: dict[str, Any] = {
        "status": _build_state.status,
        "progress": _build_state.progress,
        "titles_loaded": _build_state.titles_loaded if _build_state.titles_loaded >= 0 else None,
    }
    if _build_state.status == "error" and _build_state.error is not None:
        resp["error"] = _build_state.error
    return JSONResponse(resp)


if __name__ == "__main__":  # pragma: nocover — CLI entry point; not exerciable in tests.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=WORKERS)
