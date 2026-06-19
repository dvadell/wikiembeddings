"""
Wikipedia Semantic Search — Persistent Server
==============================================
Loads the FAISS index and sentence-transformer model once at startup,
then serves queries over HTTP. Each search takes ~10ms instead of ~10s.

Install:
    pip install sentence-transformers numpy faiss-cpu fastapi uvicorn

Run:
    python wiki_server.py

Then query from another terminal:
    curl "http://localhost:8000/search?q=How+do+plants+convert+sunlight"

    # pretty JSON
    curl -s "http://localhost:8000/search?q=photosynthesis" | python -m json.tool

    # change number of results
    curl "http://localhost:8000/search?q=black+holes&k=10"
"""

import os
import sys
import time
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# ── Config (must match wiki_search.py) ───────────────────────────────────────
MODEL_NAME  = "all-MiniLM-L6-v2"
EMBED_DIM   = 384
FAISS_INDEX = "wiki_faiss.index"
TITLES_FILE = "wiki_titles.txt"
DEFAULT_K   = 5
DEFAULT_NPROBE = 64
# ─────────────────────────────────────────────────────────────────────────────

# Global state — loaded once at startup
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load everything into memory before accepting requests."""
    try:
        import faiss
    except ImportError:
        sys.exit("FAISS not installed. Run: pip install faiss-cpu")

    from sentence_transformers import SentenceTransformer

    for path in [FAISS_INDEX, TITLES_FILE]:
        if not os.path.exists(path):
            sys.exit(f"'{path}' not found. Run wiki_search.py --build-index first.")

    print("Loading titles …", flush=True)
    t0 = time.time()
    with open(TITLES_FILE, "r", encoding="utf-8") as f:
        state["titles"] = f.read().splitlines()
    print(f"  {len(state['titles']):,} titles loaded in {time.time()-t0:.1f}s")

    print("Loading FAISS index …", flush=True)
    t0 = time.time()
    state["index"] = faiss.read_index(FAISS_INDEX)
    state["index"].nprobe = DEFAULT_NPROBE
    print(f"  Index loaded in {time.time()-t0:.1f}s")

    print(f"Loading model '{MODEL_NAME}' …", flush=True)
    t0 = time.time()
    state["model"] = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print("\nServer ready. Listening on http://localhost:8000\n")
    yield
    state.clear()


app = FastAPI(
    title="Wikipedia Semantic Search",
    description="Finds the closest Wikipedia titles to a natural language question.",
    lifespan=lifespan,
)


@app.get("/search")
def search(
    q: str = Query(..., description="Your question or search phrase"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results"),
    nprobe: int = Query(DEFAULT_NPROBE, ge=1, le=4096,
                        description="FAISS cells to search (higher = more accurate)"),
):
    model  = state["model"]
    index  = state["index"]
    titles = state["titles"]

    index.nprobe = nprobe

    t0    = time.time()
    q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
    q_emb /= np.linalg.norm(q_emb, keepdims=True)

    scores, indices = index.search(q_emb, k)
    elapsed_ms = (time.time() - t0) * 1000

    results = [
        {"rank": i + 1, "title": titles[idx], "score": round(float(score), 4)}
        for i, (score, idx) in enumerate(zip(scores[0], indices[0]))
    ]

    return JSONResponse({
        "query":      q,
        "results":    results,
        "elapsed_ms": round(elapsed_ms, 1),
    })


@app.get("/health")
def health():
    return {"status": "ok", "titles_loaded": len(state.get("titles", []))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
