"""Configuration — all runtime settings from environment variables (PRD §9)."""

import os

# ── Application logging (T21.5) ────────────────────────────────────────

# LOG_LEVEL env var: one of DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

# Name of the sentence-transformers model used to embed titles (e.g., 'all-MiniLM-L6-v2' on HF Hub)
MODEL_NAME: str = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")

# Embedding dimensionality in integers; must match the chosen
# model's output size (384 for MiniLM-L6-v2)
EMBED_DIM: int = int(os.environ.get("EMBED_DIM", "384"))

# Filename or path of the FAISS IVF index file on disk (loaded at startup into memory)
FAISS_INDEX: str = os.environ.get("FAISS_INDEX", "wiki_faiss.index")

# Filename or path of the Wikipedia titles list file (one title per line; loaded at startup)
TITLES_FILE: str = os.environ.get("TITLES_FILE", "wiki_titles.txt")

# Default number of nearest-neighbour results to return for /search when the caller omits *k*
DEFAULT_K: int = int(os.environ.get("DEFAULT_K", "5"))

# Default FAISS IVF _nprobe_ — how many inverted cells are examined
# per search (↔ accuracy/speed trade-off)
DEFAULT_NPROBE: int = int(os.environ.get("DEFAULT_NPROBE", "64"))

# TCP port the uvicorn server listens on inside the container
PORT: int = int(os.environ.get("PORT", "8000"))

# Number of uvicorn worker processes (single process for simplicity at this scale)
WORKERS: int = int(os.environ.get("WORKERS", "1"))

# ── Index-build env vars (PRD §9, T11) ────────────────────────────────

# URL of the latest Wikimedia Cirrussearch JSON dump. Leave empty to use
# the default which resolves to the latest English Wikipedia dump.
WIKI_DUMP_URL: str = (
    os.environ.get("WIKI_DUMP_URL", "").rstrip("/")
    or "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz"
)

# Batch size for title embedding during index build (T13)
BUILD_BATCH_SIZE: int = int(os.environ.get("BUILD_BATCH_SIZE", "512"))

# FAISS IVF *nlist* — number of clusters used during index training (T14)
BUILD_NLIST: int = int(os.environ.get("BUILD_NLIST", "4096"))

# Fraction of vectors sampled (randomly, deterministically via
# ``random.Random(42).sample``) to train the IVF quantizer before adding all.
BUILD_SAMPLE_FRAC: float = float(os.environ.get("BUILD_SAMPLE_FRAC", "0.1"))

# Whether to skip already-completed build stages on restart (T7.4). Accepts::
#   ``"true"``, ``"1"`` → ``True``   (also the default)
#   ``"false"``, ``"0"`` → ``False``
BUILD_RESUME: bool = os.environ.get("BUILD_RESUME", "true").lower() in ("true", "1")

# Sentinel filename written on successful index build; presence signals that a
# previously-built index is available and the build pipeline can skip to loading.
BUILD_MANIFEST: str = os.environ.get("BUILD_MANIFEST", "build_manifest.json")
