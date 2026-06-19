"""Configuration — all runtime settings from environment variables (PRD §9)."""

import os

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
