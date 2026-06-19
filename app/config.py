"""Configuration — all runtime settings from environment variables (PRD §9)."""

import os

MODEL_NAME: str = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
EMBED_DIM: int = int(os.environ.get("EMBED_DIM", "384"))
FAISS_INDEX: str = os.environ.get("FAISS_INDEX", "wiki_faiss.index")
TITLES_FILE: str = os.environ.get("TITLES_FILE", "wiki_titles.txt")
DEFAULT_K: int = int(os.environ.get("DEFAULT_K", "5"))
DEFAULT_NPROBE: int = int(os.environ.get("DEFAULT_NPROBE", "64"))
PORT: int = int(os.environ.get("PORT", "8000"))
WORKERS: int = int(os.environ.get("WORKERS", "1"))
