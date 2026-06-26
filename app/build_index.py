"""build_index.py — Wikipedia index build pipeline (stages 1-3).

Re-exports everything from ``app.build`` so every existing import survives.
"""  # ponytail: thin re-export stub; deletes easily if callers switch to app.build.

from app.build.download import (
    _count_titles_file,
    _fetch_stream,
    download_titles,
)
from app.build.embedding import (
    _load_model,
    generate_embeddings,
)
from app.build.faiss import build_faiss_index
from app.build.stages import STAGE_RANGES, start_pipeline
from app.build.state import BuildState, load_titles_from_file

__all__ = [
    "BuildState",
    "_count_titles_file",
    "_fetch_stream",
    "_load_model",
    "STAGE_RANGES",
    "build_faiss_index",
    "download_titles",
    "generate_embeddings",
    "load_titles_from_file",
    "start_pipeline",
]
