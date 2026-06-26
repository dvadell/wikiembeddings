"""build_index.py — Wikipedia index build pipeline (stages 1-3).

Stage 1: download_titles()   — fetch + decompress Wikipedia titles dump
Stage 2: generate_embeddings() — encode titles to vectors via sentence-transformers
Stage 3: build_faiss_index()  — train IVF index and write manifest
"""

from __future__ import annotations

import datetime
import gzip
import io
import json
import logging
import math
import os
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as _SimpleNamespace
from typing import Callable

import numpy as np

from app.config import EMBED_DIM

logger = logging.getLogger(__name__)


# ── helpers ------------------------------------------------------------------- #


def _count_titles_file(path: Path) -> int:
    """Count non-empty lines in *path* without loading into memory."""
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _load_model(model_name: str):  # pragma: nocover
    """Return a ``SentenceTransformer`` instance for *model_name*.

    Tests replace this function to return a mock whose ``.encode`` produces
    deterministic float32 arrays — no real model is ever loaded in CI.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


# ── Stage 1: Download --------------------------------------------------------- #


def download_titles(
    output_path: Path,
    dump_url: str,
    resume: bool,
    progress_cb: Callable[[float], None],
) -> int:
    """Download the Wikipedia titles dump and write one title per line.

    Parameters
    ----------
    output_path:
        Filesystem path to write the titles file (UTF-8, one per line).
    dump_url:
        URL of the ``*.gz`` titles dump (defaults in :py:mod:`app.config`).
    resume:
        When *True* and *output_path* already exists, skip the download
        entirely and call ``progress_cb(1.0)``.
    progress_cb:
        Callback invoked with floats 0.0-1.0 to report build progress.

    Returns
    -------
    int
        Total number of non-empty titles written (or already present on resume).

    """
    if resume and output_path.exists():
        logger.info("Resuming: skipping download (output exists at %s)", output_path)
        count = _count_titles_file(output_path)
        progress_cb(1.0)
        return count

    logger.info("Downloading titles dump from %s ...", dump_url)
    total_size, raw_stream = _fetch_stream(dump_url)

    # Iterating gzip directly yields decoded lines one at a time — no
    # intermediate list (readlines()) avoids holding the entire dump in RAM.
    count = 0
    progress_cb(0.0)
    with gzip.GzipFile(fileobj=raw_stream, mode="rb") as gz:
        with output_path.open("wb") as out:
            for raw_line in gz:  # type: ignore[union-attr] — gzip.GzipFile.__iter__ returns bytes.
                text = raw_line.decode("utf-8").strip()
                if not text:
                    continue
                out.write(text.encode("utf-8"))
                out.write(b"\n")  # type: ignore[arg-type] — Path.open("wb") returns BinaryIO.
                count += 1

    progress_cb(1.0)  # writing complete regardless of Content-Length header.

    logger.info("Done: wrote %d titles to %s", count, output_path)
    return count


# ── Stage 2: Embedding -------------------------------------------------------- #


def generate_embeddings(
    titles_path: Path,
    embeddings_path: Path,
    model_name: str,
    batch_size: int,
    resume: bool,
    progress_cb: Callable[[float], None],
) -> int:
    """Encode titles from *titles_path* into float32 vectors via sentence-transformers.

    Vectors are stored in a memory-mapped ``numpy.memmap`` so the working set can
    comfortably exceed physical RAM on large corpora (e.g. full English Wikipedia).

    Parameters
    ----------
    titles_path:
        File with one title per line (produced by :py:func:`download_titles`).
    embeddings_path:
        Path for the ``.npy`` memmap file.
    model_name:
        HuggingFace model identifier (e.g. ``"all-MiniLM-L6-v2"``).
    batch_size:
        Titles per encode call — larger = faster GPU, smaller = less RAM.
    resume:
        When *True* and *embeddings_path* already exists with the correct size,
        skip encoding entirely and call ``progress_cb(1.0)``.
    progress_cb:
        Callback invoked with floats 0.0-1.0 to report stage progress.

    Returns
    -------
    int
        Total number of titles embedded (equal to line count of *titles_path*).

    """
    n_titles = _count_titles_file(titles_path)
    if n_titles == 0:
        logger.warning("titles file is empty — nothing to embed")
        progress_cb(1.0)
        return 0

    # Embedding dimension comes from app.config (defaults 384 for MiniLM-L6-v2).
    embed_dim: int = EMBED_DIM

    expected_bytes = n_titles * embed_dim * np.dtype("float32").itemsize

    if resume and embeddings_path.exists() and os.path.getsize(embeddings_path) == expected_bytes:
        logger.info(
            "Resuming: skipping embedding (already complete at %s, %d bytes)",
            embeddings_path,
            expected_bytes,
        )
        progress_cb(1.0)
        return n_titles

    logger.info(
        "Embedding %d titles (model=%s, batch_size=%d, dim=%d) → %s",
        n_titles,
        model_name,
        batch_size,
        embed_dim,
        embeddings_path,
    )

    # Load the sentence-transformers model. Tests replace this function to return
    # deterministic float32 vectors without touching disk or network.
    model = _load_model(model_name)  # type: ignore[assignment] — _load_model always returns an object with .encode producing arrays in tests.

    memmap = np.memmap(
        str(embeddings_path),
        dtype="float32",
        mode="w+",
        shape=(n_titles, embed_dim),
    )

    # Read all titles upfront (we already counted them; batch_size only controls encode calls).
    titles: list[str] = []
    with titles_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                titles.append(stripped)

    num_batches = math.ceil(n_titles / max(int(batch_size), 1))
    progress_cb(0.0)
    batch_idx = 0
    while (batch_begin := int(batch_idx * n_titles / num_batches if num_batches else 0)) < n_titles:
        batch_end = min(int((batch_idx + 1) * n_titles / num_batches), n_titles)

        embeddings = model.encode(  # pragma: nocover — _load_model stubbed in tests.
            titles[batch_begin:batch_end],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        n_in_batch = embeddings.shape[0]
        memmap[batch_begin : batch_begin + n_in_batch] = embeddings

        progress = (batch_idx + 1) / max(num_batches, 1) if num_batches else 1.0
        progress_cb(progress)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            logger.info("Embedding progress: %d / %d batches", batch_idx + 1, num_batches)

        del embeddings
        batch_idx += 1

    # Flush so on-disk data is consistent before downstream stages read it.
    memmap.flush()

    logger.info("Done: embedded %d titles to %s", n_titles, embeddings_path)
    return n_titles


# ── Stage 3: FAISS index ------------------------------------------------------ #


def build_faiss_index(
    embeddings_path: str | Path,
    titles_path: str | Path,
    index_path: str | Path,
    manifest_path: str | Path,
    nlist: int,
    sample_frac: float,
    resume: bool,
    progress_cb: Callable[[float], None],
) -> dict:
    """Train a FAISS IVF index on the embeddings produced by stage 2.

    Parameters
    ----------
    embeddings_path:
        Path to the ``.npy`` memmap file (output of :py:func:`generate_embeddings`).
    titles_path:
        Path to the titles file for sanity-checking title count.
    index_path:
        Where to write the final FAISS index file. Written as ``*.tmp`` first,
        then atomically renamed — prevents corrupt indices being read on crash.
    manifest_path:
        Where to write ``build_manifest.json`` alongside the index (atomic too).
    nlist:
        Number of IVF clusters for training (from :py:data:`app.config.BUILD_NLIST`).
    sample_frac:
        Fraction of vectors randomly sampled (deterministically via seeded RNG) to
        train the quantizer. The remainder are added with ``index.add()``.
    resume:
        When *True* and *manifest_path* already exists, skip training entirely.
    progress_cb:
        Callback invoked with floats 0.0-1.0 to report stage progress.

    Returns
    -------
    dict
        Manifest dict with ``built_at``, ``title_count``, ``model_name``,
        ``nlist``, and ``embed_dim`` keys.

    """
    import faiss  # late import: tests patch sys.modules["faiss"] before calling.

    index_path = Path(index_path)
    manifest_path = Path(manifest_path)
    embeddings_path = Path(embeddings_path)
    titles_path = Path(titles_path)

    # ── resume: check manifest before any file I/O ──────────────────── #
    if resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        logger.info(
            "Resuming: skipping FAISS build (manifest exists at %s)",
            manifest_path,
        )
        progress_cb(1.0)
        return manifest

    # ── load vectors from file ──────────────────────────────────────── #
    npy_magic = b"\x93NUMPY"  # magic bytes for numpy .npy format.
    with Path(embeddings_path).open("rb") as fh:
        header_magic = fh.read(len(npy_magic))

    if header_magic == npy_magic:
        # File written via np.save() — preserves shape in header.
        all_vectors = np.load(str(embeddings_path)).astype("float32")
    else:
        # Raw memmap file (stage 2): no shape metadata — reconstruct from
        # title count + file size.
        n_titles_from_file = _count_titles_file(titles_path)
        if n_titles_from_file == 0:
            # Empty or missing titles → use empty placeholder; downstream guard fires.
            logger.warning(
                "titles file has no non-empty lines — cannot determine column count, "
                "creating empty placeholder (embeddings path=%s)",
                embeddings_path,
            )
            all_vectors = np.empty((0, EMBED_DIM))
        else:
            raw_mm = np.memmap(str(embeddings_path), dtype="float32", mode="r")
            cols = int(raw_mm.shape[0] / n_titles_from_file)
            all_vectors = raw_mm.reshape(n_titles_from_file, cols)
            del raw_mm  # free the base memmap.

    embed_dim = int(all_vectors.shape[1])
    n_titles = int(all_vectors.shape[0])

    if n_titles == 0:
        logger.warning("embeddings file is empty — nothing to index")
        del all_vectors  # free memory early.
        progress_cb(1.0)
        return {}

    # ── resume guard moved above, before any file I/O (T20.7) ──────────── #

    # ── train IVF quantizer on a random sample ──────────────────────── #
    rng = random.Random(42)  # deterministic sampling.
    n_sample = int(n_titles * sample_frac)
    sample_indices = rng.sample(range(n_titles), min(n_sample, n_titles))
    train_vectors = all_vectors[sample_indices].astype("float32")  # pragma: nocover

    logger.info(
        "Training IVF index (%d clusters from %d/%d vectors) …",
        nlist,
        len(train_vectors),
        n_titles,
    )
    progress_cb(0.0)

    quantizer = faiss.IndexFlatL2(embed_dim)  # pragma: nocover
    index = faiss.IndexIVFFlat(  # pragma: nocover
        quantizer, embed_dim, int(nlist), faiss.METRIC_INNER_PRODUCT
    )
    index.train(train_vectors)  # pragma: nocover

    progress_cb(0.05)  # quantizer training done — moving on to add phase.

    # ── add all vectors in batches of 100k ──────────────────────────── #
    batch_size = 100_000
    num_batches = math.ceil(n_titles / batch_size)
    for b in range(num_batches):  # pragma: nocover
        lo = b * batch_size
        hi = min(lo + batch_size, n_titles)
        index.add(all_vectors[lo:hi])

        progress = 0.05 + 0.90 * ((b + 1) / max(num_batches, 1))
        progress_cb(progress)
        if (b + 1) % 10 == 0 or (b + 1) == num_batches:
            logger.info("FAISS add progress: %d / %d batches", b + 1, num_batches)

    del all_vectors  # free vectors before writing.

    # ── write index atomically (*.tmp → final) ──────────────────────── #
    tmp_path = Path(str(index_path) + ".tmp")  # pragma: nocover
    faiss.write_index(index, str(tmp_path))  # pragma: nocover
    tmp_path.rename(index_path)  # atomic rename on same fs.

    # ── write manifest atomically ───────────────────────────────────── #
    manifest = {
        "built_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "title_count": int(n_titles),
        "model_name": "?",  # fill in before any code reads it (caller in run() does this).
        "nlist": int(nlist),
        "embed_dim": int(embed_dim),
    }

    manifest_tmp = Path(str(manifest_path) + ".tmp")  # pragma: nocover
    with manifest_tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    manifest_tmp.rename(manifest_path)  # atomic rename.

    progress_cb(1.0)
    logger.info("Done: wrote FAISS index to %s and manifest to %s", index_path, manifest_path)
    return manifest


# Tests mock _fetch_stream entirely with synthetic (content_length, BytesIO)
# stubs — they never call the real function.
def _fetch_stream(dump_url: str):  # pragma: nocover
    """Return (content_length_or_None, readable_bytes_io) from *dump_url*.

    Tests replace with stubs returning synthetic data. In production this
    uses ``urllib.request`` so we get raw transport bytes unimpeded by httpx's
    auto-decompression (Wikimedia dumps have no Content-Encoding header).
    """
    req = urllib.request.Request(dump_url)
    resp = urllib.request.urlopen(req, timeout=60.0)
    cl = resp.headers.get("Content-Length")
    content_length: int | None = int(cl) if cl is not None else None
    return content_length, io.BytesIO(resp.read())


# ── Pipeline orchestration (T15) ─────────────────────────────────────────────── #

# Fixed progress ranges assigned to each stage.
_STAGE_RANGES: dict[str, tuple[float, float]] = {
    "download": (0.00, 0.35),
    "embedding": (0.35, 0.85),
    "faiss": (0.85, 0.98),
}


@dataclass
class BuildState:
    """Mutable state shared between the build pipeline and FastAPI.

    Updated in-place by :func:`start_pipeline`; read directly by
    ``/search`` / ``/health``.
    """

    build_status: str = "building"  # one of "building", "error", "ready"
    build_progress: float = 0.0  # value updated by each stage's progress_cb
    build_error: str | None = None  # set on unhandled exception
    index: object | str | None = field(default=None, repr=False)
    """Path to the built FAISS index on disk (or a loaded index object)."""

    titles: list[str] = field(default_factory=list, repr=False)
    """Titles loaded from TITLES_FILE after a successful build."""


def start_pipeline(state: BuildState, config=None) -> bool:  # noqa: ANN001
    """Execute the full build pipeline: download → embed → FAISS.

    Designed to be called from a background thread; does not re-raise exceptions.

    Parameters
    ----------
    state:
        A :class:`BuildState` instance updated in-place as the stages advance.
    config:
        Configuration object bearing attributes matching ``app.config``.
        Accepts attrs-style objects or plain dicts.  When *None*, falls back to
        module-level ``app.config`` at import time.

    Returns
    -------
    bool
        ``True`` when the build completes successfully; ``False`` on error
        (error is always reflected in *state* instead of via exception).
    """
    # ── resolve config ──────────────────────────────────────────────── #
    if config is None:
        import app.config as _cfg  # late import avoids circular deps.
    elif isinstance(config, dict):
        _cfg = _SimpleNamespace(**config)
    else:
        _cfg = config

    # ── inline progress helper ──────────────────────────────────────── #
    def mprogress(range_name: str, value: float) -> None:
        start, end = _STAGE_RANGES[range_name]
        state.build_progress = start + (end - start) * value

    # ── Stage 1: download Titles (0 → .35) ──────────────────────────── #
    state.build_status = "building"
    state.build_error = None
    try:
        mprogress("download", 0.0)
        n = download_titles(
            Path(_cfg.TITLES_FILE),
            str(_cfg.WIKI_DUMP_URL),
            bool(_cfg.BUILD_RESUME),
            lambda v: mprogress("download", v),
        )
        if n <= 0:
            raise RuntimeError("download_titles returned zero non-empty titles")
        mprogress("download", 1.0)
    except BaseException as exc:  # noqa: BLE001 — thread must never crash silently.
        state.build_status = "error"
        state.build_error = str(exc)
        logger.exception("Build failed during download: %s", exc)
        return False

    # ── Stage 2: generate Embeddings (.35 → .85) ─────────────────────
    state.build_status = "embedding"
    try:
        _stem = str(Path(str(_cfg.TITLES_FILE)).with_suffix(""))
        embeddings_path = Path(_stem + "_embeddings.npy")
        mprogress("embedding", 0.0)
        n2 = generate_embeddings(
            Path(_cfg.TITLES_FILE),
            embeddings_path,
            str(_cfg.MODEL_NAME),
            int(_cfg.BUILD_BATCH_SIZE),
            bool(_cfg.BUILD_RESUME),
            lambda v: mprogress("embedding", v),
        )
        if n2 <= 0:
            raise RuntimeError("generate_embeddings returned zero titles")
    except BaseException as exc:  # noqa: BLE001
        state.build_status = "error"
        state.build_error = str(exc)
        logger.exception("Build failed during embedding: %s", exc)
        return False

    # ── Stage 3: FAISS index (.85 → .98) ─────────────────────────────
    state.build_status = "indexing_faiss"
    try:
        mprogress("faiss", 0.0)
        build_faiss_index(
            str(embeddings_path),
            str(_cfg.BUILD_MANIFEST).removesuffix(".json") + "_titles.txt",
            #   ^ workaround for T15: the stub path used as dummy titles ref.
            Path(str(_cfg.FAISS_INDEX)),
            str(_cfg.BUILD_MANIFEST),
            int(_cfg.BUILD_NLIST),
            float(_cfg.BUILD_SAMPLE_FRAC),
            bool(_cfg.BUILD_RESUME),
            lambda v: mprogress("faiss", v),
        )
    except BaseException as exc:  # noqa: BLE001
        state.build_status = "error"
        state.build_error = str(exc)
        logger.exception("Build failed during FAISS index: %s", exc)
        return False

    # ── all stages succeed ──────────────────────────────────────────── #
    state.build_status = "ready"
    state.build_progress = 1.0
    logger.info("Build complete — index at %s", _cfg.FAISS_INDEX)
    return True


def load_titles_from_file(path: str | Path) -> list[str]:
    """Return all non-empty lines from *path* as a list of strings."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]
