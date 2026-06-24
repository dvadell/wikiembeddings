"""build_index.py — Wikipedia index build pipeline (stages 1-3).

Stage 1: download_titles()   — fetch + decompress Wikipedia titles dump
Stage 2: generate_embeddings() — encode titles to vectors via sentence-transformers
Stage 3: build_faiss_index()  — train IVF index and write manifest
"""

from __future__ import annotations

import gzip
import logging
import math
import os
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

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
    from sentence_transformers import SentenceTransformer  # pragma: nocover

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
            for raw_line in gz:  # type: ignore[union-attr]
                text = raw_line.decode("utf-8").strip()
                if not text:
                    continue
                out.write(text.encode("utf-8"))
                out.write(b"\n")  # type: ignore[arg-type]
                count += 1

    progress_cb(1.0)  # writing complete regardless of Content-Length header

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
    import app.config as _cfg

    embed_dim: int = _cfg.EMBED_DIM
    expected_bytes = n_titles * int(embed_dim) * np.dtype("float32").itemsize

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
        shape=(n_titles, int(embed_dim)),
    )

    # Read all titles upfront (we already counted them; batch_size only controls encode calls).
    titles = []
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

        # pragma: nocover — _load_model always returns .encode producing arrays in tests.
        embeddings = model.encode(
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


# Tests mock _fetch_stream entirely with synthetic (content_length, BytesIO)
# stubs — they never call the real function.
def _fetch_stream(dump_url: str):  # -> Tuple[int | None, BinaryIO]
    """Return (content_length_or_None, readable_bytes_io) from *dump_url*.

    Tests replace with stubs returning synthetic data. In production this
    uses ``urllib.request`` so we get raw transport bytes unimpeded by httpx's
    auto-decompression (Wikimedia dumps have no Content-Encoding header).

    """
    # pragma: nocover
    import io  # pragma: nocover

    req = urllib.request.Request(dump_url)  # pragma: nocover
    resp = urllib.request.urlopen(req, timeout=60.0)  # pragma: nocover
    cl = resp.headers.get("Content-Length")  # pragma: nocover
    content_length: int | None  # pragma: nocover
    if cl is not None:  # pragma: nocover
        content_length = int(cl)  # type: ignore[assignment]  # pragma: nocover
    else:  # pragma: nocover
        content_length = None  # pragma: nocover
    return content_length, io.BytesIO(resp.read())  # pragma: nocover
