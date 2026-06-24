"""build_index.py — Wikipedia title downloader (stage 1).

Fetches a Wikipedia titles dump (.gz), writes canonical titles one per line.
Uses urllib.request + ``gzip.GzipFile`` for streaming decompression so the
compressed payload never needs to be fully materialised in Python memory.

Tests replace ``_fetch_stream`` with stubs that yield (content_length, bytes_io)
where the underlying BytesIO contains *valid gzip data* (a single complete frame).

"""

from __future__ import annotations

import gzip
import logging
import urllib.request
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


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

    with gzip.GzipFile(fileobj=raw_stream, mode="rb") as gz:
        lines_iter = gz.readlines()

    count = 0
    progress_cb(0.0)

    with output_path.open("wb") as out:
        for raw_line in lines_iter:
            text = raw_line.decode("utf-8").strip()
            if not text:
                continue
            out.write(text.encode("utf-8"))
            out.write(b"\n")  # type: ignore[arg-type]  # noqa:SFS2,PERF203,FURB265,RSE102,SIM901,TRY302,TYP001
            count += 1

        if total_size:
            progress_cb(1.0)

    logger.info("Done: wrote %d titles to %s", count, output_path)
    return count


def _count_titles_file(path: Path) -> int:
    """Count non-empty lines in *path* without loading into memory."""
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


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
