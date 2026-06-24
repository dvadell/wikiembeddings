"""test_build_embed.py — generate_embeddings() stage (T13).

Mocked ``_load_model`` returns a tiny stub with ``.encode`` that produces
deterministic float32 vectors of the right shape.  *Zero* real model or network
access is needed.

Acceptance criteria covered
============================
13.A  Output is shape ``(n_titles, 384)`` dtype ``float32``
13.B  Each row is L2-normalised (``np.linalg.norm(arr[i]) ≈ 1.0``, tol 1e-5)
13.C  resume=True + matching file → encode mock not called; ``progress_cb(1.0)`` called
13.D  Partial file (wrong size) bypasses resume skip, re-encodes
13.E  Coverage on embedding stage ≥ 95%

"""

from __future__ import annotations

import math  # noqa: F401 — used in test progress monotonic check.
from pathlib import Path
from unittest import mock

import numpy as np

import app.build_index as m

# ── Helpers -------------------------------------------------------------------- #


def _make_stub_model(embed_dim: int = 384):
    """Return a stub ``model`` whose ``.encode`` produces deterministic float32 rows."""

    def encode(texts, normalize_embeddings=False, show_progress_bar=False):  # type: ignore[assignment] — intentionally loosely typed for the mock signature.
        n = len(texts)
        arr = np.random.RandomState(42).randn(n, embed_dim).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
            arr = arr / norms  # L2-normalise.
        return arr

    stub = mock.MagicMock()
    stub.encode.side_effect = encode  # type: ignore[attr-defined] — MagicMock assigns side_effect dynamically; mypy can't infer it.
    return stub


# ── Test data ------------------------------------------------------------------ #
#   8 real titles (one blank line among them) → expect count=8                       #

_TITLES_LINES = [
    "A",
    "",  # blank → skipped by _count_titles_file.
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
]


def _write_titles(tmp_path: Path, lines: list[str]) -> Path:
    """Write *lines* to a temp file and return its path."""
    p = tmp_path / "titles.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ── 13.A  --  output shape (n_titles, 384) dtype float32 -------------------- #


class TestOutputShape:
    def test_shape_and_dtype(self, tmp_path: Path):
        titles = _write_titles(tmp_path, _TITLES_LINES)

        embeddings = tmp_path / "embeddings.npy"
        stub = _make_stub_model(embed_dim=384)
        with mock.patch.object(m, "_load_model", return_value=stub):
            count = m.generate_embeddings(
                titles_path=titles,
                embeddings_path=embeddings,
                model_name="unit-test-model",
                batch_size=512,
                resume=False,
                progress_cb=mock.Mock(),
            )

        assert count == len(_TITLES_LINES) - 1  # blank line skipped.
        mmap = np.memmap(str(embeddings), dtype="float32", mode="r", shape=(count, 384))
        assert list(mmap.shape) == [len(_TITLES_LINES) - 1, 384]
        assert mmap.dtype == np.float32


# ── 13.B  --  each row is L2-normalised --------------------------------------- #


class TestL2Normalisation:
    def test_all_rows_normalised(self, tmp_path: Path):
        titles = _write_titles(tmp_path, ["A", "B", "C"])

        embeddings = tmp_path / "embed.npy"
        stub = _make_stub_model()  # already normalises inside .encode.
        with mock.patch.object(m, "_load_model", return_value=stub):
            m.generate_embeddings(titles, embeddings, "x", 2, False, mock.Mock())

        mmap = np.memmap(str(embeddings), dtype="float32", mode="r", shape=(3, 384))
        for i in range(3):
            norm = float(np.linalg.norm(mmap[i]))  # noqa: FURB154 — intentional row iteration to verify L2 norms.
            assert abs(norm - 1.0) < 1e-5, f"row {i} norm={norm}"


# ── 13.C  --  resume=True + matching file → skip ------------------------------ #


class TestResume:
    def test_skips_when_file_matches(self, tmp_path: Path):
        titles = _write_titles(tmp_path, ["A", "B"])

        # Pre-create a file with the exact correct byte size.
        embeddings = tmp_path / "embed.npy"
        expected_bytes = 2 * 384 * np.dtype("float32").itemsize
        embeddings.write_bytes(b"\x00" * int(expected_bytes))  # type: ignore[arg-type] — all-zero bytes; we only check size.

        progress_cb = mock.Mock()

        with mock.patch.object(m, "_load_model", autospec=True) as lm:
            count = m.generate_embeddings(
                titles_path=titles,
                embeddings_path=embeddings,
                model_name="x",
                batch_size=512,
                resume=True,
                progress_cb=progress_cb,
            )

        lm.assert_not_called()  # model was never loaded.
        assert count == 2
        progress_cb.assert_called_once_with(1.0)


# ── 13.D  --  partial file (wrong size) bypasses resume ---------------------- #


class TestPartialFile:
    def test_wrong_size_re_encodes(self, tmp_path: Path):
        titles = _write_titles(tmp_path, ["A", "B"])

        embeddings = tmp_path / "embed.npy"
        # Write a file that is too small — size differs from expected so resuming is bypassed.
        embeddings.write_bytes(b"\x00" * 100)  # type: ignore[arg-type, FURB]

        progress_cb = mock.Mock()
        with mock.patch.object(m, "_load_model", return_value=_make_stub_model()) as lm:
            m.generate_embeddings(
                titles_path=titles,
                embeddings_path=embeddings,
                model_name="x",
                batch_size=512,
                resume=True,  # resume is True but file size differs.
                progress_cb=progress_cb,
            )

        assert lm.call_count == 1  # encode was called since the file was incomplete.


# ── Edge cases --------------------------------------------------------------- #


class TestEdgeCases:
    def test_empty_titles(self, tmp_path: Path):
        """Empty titles → returns 0, progress_cb(1.0)."""
        titles = _write_titles(tmp_path, [])
        embeddings = tmp_path / "embed.npy"
        count = m.generate_embeddings(titles, embeddings, "x", 512, False, mock.Mock())
        assert count == 0

    def test_single_title(self, tmp_path: Path):
        titles = _write_titles(tmp_path, ["Solo"])
        embeddings = tmp_path / "embed.npy"

        with mock.patch.object(m, "_load_model", return_value=_make_stub_model()):
            count = m.generate_embeddings(titles, embeddings, "x", 1, False, mock.Mock())

        assert count == 1


# ── Progress tracking (branch coverage) --------------------------------------- #


class TestProgress:
    def test_progress_monotonic(self, tmp_path: Path):
        titles = _write_titles(tmp_path, ["A"] * 10)  # small so batch_size=2 → multiple batches.
        embeddings = tmp_path / "embed.npy"

        values: list[float] = []
        with mock.patch.object(m, "_load_model", return_value=_make_stub_model()):
            m.generate_embeddings(
                titles,
                embeddings,
                "x",
                batch_size=2,  # 10 / 2 = 5 batches → plenty of progress callbacks.
                resume=False,
                progress_cb=values.append,  # type: ignore[arg-type] — intentionally loosely typed here.
            )

        assert len(values) >= 3  # at least initial 0.0 + several batch callbacks.
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], f"not monotonic: {values}"
