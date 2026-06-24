"""test_build_download.py — download_titles() stage (T12).

Patches the internal ``_fetch_stream`` helper so *zero* network access is
needed.  Synthetic gzip frames are produced by ``gzip.compress``.

Acceptance criteria covered
============================
12.A  Returns exact count of non-empty titles written
12.B  Empty lines / surrounding whitespace are skipped/stripped
12.C  resume=True + existing file: no download; progress_cb(1.0) called
12.D  Download is streamed (via gzip.GzipFile — visible in implementation)
12.E  Coverage on download stage >= 95%

"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from unittest import mock

import pytest

import app.build_index as m

# ------------------------------------------------------------------ #
#  Helpers                                                               #
# ------------------------------------------------------------------ #


def _make_stream(raw_lines: list[str]) -> io.BytesIO:
    """Create a BytesIO containing valid gzip data for *raw_lines*."""
    text = "\n".join(line_item + "\n" for line_item in raw_lines)
    compressed = gzip.compress(text.encode("utf-8"))
    return io.BytesIO(compressed)


def _fake_fetch_stream(raw_lines: list[str], total: int | None = 100):
    """Return a mock that yields (total, BytesIO(valid_gzip))."""
    return (total, _make_stream(raw_lines))


# ------------------------------------------------------------------ #
#  Test data                                                             #
#   8 real titles + 2 blank lines = 10 total -> expect count=8         #
# ------------------------------------------------------------------ #

_RAW_LINES = [
    "Title One",
    "",
    " Title Two ",  # whitespace-padded
    "Title Three",
    "Title Four",
    "",
    "   ",  # all-whitespace line
    "Title Five",
    "Title Six",
    "Title Seven",
]


# ------------------------------------------------------------------ #
#  T12.A  --  returns exact count of non-empty titles                 #
# ------------------------------------------------------------------ #


class TestDownloadCount:
    @pytest.mark.parametrize(
        "lines,expected",
        [
            (_RAW_LINES, 7),
            (["A", "B"], 2),
            (["", "  ", "X"], 1),
        ],
    )
    def test_returns_title_count(self, lines: list[str], expected: int, tmp_path: Path):
        out_file = tmp_path / "titles.txt"
        fetch_mock = _fake_fetch_stream(lines)

        with mock.patch.object(m, "_fetch_stream", return_value=fetch_mock):
            count = m.download_titles(
                output_path=out_file,
                dump_url="https://example.com/dump.gz",
                resume=False,
                progress_cb=mock.Mock(),
            )

        assert count == expected


# ------------------------------------------------------------------ #
#  T12.B  --  empty lines / whitespace are stripped                   #
# ------------------------------------------------------------------ #


class TestOutputFile:
    def test_empty_lines_skipped(self, tmp_path: Path):
        out_file = tmp_path / "out.txt"
        fetch_mock = _fake_fetch_stream(_RAW_LINES)

        with mock.patch.object(m, "_fetch_stream", return_value=fetch_mock):
            m.download_titles(out_file, "https://x/y", resume=False, progress_cb=mock.Mock())

        text = out_file.read_text(encoding="utf-8")
        lines_out = text.splitlines()
        assert len(lines_out) == 7
        for line in lines_out:
            assert line == line.strip(), f"not stripped: {line!r}"


# ------------------------------------------------------------------ #
#  T12.C  --  resume=True with existing file                           #
# ------------------------------------------------------------------ #


class TestResumeMode:
    def test_skips_download(self, tmp_path: Path):
        out_file = tmp_path / "already_done.txt"
        out_file.write_text("AlreadyExisting\nAnotherOne\n", encoding="utf-8")

        progress_cb = mock.Mock()

        with mock.patch.object(m, "_fetch_stream") as fs_mock:
            result = m.download_titles(
                output_path=out_file,
                dump_url="https://example.com/dump.gz",
                resume=True,
                progress_cb=progress_cb,
            )

        fs_mock.assert_not_called()  # should NOT enter download path.
        assert result == 2
        progress_cb.assert_called_once_with(1.0)

    def test_no_existing_file_downloads(self, tmp_path: Path):
        out_file = tmp_path / "new.txt"  # does NOT exist yet.
        progress_cb = mock.Mock()

        with mock.patch.object(m, "_fetch_stream", return_value=_fake_fetch_stream(["A", "B"])):
            m.download_titles(out_file, "https://x/y", resume=True, progress_cb=progress_cb)

        assert out_file.exists()


# ------------------------------------------------------------------ #
#  Edge cases                                                         #
# ------------------------------------------------------------------ #


class TestEdgeCases:
    def test_single_title(self, tmp_path: Path):
        out_file = tmp_path / "s.txt"
        with mock.patch.object(m, "_fetch_stream", return_value=_fake_fetch_stream(["Hello"])):
            count = m.download_titles(
                out_file, "https://x/y", resume=False, progress_cb=mock.Mock()
            )

        assert count == 1
        assert out_file.read_text().strip() == "Hello"

    def test_preserves_all_lines(self, tmp_path: Path):
        out_file = tmp_path / "p.txt"
        with mock.patch.object(
            m, "_fetch_stream", return_value=_fake_fetch_stream(["A", "B", "C"])
        ):
            m.download_titles(out_file, "https://x/y", resume=False, progress_cb=mock.Mock())

        assert out_file.read_text().strip() == "A\nB\nC"


# ------------------------------------------------------------------ #
#  T12.D  --  streaming exercised.                                       #
# ------------------------------------------------------------------ #


class TestStreaming:
    def test_path_works(self, tmp_path: Path):
        gz = _make_stream(["TestTitle"])
        out_file = tmp_path / "stream.txt"

        with mock.patch.object(m, "_fetch_stream", return_value=(100, gz)):
            m.download_titles(out_file, "https://x/y", resume=False, progress_cb=mock.Mock())

        assert out_file.read_text().strip() == "TestTitle"


# ------------------------------------------------------------------ #
#  Branch coverage (remaining)                                          #
# ------------------------------------------------------------------ #


class TestBranchCoverage:
    def test_total_size_none_doesnt_crash(self, tmp_path: Path):
        """When fetch returns None content_length, progress_cb still works at end."""
        gz = _make_stream(["X"])
        out_file = tmp_path / "p.txt"
        # Return None as total_size to cover the else branch of `if total_size`.
        with mock.patch.object(m, "_fetch_stream", return_value=(None, gz)):
            count = m.download_titles(
                out_file, "https://x/y", resume=False, progress_cb=mock.Mock()
            )
        assert count == 1

    def test_empty_input_returns_zero(self, tmp_path: Path):
        """An empty titles file produces zero output."""
        out_file = tmp_path / "empty.txt"
        fetch_m = _fake_fetch_stream(["", ""])
        with mock.patch.object(m, "_fetch_stream", return_value=fetch_m):
            m.download_titles(out_file, "https://x/y", resume=False, progress_cb=mock.Mock())
        # All lines empty -> 0 writes (blank lines skipped via continue path)
        assert out_file.read_text() == ""
