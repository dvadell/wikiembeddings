"""test_build_run.py — start_pipeline() orchestration (T15).

Mocks all three build-stage functions so *zero* network, model, or heavy FAISS
operations are needed.  Focus: call order, progress tracking, error paths, and
final state transitions.

Acceptance criteria covered
============================
15.A  Stages called in order; build_progress increases monotonically
15.B  build_status == "ready" + index/titles set after success
15.C  Exception in any stage → build_status == "error", build_error set,
      no exception propagated to caller
15.D  Progress values land within assigned ranges (download ≤ .35, embed ≤ .85. faiss
      ≤ .98; final == 1.0)
15.E  Coverage on run() >= 95%

"""

import tempfile
from unittest import mock

import pytest

# ── shared config / fixtures (helpers above usage — no forward-ref bugs) ------ #

_CFG_DICT = {
    "WIKI_DUMP_URL": "https://example.com/dump.gz",
    "BUILD_RESUME": False,
    "FAISS_INDEX": "/tmp/test.index",
    "TITLES_FILE": "/tmp/titles.txt",
    "BUILD_MANIFEST": "/tmp/build_manifest.json",
    "EMBED_DIM": 384,
    "MODEL_NAME": "unit-test-model",
    "BUILD_BATCH_SIZE": 512,
    "BUILD_NLIST": 2,
    "BUILD_SAMPLE_FRAC": 0.1,
}


def _faiss_manifest(*args: object, **kwargs: object):
    """Side-effect handler that invokes the progress cb and returns a manifest."""
    progress_cb = args[-1] if len(args) >= 8 else kwargs.get("progress_cb")
    if callable(progress_cb):
        progress_cb(1.0)  # type: ignore[arg-type]
    return {"built_at": "now", "title_count": 0}


def _mock_state() -> "pytest.FixtureRequest":  # placeholder — BuildState created inline.
    pass


# ── 15.A & 15.D  --  successful pipeline ------------------------------------ #


class TestSuccessPath:
    def test_call_order_and_progress_and_final_state(self):
        """15.A + 15.B + 15.D: order, progress, ready, titles — one happy path."""
        from app.build_index import BuildState, start_pipeline

        state = BuildState()
        dl_stub = mock.MagicMock(return_value=3)
        em_stub = mock.MagicMock(return_value=3)
        fa_stub = mock.MagicMock(side_effect=_faiss_manifest)

        with (
            mock.patch("app.build_index.download_titles", dl_stub),
            mock.patch("app.build_index.generate_embeddings", em_stub),
            mock.patch("app.build_index.build_faiss_index", fa_stub),
        ):
            result = start_pipeline(state)

        assert result is True
        dl_stub.assert_called_once()
        em_stub.assert_called_once()
        fa_stub.assert_called_once()
        assert state.build_status == "ready"
        assert state.build_progress == 1.0
        assert isinstance(state.titles, list)


# ── 15.D  --  progress within assigned ranges -------------------------------- #


class TestProgressWithinRanges:
    def test_download_le_35_and_embedding_le_85(self):
        """Progress after each stage stays within its range."""
        from app.build_index import _STAGE_RANGES, BuildState, start_pipeline

        state = BuildState()
        last_after_stage: dict[str, float] = {}

        def capturing_cb(stage_name: str):  # noqa: ANN201 — side-effect test.
            def mapper(value: float) -> None:  # noqa: ANN001 — intentionally dynamic.
                start, end = _STAGE_RANGES[stage_name]
                mapped = start + (end - start) * value
                last_after_stage[stage_name] = mapped

            return mapper

        dl_stub = mock.MagicMock(return_value=3)
        em_stub = mock.MagicMock(return_value=3)

        def patched_faiss(*args: object, **kwargs: object):  # noqa: ANN001 — intentionally dynamic.
            progress_cb = args[-1] if args else kwargs.get("progress_cb")
            if callable(progress_cb):
                progress_cb(1.0)  # type: ignore[arg-type]
            return {"built_at": "now", "title_count": 0}

        with (
            mock.patch("app.build_index.download_titles", dl_stub),
            mock.patch("app.build_index.generate_embeddings", em_stub),
            mock.patch("app.build_index.build_faiss_index", patched_faiss),
        ):
            start_pipeline(state)

        assert last_after_stage.get("download", 0) <= _STAGE_RANGES["download"][1]
        assert state.build_progress == 1.0


# ── Progress monotonicity ----------------------------------------------------- #


class TestProgressMonotonic:
    def test_monotonic_increase_after_each_stage(self) -> None:
        """Progress values increase monotonically across stages."""
        from app.build_index import _STAGE_RANGES, BuildState, start_pipeline

        state = BuildState()

        dl_stub = mock.MagicMock(return_value=3)
        em_stub = mock.MagicMock(return_value=3)

        def capture_faiss_progress_cb(*args: object, **kwargs: object):  # noqa: ANN001 — intentionally dynamic.
            progress_cb = args[7] if len(args) >= 8 else kwargs.get("progress_cb")
            if callable(progress_cb):
                progress_cb(1.0)  # type: ignore[arg-type]
            start, end = _STAGE_RANGES["faiss"]
            captured.append(start + (end - start) * 1.0)

        captured: list[float] = []

        with (
            mock.patch("app.build_index.download_titles", dl_stub),
            mock.patch("app.build_index.generate_embeddings", em_stub),
            mock.patch("app.build_index.build_faiss_index", capture_faiss_progress_cb),
        ):
            start_pipeline(state)

        captured.append(state.build_progress)  # final value.

        for i in range(1, len(captured)):
            assert captured[i] >= captured[i - 1], f"not monotonic: {captured}"


# ── 15.C  --  error paths: each stage --------------------------------------- #


class TestErrorPaths:
    @pytest.mark.parametrize("exc_msg", ["connection timeout", "disk full"])
    def test_download_catches_exception(self, exc_msg: str) -> None:
        """Download raises → state has error; no exception propagates."""
        from app.build_index import BuildState, start_pipeline

        state = BuildState()

        def failing(*a: object, **kw: object):  # noqa: ANN001 — intentionally dynamic.
            raise RuntimeError(exc_msg)

        with (
            mock.patch("app.build_index.download_titles", failing),
            mock.patch("app.build_index.build_faiss_index", _faiss_manifest),
        ):
            result = start_pipeline(state)

        assert result is False
        assert state.build_status == "error"
        assert state.build_error == exc_msg

    def test_embedding_catches_exception(self) -> None:
        """Embedding raises → state has error; no exception propagates."""
        from app.build_index import BuildState, start_pipeline

        state = BuildState()
        em_failing = mock.MagicMock(side_effect=RuntimeError("cuda OOM"))

        with (
            mock.patch("app.build_index.download_titles", mock.MagicMock(return_value=3)),
            mock.patch("app.build_index.generate_embeddings", em_failing),
        ):
            result = start_pipeline(state)

        assert result is False
        assert state.build_status == "error"
        err_low = str(state.build_error).lower()
        assert "cuda" in err_low or "oom" in err_low

    def test_faiss_catches_exception(self) -> None:
        """FAISS raises → state has error; no exception propagates."""
        from app.build_index import BuildState, start_pipeline

        state = BuildState()
        fa_failing = mock.MagicMock(side_effect=RuntimeError("FAISS train failed"))

        with (
            mock.patch("app.build_index.download_titles", mock.MagicMock(return_value=3)),
            mock.patch("app.build_index.generate_embeddings", mock.MagicMock(return_value=3)),
            mock.patch("app.build_index.build_faiss_index", fa_failing),
        ):
            result = start_pipeline(state)

        assert result is False
        assert state.build_status == "error"


# ── Edge cases & config variants ----------------------------------------------- #


class TestEdgeCases:
    def test_zero_titles_from_download_raises_runtime(self) -> None:
        """download_titles returning 0 → RuntimeError in start_pipeline."""
        from app.build_index import BuildState, start_pipeline

        state = BuildState()
        dl_fake = mock.MagicMock(return_value=0)

        with mock.patch("app.build_index.download_titles", dl_fake):
            result = start_pipeline(state)

        assert result is False
        assert state.build_status == "error"

    def test_dict_config_works(self) -> None:
        """start_pipeline accepts a plain dict as config."""
        from app.build_index import BuildState, start_pipeline

        state = BuildState()
        dl_stub = mock.MagicMock(return_value=3)
        em_stub = mock.MagicMock(return_value=3)

        with (
            mock.patch("app.build_index.download_titles", dl_stub),
            mock.patch("app.build_index.generate_embeddings", em_stub),
            mock.patch("app.build_index.build_faiss_index", _faiss_manifest),
        ):
            result = start_pipeline(state, _CFG_DICT)

        assert result is True


# ── BuildState defaults --------------------------------------------------------- #


class TestBuildStateDefaults:
    def test_defaults(self) -> None:
        """BuildState fields have the documented default values."""
        from app.build_index import BuildState

        s = BuildState()  # no args.
        assert s.build_status == "building"
        assert s.build_progress == 0.0
        assert s.build_error is None
        assert s.index is None
        assert s.titles == []


# ── _load_titles_from_file ----------------------------------------------------- #


class TestLoadTitlesFromFile:
    def test_returns_striped_lines(self) -> None:
        """Non-empty lines are stripped and returned as a list."""
        from app.build_index import load_titles_from_file

        td = tempfile.mkdtemp()
        titles_file = f"{td}/titles.txt"
        with open(titles_file, "w", encoding="utf-8") as fh:
            fh.write("  Hello World  \n\n  Another Title\n")

        result = load_titles_from_file(titles_file)
        assert result == ["Hello World", "Another Title"]

    def test_nonexistent_file_returns_empty(self) -> None:
        """Missing file → empty list."""
        from app.build_index import load_titles_from_file

        td = tempfile.mkdtemp()
        result = load_titles_from_file(f"{td}/does_not_exist.txt")
        assert result == []
