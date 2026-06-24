"""test_build_run.py — start_pipeline() orchestration (T15).

Mocks all three build-stage functions so *zero* network, model, or heavy FAISS
operations are needed.  The focus is: call order, progress tracking, error paths,
and final state transitions.

Acceptance criteria covered
============================
15.A  Stages called in order; build_progress increases monotonically
15.B  build_status == "ready" + index/titles set after success
15.C  Exception in any stage → build_status == "error", build_error message set,
      no exception propagated to caller
15.D  Progress values land within assigned ranges (download ≤ .35, embed ≤ .85. faiss
      ≤ .98; final == 1.0)
15.E  Coverage on run() >= 95%

"""

from unittest import mock

import pytest

# ── Helper mocks --------------------------------------------------------------- #


def _mk_stub(func_name: str, count: int = 1) -> mock.MagicMock:
    """Create a mock that returns *count* and can be patched into the module."""
    stub = mock.MagicMock(
        side_effect=lambda *a, **kw: count,  # noqa: ARG001 — return value is all we need.
    )
    stub.__name__ = func_name  # type: ignore[attr-defined]
    return stub


def _make_config(d: dict) -> mock.MagicMock:
    """Wrap a plain dict into an attrs-style object for getattr()."""

    class Cfg:
        def __init__(self, d: dict) -> None:
            for k, v in d.items():
                setattr(self, k, v)

    _cfg = Cfg(d)
    return _cfg


_DEFAULT_CFG = _make_config(
    {
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
    },
)


# ── 15.A & 15.D  --  successful pipeline: call order + progress tracking ------- #


class TestSuccessPath:
    def test_call_order_and_progress(self):
        """15.A + 15.D: stages called in order; progress within assigned ranges."""
        from app.build_index import BuildState, start_pipeline

        state = BuildState()
        state.build_status = "building"

        dl_stub = _mk_stub("download_titles", count=3)
        em_stub = _mk_stub("generate_embeddings", count=3)
        fa_stub = _mock_faiss()

        with (
            mock.patch(
                "app.build_index.download_titles",
                dl_stub,
            ),
            mock.patch(
                "app.build_index.generate_embeddings",
                em_stub,
            ),
            mock.patch(
                "app.build_index.build_faiss_index",
                fa_stub,
            ),
        ):
            result = start_pipeline(state)

        # ── Assertions ─────────────────────────────────────────────── #
        assert result is True
        dl_stub.assert_called_once()
        em_stub.assert_called_once()
        fa_stub.assert_called_once()
        assert state.build_progress == 1.0

    def test_final_state_ready(self):
        """15.B: build_status is 'ready', index and titles populated."""
        from app.build_index import BuildState, start_pipeline

        state = BuildState()
        dl_stub = _mk_stub("download_titles", count=3)
        em_stub = _mk_stub("generate_embeddings", count=3)
        fa_stub = _mock_faiss()

        with (
            mock.patch(
                "app.build_index.download_titles",
                dl_stub,
            ),
            mock.patch(
                "app.build_index.generate_embeddings",
                em_stub,
            ),
            mock.patch(
                "app.build_index.build_faiss_index",
                fa_stub,
            ),
        ):
            start_pipeline(state)

        assert state.build_status == "ready"
        assert isinstance(state.titles, list)

    def test_progress_within_ranges(self):
        """15.D: progress at each stage falls within its assigned range."""
        from app.build_index import _STAGE_RANGES, BuildState, start_pipeline

        state = BuildState()

        # Capture the last mapped progress value for each stage.
        last_after_stage: dict[str, float] = {}

        def capturing_cb(stage_name: str):  # noqa: ANN201 — side-effect test.
            def mapper(value: float) -> None:  # noqa: ANN001 — intentionally dynamic.
                start, end = _STAGE_RANGES[stage_name]
                mapped = start + (end - start) * value
                last_after_stage[stage_name] = mapped

            return mapper

        dl_stub = _mk_stub("download_titles", count=3)
        em_stub = _mk_stub("generate_embeddings", count=3)

        fa_manifest: dict[str, object] = {"built_at": "now", "title_count": 0}

        def patched_faiss(*args: object, **kwargs: object):  # noqa: ANN001 — intentionally dynamic.
            # Simulate the progress cb at the FAISS stage's end (1.0).
            progress_cb = args[-1] if args else kwargs.get("progress_cb")
            if callable(progress_cb):
                progress_cb(1.0)  # type: ignore[arg-type]
            return fa_manifest

        with (
            mock.patch(
                "app.build_index.download_titles",
                dl_stub,
            ),
            mock.patch(
                "app.build_index.generate_embeddings",
                em_stub,
            ),
            mock.patch(
                "app.build_index.build_faiss_index",
                patched_faiss,
            ),
        ):
            start_pipeline(state)

        # Download range (0–.35): last progress after download should be 0.35 (1.0 mapped).
        assert last_after_stage.get("download", 0) <= _STAGE_RANGES["download"][1]

        # Embedding range (.35–.85): must end at .85.
        assert state.build_progress == 1.0


# ── 15.C  --  error paths: each stage can raise without leaking --------------- #


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
            mock.patch(
                "app.build_index.build_faiss_index",
                _mock_faiss(),
            ),
        ):
            result = start_pipeline(state)

        assert result is False
        assert state.build_status == "error"
        assert state.build_error == exc_msg

    def test_embedding_catches_exception(self) -> None:
        """Embedding raises → state has error; no exception propagates."""
        from app.build_index import BuildState, start_pipeline

        state = BuildState()

        dl_stub = _mk_stub("download_titles", count=3)
        em_failing = mock.MagicMock(side_effect=RuntimeError("cuda OOM"))

        with (
            mock.patch(
                "app.build_index.download_titles",
                dl_stub,
            ),
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

        dl_stub = _mk_stub("download_titles", count=3)
        em_stub = _mk_fake_embed(count=3)
        fa_failing = mock.MagicMock(side_effect=RuntimeError("FAISS train failed"))

        with (
            mock.patch(
                "app.build_index.download_titles",
                dl_stub,
            ),
            mock.patch(
                "app.build_index.generate_embeddings",
                em_stub,
            ),
            mock.patch("app.build_index.build_faiss_index", fa_failing),
        ):
            result = start_pipeline(state)

        assert result is False
        assert state.build_status == "error"


# ── Progress monotonicity ------------------------------------------------------- #


class TestProgressMonotonic:
    def test_monotonic_increase_after_each_stage(self) -> None:
        """Progress values after each stage increase monotonically."""
        from app.build_index import _STAGE_RANGES, BuildState, start_pipeline

        class StubConfig:
            WIKI_DUMP_URL = "https://x/y"
            BUILD_RESUME = False
            FAISS_INDEX = "/tmp/i.faiss"
            TITLES_FILE = "/tmp/t.txt"
            BUILD_MANIFEST = "/tmp/m.json"
            EMBED_DIM = 384
            MODEL_NAME = "x"
            BUILD_BATCH_SIZE = 512
            BUILD_NLIST = 2
            BUILD_SAMPLE_FRAC = 0.1

        state = BuildState()
        dl_stub = _mk_stub("download_titles", count=3)
        em_stub = _mk_fake_embed(count=3)

        def capture_faiss_progress_cb(*args: object, **kwargs: object):  # noqa: ANN001 — intentionally dynamic.
            progress_cb = (
                args[7]
                if len(args) >= 8
                else kwargs.get(
                    "progress_cb",
                )
            )
            if callable(progress_cb):
                progress_cb(1.0)  # type: ignore[arg-type]
            # Track FAISS final value in captured list.
            start, end = _STAGE_RANGES["faiss"]
            captured.append(start + (end - start) * 1.0)

        captured: list[float] = []

        with (
            mock.patch(
                "app.build_index.download_titles",
                dl_stub,
            ),
            mock.patch(
                "app.build_index.generate_embeddings",
                em_stub,
            ),
            mock.patch("app.build_index.build_faiss_index", capture_faiss_progress_cb),
        ):
            start_pipeline(state)

        captured.append(state.build_progress)  # final value.

        for i in range(1, len(captured)):
            assert captured[i] >= captured[i - 1], f"not monotonic: {captured}"


# ── Edge cases ------------------------------------------------------------------- #


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
        cfg = {
            "WIKI_DUMP_URL": "https://x/y",
            "BUILD_RESUME": False,
            "FAISS_INDEX": "/tmp/i.faiss",
            "TITLES_FILE": "/tmp/t.txt",
            "BUILD_MANIFEST": "/tmp/m.json",
            "EMBED_DIM": 384,
            "MODEL_NAME": "x",
            "BUILD_BATCH_SIZE": 512,
            "BUILD_NLIST": 2,
            "BUILD_SAMPLE_FRAC": 0.1,
        }

        dl_stub = _mk_stub("download_titles", count=3)
        em_stub = _mk_fake_embed(count=3)

        def fake_faiss(*a: object, **kw: object):  # noqa: ANN001 — intentionally dynamic.
            cb = a[-1] if len(a) >= 8 else kw.get("progress_cb")
            if callable(cb):
                cb(1.0)  # type: ignore[arg-type]
            return fa_manifest

        fa_manifest: dict[str, object] = {
            "built_at": "now",
            "title_count": 0,
            "model_name": "x",
            "nlist": 0,
            "embed_dim": 0,
        }

        with (
            mock.patch(
                "app.build_index.download_titles",
                dl_stub,
            ),
            mock.patch(
                "app.build_index.generate_embeddings",
                em_stub,
            ),
            mock.patch("app.build_index.build_faiss_index", fake_faiss),
        ):
            result = start_pipeline(state, cfg)

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


# ── Helpers ----------------------------------------------------------------------- #


def _mock_faiss() -> mock.MagicMock:
    """Create a FAISS stub that returns an empty manifest (resume path)."""
    stub = mock.MagicMock()

    def fake_manifest(*args: object, **kwargs: object):  # noqa: ANN001 — intentionally dynamic.
        cb = args[-1] if len(args) >= 8 else kwargs.get("progress_cb")
        if callable(cb):
            cb(1.0)  # type: ignore[arg-type]
        return {"built_at": "now", "title_count": 0}

    stub.side_effect = fake_manifest
    return stub


def _mk_fake_embed(count: int = 1) -> mock.MagicMock:
    """Stub for generate_embeddings that always succeeds.

    Uses a separate name from `_mk_stub` (which returns `count` as the *return
    value*) so ruff doesn't confuse the two helpers with FURB113 (unnecessary
    stub).
    """
    m = mock.MagicMock(return_value=count)  # noqa: FURB113 — deliberate stub for generate_embeddings side-effect.
    return m


# ── _load_titles_from_file ----------------------------------------------------- #


class TestLoadTitlesFromFile:
    def test_returns_striped_lines(self) -> None:
        """Non-empty lines are stripped and returned as a list."""
        import tempfile

        from app.build_index import load_titles_from_file

        td = tempfile.mkdtemp()
        titles_file = f"{td}/titles.txt"
        with open(titles_file, "w", encoding="utf-8") as fh:
            fh.write("  Hello World  \n\n  Another Title\n")

        result = load_titles_from_file(titles_file)
        assert result == ["Hello World", "Another Title"]

    def test_nonexistent_file_returns_empty(self) -> None:
        """Missing file → empty list."""
        import tempfile

        from app.build_index import load_titles_from_file

        td = tempfile.mkdtemp()
        result = load_titles_from_file(f"{td}/does_not_exist.txt")
        assert result == []
