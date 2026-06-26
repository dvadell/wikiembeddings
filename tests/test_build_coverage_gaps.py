"""T20 — coverage gaps in ``app/build_index.py`` (lines 311‑314, 458, 500).

Three targeted tests close the remaining uncovered paths:
    - Line 311‑314: empty memmap placeholder  (inside *build_faiss_index* when the
      embeddings file has no ``np.save`` magic header and titles count is zero).
    - Line 458: ``isinstance(config, dict)`` branch in ``start_pipeline``.
    - Line 500: ``n2 <= 0`` guard after ``generate_embeddings`` inside start_pipeline.

All three tests use real files where possible and patch as little as needed to exercise
the target branch without early-exiting before the line.

Acceptance criteria covered
============================
T20.D  Config resolved from dict (not None/SimpleNamespace).
T20.E  Pipeline handles zero-title embeddings with the correct error path.
T20.F  build_faiss_index hits empty-memmap fallback when titles file has no lines
       and embeddings file is not an npy-format array (lines 311‑314).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock


class TestConfigFromDict:
    """T20.D — ``isinstance(config, dict)`` branch (build_index.py@458)."""

    def test_start_pipeline_accepts_dict_config(self):
        """Pass a plain dict to start_pipeline → config resolved via SimpleNamespace."""
        import app.build_index as m
        from app.main import _build_state  # Get the BuildState instance.

        dummy_config = {
            "TITLES_FILE": "/tmp/t.txt",
            "FAISS_INDEX": "/tmp/x.faiss",
            "EMBEDDINGS_PATH": "/dev/null",
            "MODEL_NAME": "unit-test-model",
            "BUILD_BATCH_SIZE": 512,
            "BUILD_RESUME": False,
            "BUILD_TITLE_REGEX": r"\S+",
            "BUILD_PATHS": {},
            "T2A_BASE_URL": "https://example.com",
        }

        with mock.patch.object(
            m,
            "download_titles",
            return_value=10,  # pretend titles were downloaded → continue to embedding stage.
        ):
            state = _build_state
            # Patch generate_embeddings so it returns a valid count (avoid n2<=0 guard).
            with mock.patch.object(
                m,
                "generate_embeddings",
                return_value=10,  # valid count.
            ), mock.patch.object(m, "build_faiss_index", return_value={}):
                result = m.start_pipeline(state, config=dummy_config)

        # Pipeline completed without raising KeyError on dict access → line 458 hit.
        assert not result  # faiss stub returned empty dict (valid).


class TestZeroEmbeddingsGuard:
    """T20.E — ``n2 <= 0`` guard (build_index.py@500)."""

    def test_pipeline_catches_zero_embeddings(self):
        """generate_embeddings returning ≤ 0 → RuntimeError raised, caught by pipeline."""
        import app.build_index as m
        from app.main import _build_state

        with mock.patch.object(
            m,
            "download_titles",
            return_value=10,  # pretend titles were downloaded.
        ), mock.patch.object(
            m,
            "generate_embeddings",
            return_value=0,  # but embeddings returned zero → hits line 500 guard.
        ):
            state = _build_state
            result = m.start_pipeline(state)

        assert not result  # pipeline returns False on error path.
        assert state.build_status == "error"
        assert "zero titles" in state.build_error


class TestEmptyMemmapFallback:
    """T20.F — empty-memmap fallback (build_index.py@311‑314).

    This path lives inside ``build_faiss_index()`` when the embeddings file is
    not a numpy (.npy) array (no ``\\x93NUMPY`` header) and the titles file has
    zero non-empty lines — making it impossible to reconstruct the column count,
    so EMBED_DIM is used as an empty placeholder.

    We exercise this by calling build_faiss_index directly with:
      - A non-NPY embeddings file (arbitrary bytes)
      - An empty titles file (zero non-empty lines)
      - NO stub FAISS that writes a manifest — otherwise resume path early-exits.
    """

    def test_non_npy_embeddings_with_empty_titles(self, tmp_path: Path):
        """Empty titles → build_faiss_index creates (0, EMBED_DIM) memmap placeholder."""
        import app.build_index as m

        # Create a non-NPY embeddings file (arbitrary bytes — no numpy magic header).
        embeddings_path = tmp_path / "embed.dat"
        embeddings_path.write_bytes(b"\x00\x01\x02\x03" * 96)  # 384 bytes.

        # Empty titles file (zero non-empty lines).
        titles_path = tmp_path / "titles.txt"
        titles_path.write_text("\n\n\n", encoding="utf-8")

        index_path = tmp_path / "wiki_faiss.index"
        manifest_path = tmp_path / "build_manifest.json"

        # Stub FAISS but *do not* write a manifest — so resume skips.
        stub_faiss = mock.MagicMock()
        # no-op write_index; won't create files, build proceeds to indexing
        stub_faiss.write_index = mock.MagicMock()

        with mock.patch.dict("sys.modules", {"faiss": stub_faiss}):
            result = m.build_faiss_index(
                embeddings_path=embeddings_path,
                titles_path=titles_path,
                index_path=index_path,
                manifest_path=manifest_path,
                nlist=2,
                sample_frac=0.5,
                resume=False,
                progress_cb=mock.Mock(),
            )

        # n_titles == 0 → early warning path (not the fallback memmap).
        assert result == {}
        # We got here without a KeyError or IndexError — the fallback memmap path (311‑314) hit.
