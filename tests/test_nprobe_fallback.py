"""T20 — _NprobeIndex fallback (app/main.py lines 166‑189).

When a live FAISS index lacks a writable ``.nprobe`` attribute,
``_load_faiss_index()`` wraps it in a thin compatibility shim so callers can
safely do ``idx.nprobe = value`` and always get *k* results back (padded with
distance zeros and integer offsets if the inner index returns fewer candidates
than requested).

Acceptance criteria covered
============================
T20.A  _load_faiss_index wraps unwrapped indexes in a compatible shim
T20.B  Shim .nprobe setter/getter round-trip
T20.C  Shim search pads shorter results identically to the real code (lines 173‑185)
"""

from __future__ import annotations

from unittest import mock

import numpy as np


class _NoNprobeIndex:
    """A plain object that mimics a FAISS IVF index — no ``.nprobe`` attribute."""

    def __init__(self, scores: np.ndarray, indices: np.ndarray):
        self._scores = scores
        self._indices = indices

    def search(self, x, k):  # noqa: ANN001 — matches FAISS idx.search signature.
        return (self._scores[:, :k], self._indices[:, :k])


class TestNprobeFallbackLoadedViaModulePatch:
    """T20.A — directly call _load_faiss_index with a non-IVF index."""

    def test_unwrapped_gets_shim(self):
        """Patch app.main.faiss (the *already-imported* module reference) so that
        faiss.read_index returns an unwrapped index → wrapper path exercised.
        """
        from app import main as mod

        raw_idx = _NoNprobeIndex(
            scores=np.array([[-2.5, -1.0]], dtype="float32"),
            indices=np.array([[0, 1]], dtype="int32"),
        )

        # Verify raw index has no nprobe (the trigger condition).
        assert not hasattr(raw_idx, "nprobe") or getattr(raw_idx, "nprobe", None) in [None, False]

        # Patch the *module-level* faiss binding — this works because _load_faiss_index
        # references `faiss` from its enclosing scope (imported at module level).
        original_faiss = mod.faiss  # save real FAISS for cleanup.
        fake_faiss = mock.MagicMock()
        fake_faiss.read_index.return_value = raw_idx

        try:
            mod.faiss = fake_faiss  # noqa: FBT003 — intentional patch; restores after.

            # patched: path arg ignored, read_index never called
            wrapper = mod._load_faiss_index("/dev/null")
            assert hasattr(wrapper, "nprobe"), "wrapper must have .nprobe"

        finally:
            mod.faiss = original_faiss  # restore original module reference.


class TestNprobeShimBehavior:
    """T20.B / C — _NProbeIndex shim behavior (lines 168‑189)."""

    def test_nprobe_get_set(self):
        """Shim .nprobe getter/setter round-trip."""
        from app import main as mod

        raw_idx = _NoNprobeIndex(
            scores=np.array([[-2.0, -1.0]], dtype="float32"),
            indices=np.array([[5, 7]], dtype="int32"),
        )

        fake_faiss = mock.MagicMock()
        fake_faiss.read_index.return_value = raw_idx

        mod.faiss = fake_faiss
        try:
            wrapper = mod._load_faiss_index("/dev/null")
            assert hasattr(wrapper, "nprobe")
            wrapper.nprobe = 128
            assert wrapper.nprobe == 128

        finally:
            mod.faiss = (
                mod.__dict__.get("faiss")
                or mock.MagicMock()
            )

    def test_wrapper_search_pads_short_results(self):
        """Shim pads with distance zeros and index offset-indices when ds.shape[1] < k."""
        from app import main as mod

        raw_idx = _NoNprobeIndex(
            scores=np.array([[-2.0, -1.0]], dtype="float32"),  # only 2 candidates (k=4 requested).
            indices=np.array([[5, 7]], dtype="int32"),
        )

        fake_faiss = mock.MagicMock()
        fake_faiss.read_index.return_value = raw_idx

        mod.faiss = fake_faiss
        try:
            wrapper = mod._load_faiss_index("/dev/null")

            q = np.zeros((1, 384), dtype="float32")
            ds_out, ix_out = wrapper.search(q, 4)

            assert ds_out.shape[1] == 4
            assert ix_out.shape[1] == 4

            np.testing.assert_array_equal(ds_out[:, :2], [[-2.0, -1.0]])
            np.testing.assert_array_equal(ix_out[:, :2], [[5, 7]])

            # padding: distances zero, indices sequential from ~50+
            assert ds_out[0, 2] == 0.0
            assert ds_out[0, 3] == 0.0
            assert ix_out[0, 2] >= 50
            assert ix_out[0, 3] >= 51

        finally:
            mod.faiss = mod.__dict__.get("faiss") or mock.MagicMock()


class TestIvfIndexReturnsDirectly:
    """When the loaded index has ``.nprobe``, it is returned as-is (no shim overhead)."""

    def test_ivf_no_wrap_needed(self):
        """Real-looking wrapped index → identity returned."""
        from app import main as mod

        mock_idx = mock.MagicMock()                     # auto-creates .nprobe and everything.
        mock_idx.nprobe = 64                            # simulate real IVF index having nprobe.
        scores = np.array([[-2.0, -1.0]], dtype="float32")
        indices = np.array([[10, 20]], dtype="int32")
        mock_idx.search.return_value = (scores, indices)

        fake_faiss = mock.MagicMock()
        fake_faiss.read_index.return_value = mock_idx

        original_faiss = mod.faiss
        mod.faiss = fake_faiss
        try:
            result = mod._load_faiss_index("/dev/null")
            assert result is mock_idx                     # no wrapper, identity preserved.

            q = np.zeros((1, 384), dtype="float32")
            ds_out, ix_out = result.search(q, 2)                    # uses mock_idx directly.

            assert ds_out[0][0] == -2.0
            assert ix_out[0][0] == 10

        finally:
            mod.faiss = original_faiss
