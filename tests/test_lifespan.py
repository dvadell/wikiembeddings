"""Lifespan: state population on startup and teardown on shutdown (5d.B, 5d.C).

Exercises the real FastAPI lifespan with mocked FAISS + model deps so that
state is populated on startup and cleared on teardown.
"""

import faiss
import numpy as np
import sentence_transformers


def _make_mocks():
    """Return patched *args suitable for monkeypatch.setattr."""

    class MockFAISS:
        def __init__(self, base):  # replaces faiss.read_index
            self._base = base
            self._val = 64

        @property
        def nprobe(self):
            return self._val

        @nprobe.setter
        def nprobe(self, v):
            self._val = int(v)

        def search(self, x, k):
            ds, ix = self._base.search(x, max(k * 6, 20))
            if ds.shape[1] < k:
                pad = k - ds.shape[1]
                ds = np.pad(ds, ((0, 0), (0, pad)), constant_values=0)
                ix_p = np.tile(np.arange(50, 50 + pad), (ds.shape[0], 1))
                ix = np.concatenate([ix, ix_p], axis=1)
            return ds[:, :k].astype("float32"), ix[:, :k]

    base = faiss.IndexFlatL2(384)
    base.add(np.zeros((1, 384), dtype="float32"))
    mock_idx = MockFAISS(base)

    class MockModel:
        def __init__(self, *a, **k):  # replaces SentenceTransformer(...)
            self.encode = lambda *aa, **kk: np.zeros((1, 384), dtype="float32")

    return (
        (faiss, "read_index", lambda *a, **kw: mock_idx),
        (sentence_transformers, "SentenceTransformer", MockModel),
    )


def _boot_app(monkeypatch, tmp_data_dir):
    """Apply mocks + temp paths; return app module after importing it."""
    for target, name, value in _make_mocks():
        monkeypatch.setattr(target, name, value)

    import app.main as mod  # noqa: PLC0415 (after patches are active)
    from tests.conftest import _patch_temp_paths

    _patch_temp_paths(tmp_data_dir, monkeypatch)
    return mod


# ── tests ────────────────────────────────────────────────────────────────────


def test_lifecycle_state_populated_and_cleared(monkeypatch, tmp_data_dir):
    """5d.B + 5d.C: state keys appear after start and vanish after shutdown."""
    mod = _boot_app(monkeypatch, tmp_data_dir)
    orig = dict(mod.state)
    mod.state.clear()

    from fastapi.testclient import TestClient

    with TestClient(mod.app) as client:
        assert "titles" in mod.state
        assert "index" in mod.state
        assert "model" in mod.state

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    assert len(mod.state) == 0

    mod.state.clear()
    mod.state.update(orig)
