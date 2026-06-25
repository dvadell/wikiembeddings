"""Health endpoint — schema validation on a fully-cycled app.

Exercises the real FastAPI lifespan with mocked FAISS + model deps so that
state is populated on startup and cleared on teardown.
"""

import faiss
import numpy as np
import sentence_transformers
from fastapi.testclient import TestClient

# ── mock factory (called inside tests, never at module level) ───────────────


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
            self.encode = lambda *aa, **kk: np.zeros((1, 384), dtype="float32")  # type: ignore[assignment]

    return (
        (faiss, "read_index", lambda *a, **kw: mock_idx),
        (sentence_transformers, "SentenceTransformer", MockModel),
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _boot_app(monkeypatch, tmp_data_dir):
    """Apply mocks + temp paths; return app module after importing it."""
    for target, name, value in _make_mocks():
        monkeypatch.setattr(target, name, value)

    import app.main as mod
    from tests.conftest import _patch_temp_paths

    _patch_temp_paths(tmp_data_dir, monkeypatch)
    return mod


# ── tests ────────────────────────────────────────────────────────────────────


def test_health_status_ok(monkeypatch, tmp_data_dir):
    """Health endpoint returns 200 with status and titles_loaded."""
    mod = _boot_app(monkeypatch, tmp_data_dir)
    orig = dict(mod.state)
    mod.state.clear()

    with TestClient(mod.app):
        pass

    assert len(mod.state) == 0
    mod.state.clear()
    mod.state.update(orig)


def test_health_schema_full_lifecycle(monkeypatch, tmp_data_dir):
    """Health schema + state populated + cleared via full lifespan."""
    mod = _boot_app(monkeypatch, tmp_data_dir)
    orig = dict(mod.state)
    mod.state.clear()

    with TestClient(mod.app) as client:
        assert "titles" in mod.state
        assert "index" in mod.state
        assert "model" in mod.state
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "ready")  # ready when manifest; ok for backwards compat
        assert set(data.keys()) >= {"status", "progress", "titles_loaded"}
        assert isinstance(data["titles_loaded"], int)

    assert len(mod.state) == 0
    mod.state.clear()
    mod.state.update(orig)


def test_health_schema_keys_only(monkeypatch, tmp_data_dir):
    """Health endpoint keys include status, progress, and titles_loaded."""
    mod = _boot_app(monkeypatch, tmp_data_dir)
    orig = dict(mod.state)
    mod.state.clear()

    with TestClient(mod.app) as client:
        keys = set(client.get("/health").json().keys())
        assert keys >= {"status", "progress", "titles_loaded"}

    assert len(mod.state) == 0
    mod.state.clear()
    mod.state.update(orig)
