"""Verify lifespan: state is populated on startup and cleared on shutdown."""

from unittest.mock import MagicMock

import numpy as np

from tests.conftest import _patch_temp_paths


def test_lifespan_populates_state(monkeypatch, tmp_data_dir):
    """After lifespan runs, state should contain 'titles', 'index', 'model' keys."""
    from fastapi.testclient import TestClient

    import app.config as cfg
    import app.main as app_module

    orig_cfg_faiss, orig_cfg_titles, orig_main_faiss, orig_main_titles = _patch_temp_paths(
        tmp_data_dir, monkeypatch
    )

    original_state = dict(app_module.state)
    app_module.state.clear()

    # Replace real model loading with a mock
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *a, **k: mock_model)

    # Also replace faiss.read_index so it returns our real temp index (already IVF-compatible)
    # The key issue is: the temp file has IndexFlat2 which doesn't support .nprobe.
    # We need lifespan to load a wrapper that supports .nprobe. So we mock faiss.read_index too.
    import faiss

    dummy = np.zeros((1, 384), dtype="float32")
    flat_idx = faiss.IndexFlatL2(384)
    flat_idx.add(dummy)

    class _NprobeIndex:
        def __init__(self):
            self._n = 64

        @property
        def nprobe(self):
            return self._n

        @nprobe.setter
        def nprobe(self, value):
            self._n = int(value)

        def search(self, x, k):
            scores = np.zeros((1, 50), dtype="float32")
            ids = np.tile(np.arange(50), (1, 50))
            return scores.astype("float32"), ids.astype("int32")

    real_read = faiss.read_index
    monkeypatch.setattr(faiss, "read_index", lambda *a, **k: _NprobeIndex())

    client = TestClient(app_module.app)
    with client:
        assert "titles" in app_module.state
        assert "index" in app_module.state
        assert "model" in app_module.state
        assert len(app_module.state["titles"]) == 3

    # After exit, lifespan clear() was called
    assert len(app_module.state) == 0

    app_module.state.clear()
    app_module.state.update(original_state)

    # Restore config
    cfg.FAISS_INDEX = orig_cfg_faiss
    cfg.TITLES_FILE = orig_cfg_titles
    try:
        app_module.FAISS_INDEX = orig_main_faiss
        app_module.TITLES_FILE = orig_main_titles
    except AttributeError:
        pass

    faiss.read_index = real_read


def test_state_cleared_on_shutdown(monkeypatch, tmp_data_dir):
    """State dict is {} (empty) after lifespan context manager exits."""
    from unittest.mock import MagicMock

    import numpy as np
    from fastapi.testclient import TestClient

    import app.main as app_module

    _patch_temp_paths(tmp_data_dir, monkeypatch)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")

    class _NprobeIndex:
        def __init__(self):
            self._n = 64

        @property
        def nprobe(self):
            return self._n

        @nprobe.setter
        def nprobe(self, v):
            self._n = int(v)

        def search(self, x, k):
            return np.zeros((1, k)), np.zeros((k,), dtype="int32")

    import faiss

    monkeypatch.setattr(faiss, "read_index", lambda *a: _NprobeIndex())
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *a: mock_model)

    original = dict(app_module.state)
    app_module.state.clear()

    client = TestClient(app_module.app)
    with client:
        pass  # triggers full load → yield → clear cycle

    assert len(app_module.state) == 0

    app_module.state.clear()
    app_module.state.update(original)
