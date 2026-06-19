"""Shared fixtures for the test suite."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── nprobe-wrapper so IndexFlat also supports .nprobe ───────────────────────


class _NprobeIndex:
    """Thin wrapper that adds a writable `.nprobe` to any FAISS index."""

    __slots__ = ("_idx", "_n")

    def __init__(self, underlying):
        self._idx = underlying
        self._n = 64

    @property
    def nprobe(self):
        return self._n

    @nprobe.setter
    def nprobe(self, value):
        self._n = int(value)

    def search(self, x, k):
        ds, ix = self._idx.search(x, max(k * 5, 20))
        if ds.shape[1] < k:
            pad_n = k - ds.shape[1]
            pad_zeros = np.zeros((ds.shape[0], pad_n), dtype="float32")
            ds = np.concatenate([ds, pad_zeros], axis=1)
            ix_padded = np.tile(np.arange(50, 50 + pad_n, dtype="int32"), (ds.shape[0], 1))
            ix = np.concatenate([ix, ix_padded], axis=1)
        return ds[:, :k].astype("float32"), ix[:, :k]


def _make_wrapper_index():
    base = faiss.IndexFlatL2(384)
    base.add(np.zeros((50, 384), dtype="float32"))
    return _NprobeIndex(base)


# ── Session-scoped temp data dir with real FAISS index file ────────────────


@pytest.fixture(scope="session")
def tmp_data_dir() -> Path:
    td = Path(tempfile.mkdtemp(prefix="wiki_test_"))
    dummy = np.zeros((1, 384), dtype="float32")
    idx = faiss.IndexFlatL2(384)
    idx.add(dummy)
    faiss.write_index(idx, str(td / "wiki_faiss.index"))
    (td / "wiki_titles.txt").write_text("A\nB\nC\n", encoding="utf-8")
    return td


# ── config-path override (runs before every test via monkeypatch) ───────────


@pytest.fixture(autouse=True)
def _override_config_paths(monkeypatch, tmp_data_dir: Path):
    """Point app.config at temp files for every test."""
    import app.config as cfg

    monkeypatch.setattr(cfg, "FAISS_INDEX", str(tmp_data_dir / "wiki_faiss.index"))
    monkeypatch.setattr(cfg, "TITLES_FILE", str(tmp_data_dir / "wiki_titles.txt"))


# ── TestClient with pre-seeded state (bypasses real model/index loading) ───


@pytest.fixture()
def client():
    """Return a TestClient whose lifespan state is pre-seeded so /search & /health work."""
    from app import main as app_module

    original = dict(app_module.state)
    app_module.state["titles"] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    app_module.state["index"] = _make_wrapper_index()

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")
    app_module.state["model"] = mock_model

    yield TestClient(app_module.app)

    app_module.state.clear()
    app_module.state.update(original)


# ── Helper: patch main + config so lifespan loads from temp dir ─────────────


def _patch_temp_paths(td, monkeypatch):
    """Patch both app.config AND app.main module-level paths. Returns (orig_faiss, orig_titles)."""
    import app.config as cfg
    import app.main as mod

    temp_faiss = str(td / "wiki_faiss.index")
    temp_titles = str(td / "wiki_titles.txt")

    # Patch config module
    orig_cfg_faiss = cfg.FAISS_INDEX
    orig_cfg_titles = cfg.TITLES_FILE
    cfg.FAISS_INDEX = temp_faiss
    cfg.TITLES_FILE = temp_titles

    # Patch main module's copies (set at import time)
    orig_main_faiss = mod.FAISS_INDEX
    orig_main_titles = mod.TITLES_FILE
    mod.FAISS_INDEX = temp_faiss
    mod.TITLES_FILE = temp_titles

    return orig_cfg_faiss, orig_cfg_titles, orig_main_faiss, orig_main_titles
