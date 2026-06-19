"""Verify config module (PRD §9)."""

import importlib
import os
from unittest import mock


def test_defaults():
    """All config vars expose PRD §9 defaults."""
    # Ensure no env vars interfere
    env_vars = [
        "MODEL_NAME", "EMBED_DIM", "FAISS_INDEX", "TITLES_FILE",
        "DEFAULT_K", "DEFAULT_NPROBE", "PORT", "WORKERS",
    ]
    for var in env_vars:
        os.environ.pop(var, None)

    # Re-import to pick up un-set environment
    import app.config as cfg
    importlib.reload(cfg)

    assert cfg.MODEL_NAME == "all-MiniLM-L6-v2"
    assert cfg.EMBED_DIM == 384
    assert cfg.FAISS_INDEX == "wiki_faiss.index"
    assert cfg.TITLES_FILE == "wiki_titles.txt"
    assert cfg.DEFAULT_K == 5
    assert cfg.DEFAULT_NPROBE == 64
    assert cfg.PORT == 8000
    assert cfg.WORKERS == 1


def test_env_override_model_name():
    os.environ["MODEL_NAME"] = "bert-base-uncased"
    try:
        import app.config as cfg
        importlib.reload(cfg)
        assert cfg.MODEL_NAME == "bert-base-uncased"
    finally:
        del os.environ["MODEL_NAME"]


def test_env_override_integers():
    os.environ["EMBED_DIM"] = "768"
    os.environ["DEFAULT_K"] = "10"
    os.environ["PORT"] = "9000"
    try:
        import app.config as cfg
        importlib.reload(cfg)
        assert cfg.EMBED_DIM == 768
        assert cfg.DEFAULT_K == 10
        assert cfg.PORT == 9000
    finally:
        for key in ("EMBED_DIM", "DEFAULT_K", "PORT"):
            del os.environ[key]


def test_env_override_faiss_and_titles():
    os.environ["FAISS_INDEX"] = "custom.index"
    os.environ["TITLES_FILE"] = "custom.txt"
    try:
        import app.config as cfg
        importlib.reload(cfg)
        assert cfg.FAISS_INDEX == "custom.index"
        assert cfg.TITLES_FILE == "custom.txt"
    finally:
        for key in ("FAISS_INDEX", "TITLES_FILE"):
            del os.environ[key]


def test_env_override_nprobe_and_workers():
    os.environ["DEFAULT_NPROBE"] = "128"
    os.environ["WORKERS"] = "4"
    try:
        import app.config as cfg
        importlib.reload(cfg)
        assert cfg.DEFAULT_NPROBE == 128
        assert cfg.WORKERS == 4
    finally:
        for key in ("DEFAULT_NPROBE", "WORKERS"):
            del os.environ[key]
