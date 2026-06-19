"""Validate _validate_path error branches."""

import os
import sys


def test_validate_path_missing_raises(monkeypatch, tmp_data_dir):
    """_validate_path calls sys.exit when file doesn't exist."""
    from app.main import _validate_path
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, 'exit', lambda code=None: (_ for _ in ()).throw(SystemExit(code)))
        try:
            _validate_path(os.path.join(str(tmp_data_dir), "nonexistent.index"))
            pytest.fail("Should have called sys.exit via SystemExit")
        except SystemExit as e:
            assert "not found" in str(e)


def test_validate_path_existing_returns_path(monkeypatch, tmp_data_dir):
    """_validate_path returns the path when the file exists."""
    from app.main import _validate_path
    p = os.path.join(str(tmp_data_dir), "wiki_faiss.index")
    assert _validate_path(p) == p
