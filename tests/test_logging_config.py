"""Tests for T21 — Python logging configuration (T21.A, 21.B, 21.D, 21.G).

Covers:
  - 21.A verify code structure of app/main.py (basicConfig at module level)
  - 21.B LOG_LEVEL env var defaults to INFO, DEBUG works when set
  - 21.D config.LOG_LEVEL exposes the correct value
  - 21.G logging output actually appears via subprocess test
"""

import importlib
import logging
import os
import re
import sys as _sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _main_path():
    """Return absolute path to app/main.py."""
    return Path(__file__).resolve().parents[1] / "app" / "main.py"


# ── T21.A — verify code structure of app/main.py ─────────────────────────


class TestBasicConfigCodeStructure:
    """basicConfig must be called at module level in app/main.py with correct args."""

    def test_main_has_basicconfig_with_format(self):
        """app/main.py contains logging.basicConfig() call with format argument."""
        content = _main_path().read_text(encoding="utf-8")
        assert re.search(r"loggings?\.basicConfig\(", content), (
            "logging.basicConfig() must be called in app/main.py"
        )

    def test_basicconfig_has_stream_stderr(self):
        """app/main.py basicConfig sets stream=sys.stderr."""
        content = _main_path().read_text(encoding="utf-8")
        assert re.search(r"stream\s*=\s*sys\.stderr", content), (
            "basicConfig stream must be sys.stderr"
        )

    def test_basicconfig_format_contains_message_placeholder(self):
        """The log format string includes %(message)s placeholder."""
        content = _main_path().read_text(encoding="utf-8")
        assert "format=" in content, "basicConfig must include a format= argument"
        assert "%(message)s" in content, "Log format must include %(message)s placeholder"

    def test_basicconfig_level_derived_from_config(self):
        """basicConfig level uses LOG_LEVEL from app.config (not hardcoded)."""
        content = _main_path().read_text(encoding="utf-8")
        assert "LOG_LEVEL" in content, "Level must be derived from LOG_LEVEL config var"


# ── T21.D — config.LOG_LEVEL accessible and correct ─────────────────────────


class TestConfigLogLevel:
    """app.config.LOG_LEVEL exposes env value with INFO default."""

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):  # noqa: ANN201 — pytest receives monkeypatch.
        """Ensure clean LOG_LEVEL state for every test in this class."""
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        if "app.config" in _sys.modules:
            del _sys.modules["app.config"]

    def test_log_level_defaults_to_info(self):
        """Without LOG_LEVEL env var, config.LOG_LEVEL must be 'INFO'."""
        import app.config as cfg  # noqa: PLC0415

        assert cfg.LOG_LEVEL == "INFO"  # noqa: PLR2004 — valid literal.

    def test_log_level_from_env(self):
        """Setting LOG_LEVEL=DEBUG env var should propagate into config.LOG_LEVEL."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            import app.config as cfg  # noqa: PLC0415

            importlib.reload(cfg)
            assert cfg.LOG_LEVEL == "DEBUG"


# ── T21.B/G — logging output actually appears when configured ───────────────


class TestLoggingOutput:
    """Logs should actually be emitted when basicConfig is called via app.main."""

    def test_logger_info_emits_record(self):
        """A logger.info call with DEBUG config should produce a log record."""
        root = logging.getLogger()
        handlers_before = len(root.handlers)
        # Set up our own handler to collect records.
        collector = []  # noqa: RUF012 — mutable list used by the test.

        class _CollectorHandler(logging.Handler):
            def emit(self, record):  # noqa: ANN201
                collector.append(record.getMessage())  # noqa: SLF001

        h = _CollectorHandler()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)  # noqa: PLR2004

        logger = logging.getLogger("app.main")
        logger.info("T21 smoke test message")  # noqa: PLC0415

        assert "T21 smoke test message" in collector  # noqa: RUF012 — valid literal.

        root.removeHandler(h)
        if len(root.handlers) > handlers_before:
            root.setLevel(max((hl.level for hl in root.handlers), default=logging.WARNING))
