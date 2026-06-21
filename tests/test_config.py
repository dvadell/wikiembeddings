"""test_config.py — config loading & env-var overrides (T5a).

Tests every PRD §9 configuration variable:
  - default value when the env var is absent
  - override value when the env var is present
  - integer coercion for typed vars.

PRD refs: §9, T1.1.2
"""

import importlib
import os
import sys
from types import ModuleType

import pytest

# ------------------------------------------------------------------ #
#  constants                                                          #
# ------------------------------------------------------------------ #

_INT_VARS = (
    "EMBED_DIM",
    "DEFAULT_K",
    "DEFAULT_NPROBE",
    "PORT",
    "WORKERS",
)

# Every PRD §9 variable with its documented default.
_CONFIG_DEFAULTS: dict[str, object] = {
    "MODEL_NAME": "all-MiniLM-L6-v2",
    "EMBED_DIM": 384,
    "FAISS_INDEX": "wiki_faiss.index",
    "TITLES_FILE": "wiki_titles.txt",
    "DEFAULT_K": 5,
    "DEFAULT_NPROBE": 64,
    "PORT": 8000,
    "WORKERS": 1,
}

# Sentinel overrides for env-override tests.
_ENV_SENTINELS: dict[str, str] = {
    "MODEL_NAME": "test-bert-model",
    "EMBED_DIM": "768",
    "FAISS_INDEX": "custom.index",
    "TITLES_FILE": "custom_titles.txt",
    "DEFAULT_K": "10",
    "DEFAULT_NPROBE": "128",
    "PORT": "9000",
    "WORKERS": "4",
}

_ALL_VARS = list(_CONFIG_DEFAULTS)


def _reload_config() -> ModuleType:
    """Reload (or import) *app.config* and return the module.  Does **not**
    touch ``os.environ`` — the caller / fixture handles that."""
    if "app.config" in sys.modules:
        importlib.reload(sys.modules["app.config"])
    else:
        import app.config as mod

        sys.modules["app.config"] = mod
    return sys.modules["app.config"]


# ------------------------------------------------------------------ #
#  fixture: clean-env baseline for EVERY test                        #
# ------------------------------------------------------------------ #


@pytest.fixture(autouse=True)
def _clean_env():
    """Guarantee no config env vars leak between tests."""
    for var in _ALL_VARS:
        os.environ.pop(var, None)
    yield
    sys.modules.pop("app.config", None)


# ------------------------------------------------------------------ #
#  T5a.1 — defaults                                                  #
# ------------------------------------------------------------------ #


class TestConfigDefaults:
    """Each PRD §9 var must return its documented default when unset."""

    @pytest.mark.parametrize("var,expected", list(_CONFIG_DEFAULTS.items()))
    def test_config_var_returns_default(self, var: str, expected: object):
        cfg = _reload_config()
        actual = getattr(cfg, var)
        assert actual == expected, f"{var}: expected {expected!r}, got {actual!r}"


# ------------------------------------------------------------------ #
#  T5a.2 — env-override                                              #
# ------------------------------------------------------------------ #


class TestConfigEnvOverride:
    """Each PRD §9 var must reflect the env value when set."""

    @pytest.mark.parametrize("var,sentinel", list(_ENV_SENTINELS.items()))
    def test_env_override(self, var: str, sentinel: str):
        os.environ[var] = sentinel
        cfg = _reload_config()
        try:
            if var in _INT_VARS:
                assert getattr(cfg, var) == int(sentinel)
            else:
                assert getattr(cfg, var) == sentinel
        finally:
            del os.environ[var]


# ------------------------------------------------------------------ #
#  T5a.3 — int coercion (edge-case parsing)                         #
# ------------------------------------------------------------------ #


class TestConfigIntCoercion:
    """Verify that integer-typed vars produce *int*, not *str*."""

    @pytest.mark.parametrize("var", _INT_VARS)
    def test_int_vars_are_python_int(self, var: str):
        cfg = _reload_config()
        value = getattr(cfg, var)
        assert isinstance(value, int), f"{var} should be *int*, got {type(value).__name__}"
