"""test_build_config.py — build-stage config loading & env-var overrides (T11).

Tests every new config variable introduced in PRD §9 / T11:
  - ``WIKI_DUMP_URL``
  - ``BUILD_BATCH_SIZE``
  - ``BUILD_NLIST``
  - ``BUILD_SAMPLE_FRAC``
  - ``BUILD_RESUME``
  - ``BUILD_MANIFEST``

PRD refs: §§6, 7.4, 8.2/8.3, 9

"""

import importlib
import os
import sys
from types import ModuleType

import pytest

# ------------------------------------------------------------------ #
#  constants                                                          #
# ------------------------------------------------------------------ #

# New build-stage vars with their documented defaults (T11.A, D).
_BUILD_CONFIG_DEFAULTS: dict[str, object] = {
    "WIKI_DUMP_URL": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz",
    "BUILD_BATCH_SIZE": 512,
    "BUILD_NLIST": 4096,
    "BUILD_SAMPLE_FRAC": 0.1,
    "BUILD_RESUME": True,
    "BUILD_MANIFEST": "build_manifest.json",
}

# Sentinel overrides for env-override tests (T5a.2 pattern).
_BUILD_ENV_SENTINELS: dict[str, str] = {
    "WIKI_DUMP_URL": "https://example.com/custom-dump.json.gz",
    "BUILD_BATCH_SIZE": "1024",
    "BUILD_NLIST": "8192",
    "BUILD_SAMPLE_FRAC": "0.25",
    "BUILD_RESUME": "false",
    "BUILD_MANIFEST": "custom_manifest.json",
}

# Bool coercion edge cases (T11.B, 11.B).
_BUILD_RESUME_BOOL_CASES: list[tuple[str, bool]] = [
    ("true", True),
    ("True", True),
    ("TRUE", True),
    ("1", True),
    ("false", False),
    ("False", False),
    ("FALSE", False),
    ("0", False),
]

# Int/float typed vars (T11.C).
_INT_BUILD_VARS = ("BUILD_BATCH_SIZE", "BUILD_NLIST")
_FLOAT_BUILD_VARS = ("BUILD_SAMPLE_FRAC",)

_ALL_BUILD_VARS = list(_BUILD_CONFIG_DEFAULTS)


def _reload_config() -> ModuleType:
    """Reload (or import) *app.config* and return the module."""
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
    """Guarantee no build config env vars leak between tests."""
    for var in _ALL_BUILD_VARS:
        os.environ.pop(var, None)
    yield
    sys.modules.pop("app.config", None)


# ------------------------------------------------------------------ #
#  T11.A / D — defaults of every new variable                        #
# ------------------------------------------------------------------ #


class TestBuildConfigDefaults:
    """Each build-stage config var must return its documented default."""

    @pytest.mark.parametrize("var,expected", list(_BUILD_CONFIG_DEFAULTS.items()))
    def test_build_config_var_returns_default(self, var: str, expected: object):
        cfg = _reload_config()
        actual = getattr(cfg, var)
        assert actual == expected, f"{var}: expected {expected!r}, got {actual!r}"


# ------------------------------------------------------------------ #
#  T5a.2 pattern — env-override of new vars                          #
# ------------------------------------------------------------------ #


class TestBuildConfigEnvOverride:
    """Each build-stage var must reflect the env value when set."""

    @pytest.mark.parametrize("var,sentinel", list(_BUILD_ENV_SENTINELS.items()))
    def test_env_override(self, var: str, sentinel: str):
        os.environ[var] = sentinel
        cfg = _reload_config()
        try:
            if var in _INT_BUILD_VARS:
                assert getattr(cfg, var) == int(sentinel)
            elif var in _FLOAT_BUILD_VARS:
                assert getattr(cfg, var) == float(sentinel)
            elif var == "BUILD_RESUME":
                assert getattr(cfg, var) == (sentinel.lower() in ("true", "1"))
            else:
                assert getattr(cfg, var) == sentinel
        finally:
            del os.environ[var]


# ------------------------------------------------------------------ #
#  T11.B — BUILD_RESUME bool coercion                                #
# ------------------------------------------------------------------ #


class TestBuildResumeBoolCoercion:
    """All accepted truthy/falsy strings map to the correct Python bool."""

    @pytest.mark.parametrize("val,expected", _BUILD_RESUME_BOOL_CASES)
    def test_build_resume_coerces_string_to_bool(self, val: str, expected: bool):
        os.environ["BUILD_RESUME"] = val
        cfg = _reload_config()
        try:
            assert getattr(cfg, "BUILD_RESUME") is expected
        finally:
            del os.environ["BUILD_RESUME"]


# ------------------------------------------------------------------ #
#  T11.C — int / float coercion                                      #
# ------------------------------------------------------------------ #


class TestBuildNumericCoercion:
    """Int vars produce ``int``, float vars produce ``float``."""

    @pytest.mark.parametrize("var", _INT_BUILD_VARS)
    def test_int_build_vars_are_python_int(self, var: str):
        cfg = _reload_config()
        value = getattr(cfg, var)
        assert isinstance(value, int), f"{var}: expected *int*, got {type(value).__name__}"

    @pytest.mark.parametrize("var", _FLOAT_BUILD_VARS)
    def test_float_build_vars_are_python_float(self, var: str):
        cfg = _reload_config()
        value = getattr(cfg, var)
        assert isinstance(value, float), f"{var}: expected *float*, got {type(value).__name__}"
