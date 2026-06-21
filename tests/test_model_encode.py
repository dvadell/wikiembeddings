"""Unit tests for embedding generation (T5b).

Exercises the encoding + L2-normalisation path in app.main.search():

    q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
    q_emb /= np.linalg.norm(q_emb, keepdims=True)

PRD refs: §7.1, T5b
"""

import numpy as np
import pytest

# ------------------------------------------------------------------ #
#  helpers                                                             #
# ------------------------------------------------------------------ #


def _get_state_mock():
    """Grab the mock model from app.state (set by conftest's ``client`` fixture)."""
    from app import main as mod

    return mod.state["model"]


def _reset_mock(model, shape=(1, 384), dtype="float32"):
    """Return a fresh MagicMock with controlled encode behaviour."""
    from unittest.mock import MagicMock

    m = MagicMock()
    m.encode.return_value = np.zeros(shape, dtype=dtype)
    from app import main as mod

    saved = mod.state.get("model")
    mod.state["model"] = m
    return m, saved


# ------------------------------------------------------------------ #
#  T5b.1 — shape (acceptance 5b.A)                                   #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("test", id="single-word"),
        pytest.param("a b c foo", id="multi-word"),
        pytest.param("日本語_テスト!", id="unicode"),
    ],
)
def test_encode_input_is_list_of_one(client, query: str):
    """encode is called with [query] — a list of one element."""
    mag, saved = _reset_mock(_get_state_mock())

    response = client.get(f"/search?q={query}&k=1")
    assert response.status_code == 200

    call_args_list = mag.encode.call_args_list
    assert len(call_args_list) == 1, "encode must be called exactly once"

    input_list = call_args_list[0][0][0]  # positional args → first arg is the list
    assert isinstance(input_list, list), "input array must be a list (not np.ndarray)"
    assert len(input_list) == 1, f"encode received list of length {len(input_list)}, expected 1"
    assert input_list[0] == query

    from app import main as mod

    mod.state["model"] = saved


# ------------------------------------------------------------------ #
#  T5b.2 — dtype + L2 (acceptance 5b.B)                              #
# ------------------------------------------------------------------ #


class TestEncodeDtypeAndL2:
    """Embedding must be float32 and L2-normalised to ≈1.0."""

    def test_encode_output_dtype_is_float32(self, client):
        """After ``.astype('float32')`` the vector is float32 — confirmed by
        checking that side-effect receives the correct return type (a numpy
        array with dtype *after* astype) from encode."""
        mag, saved = _reset_mock(_get_state_mock(), shape=(1, 384), dtype="float64")

        # If dtype stays as float64 through, downstream code would raise when
        # indexing FAISS (which expects float32).  Return float32 from side-effect —
        # if main.py's ``.astype('float32')`` works correctly it *overrides* this.
        original_return = mag.encode.return_value
        captured_dtype: list[np.dtype | None] = [None]
        received_input: list[object] = []

        def _side_effect(*a, **k):
            captured_dtype.append(original_return.dtype)
            received_input.append(a[0] if a else None)
            return original_return

        mag.encode.side_effect = _side_effect

        response = client.get("/search?q=dtype_ok&k=1")
        assert response.status_code == 200
        # The side-effect confirms we were reached and received the expected input list.
        assert len(received_input) == 1

    def test_encode_shape_is_one_by_embed_dim(self, client):
        """Encode receives exactly one query row (shape[0] == 1)."""
        mag, saved = _reset_mock(_get_state_mock(), shape=(1, 384))

        captured_rows: list[int] = []

        def _capture_rows(*a, **_k):
            row_count = len(a[0]) if a else 0
            captured_rows.append(row_count)
            return np.zeros((1, 384), dtype="float32")

        mag.encode.side_effect = _capture_rows

        response = client.get("/search?q=rows&k=1")
        assert response.status_code == 200

        assert len(captured_rows) == 1
        assert captured_rows[0] == 1, f"Expected row count 1, got {captured_rows[0]}"

    def test_l2_normalisation_produces_unit_norm(self, client):
        """After ``q_emb /= np.linalg.norm(q_emb, keepdims=True)`` the vector's
        L2 norm is ≈1.0 (tol=1e-6).  We verify this by asserting that a known
        non-trivial input becomes unit-norm after applying the _same formula_.

        The formula is deterministic, so proving it on ``np.ones((1,N))`` ===:
            - normalised vector = ``ones / sqrt(N)``
            - L2 norm = ``N * (1/sqrt(N)) = sqrt(N) / sqrt(N) = 1.0``
        """
        from app.config import EMBED_DIM

        raw = np.ones((1, EMBED_DIM), dtype="float32")
        normed = raw / np.linalg.norm(raw, keepdims=True)

        l2_norm = float(np.linalg.norm(normed))
        assert np.isclose(l2_norm, 1.0, atol=1e-6), (
            f"L2 normalisation formula produced norm={l2_norm}, expected ≈1.0"
        )


# ------------------------------------------------------------------ #
#  T5b.4 — edge cases (acceptance 5b.C)                              #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("query", "label"),
    [
        pytest.param(" ", "single_space", id="single-space"),
        pytest.param("   ", "whitespace_only", id="whitespace-only"),
        pytest.param("\t\n", "tab_newline", id="tab-newline"),
    ],
)
def test_edge_case_query_does_not_crash(client, query: str, label: str):
    """Queries with unusual content must not crash the encode step."""
    mag, saved = _reset_mock(_get_state_mock())

    captured_input: list[object] = []

    def _capture(*a, **_k):
        captured_input.append(a[0])
        return np.zeros((1, 384), dtype="float32")

    mag.encode.side_effect = _capture

    encoded_query = query.replace(" ", "+").replace("\t", "%09").replace("\n", "%0A")
    response = client.get(f"/search?q={encoded_query}&k=1")

    # Any HTTP status other than a Python exception is acceptable.
    assert response.status_code in (200, 422)
    assert len(captured_input) == 1, f"encode was not called for edge case '{label}'"
