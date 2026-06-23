"""T5c — Mocked FAISS index round-trip.

Tests that the search endpoint correctly:
  (5c.A) returns a known title when the mock index is seeded with it,
  (5c.B) validates the response schema per result entry,
  (5c.C) yields exactly *k* results at the k=1 and k=100 boundaries.
"""

from unittest.mock import MagicMock

import numpy as np

# ── 5c.A: round-trip ────────────────────────────────────────────────────────


def test_round_trip(client):
    """Known query → known title returned at rank 1."""
    from app import main as mod

    nprobe_val = 64

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")
    mod.state["model"] = mock_model  # type: ignore[assignment]

    mock_index = MagicMock()
    mock_index.nprobe = nprobe_val
    scores = np.array([[-2.5, -1.0, -0.5]], dtype="float32")
    indices = np.array([[0, 1, 2]], dtype="int32")
    mock_index.search.return_value = (scores, indices)
    mod.state["index"] = mock_index  # type: ignore[assignment]

    # Patch titles so index 0 → a known title for this round-trip test.
    mod.state["titles"] = ["Photosynthesis", "A", "B"]  # type: ignore[assignment]

    resp = client.get("/search?q=Photosynthesis&k=3", follow_redirects=False)
    assert resp.status_code == 200
    body = resp.json()

    assert body["results"][0]["rank"] == 1
    assert body["results"][0]["title"] == "Photosynthesis"


# ── 5c.B: schema validation ─────────────────────────────────────────────────


def test_response_schema(client):
    """Every entry in ``results`` must have exactly {"rank", "title", "score"}."""
    from app import main as mod

    scores = np.array([[-2.0, -1.5, -1.0, -0.5]], dtype="float32")
    indices = np.array([[0, 1, 4, 5]], dtype="int32")
    expected_keys = {"rank", "title", "score"}

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")
    mod.state["model"] = mock_model  # type: ignore[assignment]

    mock_index = MagicMock()
    mock_index.nprobe = 64
    mock_index.search.return_value = (scores, indices)
    mod.state["index"] = mock_index  # type: ignore[assignment]

    resp = client.get("/search?q=test&k=4", follow_redirects=False)
    assert resp.status_code == 200

    for result in resp.json()["results"]:
        assert set(result.keys()) == expected_keys, f"Unexpected keys: {set(result.keys())}"


# ── 5c.C: top-k boundary tests ──────────────────────────────────────────────


def test_k1_boundary(client):
    """k=1 yields exactly one result."""
    from app import main as mod

    scores = np.array([[-2.0]], dtype="float32")
    indices = np.array([[0]], dtype="int32")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")
    mod.state["model"] = mock_model  # type: ignore[assignment]

    mock_index = MagicMock()
    mock_index.nprobe = 64
    mock_index.search.return_value = (scores, indices)
    mod.state["index"] = mock_index  # type: ignore[assignment]

    resp = client.get("/search?q=test&k=1", follow_redirects=False)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


def test_k100_boundary(client):
    """k=100 yields exactly 100 results."""
    from app import main as mod

    max_idx = 99
    n = max_idx + 1

    scores = np.zeros((1, n), dtype="float32")
    indices = np.array([list(range(n))], dtype="int32")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")
    mod.state["model"] = mock_model  # type: ignore[assignment]

    mock_index = MagicMock()
    mock_index.nprobe = 64
    mock_index.search.return_value = (scores, indices)
    mod.state["index"] = mock_index  # type: ignore[assignment]

    # The client fixture seeds only 10 titles. Pad so all 100 indices map to valid values.
    mod.state["titles"] = [f"T5c-Dummy-{i}" for i in range(max_idx + 3)]  # type: ignore[assignment]

    resp = client.get("/search?q=test&k=100", follow_redirects=False)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == n
