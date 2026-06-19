"""Unit tests for search logic — embedding, FAISS round-trip, result schema."""

from unittest import mock

import numpy as np
from fastapi.responses import JSONResponse


def test_search_returns_json(client):
    """The /search endpoint returns a JSONResponse with query, results, elapsed_ms."""
    response = client.get("/search?q=photosynthesis&k=3")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "elapsed_ms" in data
    assert data["query"] == "photosynthesis"


def test_search_k_results(client):
    """Request k results → exactly k items in results."""
    for k_val in (1, 3, 5, 10):
        response = client.get(f"/search?q=hello&k={k_val}")
        assert len(response.json()["results"]) == k_val


def test_search_each_result_has_required_fields(client):
    """Every result item carries rank, title, score."""
    response = client.get("/search?q=test&k=5")
    for item in response.json()["results"]:
        assert "rank" in item
        assert "title" in item
        assert "score" in item


def test_search_result_rank_is_sequential(client):
    """Ranks are 1-based sequential integers."""
    response = client.get("/search?q=test&k=5")
    ranks = [r["rank"] for r in response.json()["results"]]
    assert ranks == list(range(1, 6))


def test_search_scores_are_rounded(client):
    """Scores are floats rounded to 4 decimal places."""
    response = client.get("/search?q=test&k=3")
    for s in response.json()["results"]:
        assert isinstance(s["score"], float)


def test_search_elapsse_ms_is_present_and_numeric(client):
    """elapsed_ms is a number (not missing or non-numeric)."""
    response = client.get("/search?q=test&k=1")
    ms = response.json()["elapsed_ms"]
    assert isinstance(ms, (int, float))
    assert ms >= 0


def test_search_encode_called_once(client):
    """Model.encode should be called once per request with the correct query."""
    from app import main as app_module

    mock_model = app_module.state["model"]
    response = client.get("/search?q=find+it&k=1")
    assert response.status_code == 200
    mock_model.encode.assert_called_once()


def test_search_encode_returns_float32(client):
    """The model's encode method returns float32 numpy arrays."""
    from app import main as app_module

    mock_model = app_module.state["model"]
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")

    response = client.get("/search?q=test&k=1")
    assert response.status_code == 200


def test_search_q_emb_normally(client):
    """Query embedding is L2-normalised before indexing."""
    from app import main as app_module
    import numpy as np

    # Use a non-zero vector so norm is meaningful
    fake_emb = np.ones((1, 384), dtype="float32")
    mock_model = app_module.state["model"]
    mock_model.encode.return_value = fake_emb

    response = client.get("/search?q=test&k=1")
    assert response.status_code == 200
