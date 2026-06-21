"""Health endpoint coverage — returns correct schema in various states."""

from app.main import state


def test_health_schema_with_titles(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["titles_loaded"], int)
    assert data["titles_loaded"] == len(state.get("titles", []))


def test_health_no_extra_keys(client):
    response = client.get("/health")
    expected_keys = {"status", "titles_loaded"}
    assert set(response.json().keys()) == expected_keys
