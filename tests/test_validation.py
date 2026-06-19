"""Parameter validation — invalid k / nprobe / empty q must return HTTP 422."""


def test_empty_q_returns_422(client):
    """q is required; omitting it → 422."""
    response = client.get("/search")
    assert response.status_code == 422


def test_k_zero_returns_422(client):
    """k < 1 → 422."""
    response = client.get("/search?q=test&k=0")
    assert response.status_code == 422


def test_k_too_large_returns_422(client):
    """k > 100 → 422."""
    response = client.get("/search?q=test&k=101")
    assert response.status_code == 422


def test_negative_k_returns_422(client):
    """Negative k → 422."""
    response = client.get("/search?q=test&k=-1")
    assert response.status_code == 422


def test_nprobe_zero_returns_422(client):
    """nprobe < 1 → 422."""
    response = client.get("/search?q=test&nprobe=0")
    assert response.status_code == 422


def test_negative_nprobe_returns_422(client):
    """Negative nprobe → 422."""
    response = client.get("/search?q=test&nprobe=-5")
    assert response.status_code == 422


def test_nprobe_too_large_returns_422(client):
    """nprobe > 4096 → 422."""
    response = client.get("/search?q=test&nprobe=4097")
    assert response.status_code == 422


def test_boundary_k1_works(client):
    """k = 1 (minimum) → 200."""
    response = client.get("/search?q=test&k=1")
    assert response.status_code == 200


def test_boundary_k100_works(client):
    """k = 100 (maximum) → 200."""
    response = client.get("/search?q=test&k=100")
    assert response.status_code == 200


def test_boundary_nprobe_1_works(client):
    """nprobe = 1 (minimum) → 200."""
    response = client.get("/search?q=test&nprobe=1")
    assert response.status_code == 200


def test_boundary_nprobe_4096_works(client):
    """nprobe = 4096 (maximum accepted value) → 200."""
    response = client.get("/search?q=test&nprobe=4096")
    assert response.status_code == 200
