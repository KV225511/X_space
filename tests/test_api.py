"""Unit and Integration Tests for AstroFoundation FastAPI Service."""

import os
import time
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.rate_limiter import rate_limiter


@pytest.fixture(autouse=True)
def setup_test_env():
    """Run API tests with offline mock database flags enabled and reset rate limits."""
    os.environ["ASTRO_OFFLINE_MODE"] = "1"
    rate_limiter.reset()
    yield
    os.environ.pop("ASTRO_OFFLINE_MODE", None)
    rate_limiter.reset()


@pytest.fixture
def client():
    """Create FastAPI TestClient."""
    return TestClient(app)


def test_health_endpoint(client):
    """Verify /health endpoint returns service health and GPU/CPU configuration."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "cuda_available" in data
    assert "device" in data
    assert "timestamp" in data


def test_status_endpoint(client):
    """Verify /status endpoint returns server uptime and task metrics."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["uptime_seconds"] >= 0
    assert "active_tasks" in data
    assert "completed_tasks" in data


def test_cross_match_endpoint(client):
    """Verify /cross-match queries astronomical databases for a target."""
    payload = {"target_id": "Kepler-10"}
    response = client.post("/cross-match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_id"] == "Kepler-10"
    assert "simbad" in data
    assert "ned" in data
    assert "ads" in data
    assert data["simbad"]["found"] is True


def test_classify_endpoint_synthetic(client):
    """Verify /classify endpoint generates probability distribution across the 6 classes."""
    payload = {
        "star_id": "KIC_9002278",
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_id"] == "KIC_9002278"
    assert data["predicted_class"] == "eclipsing_binary"
    assert "probabilities" in data
    assert len(data["probabilities"]) == 6
    assert 0.0 <= data["confidence"] <= 1.0


def test_analyze_endpoint_synchronous(client):
    """Verify /analyze runs full pipeline: model -> agent -> scientific report."""
    payload = {
        "star_id": "Kepler-10",
        "kic_id": 11904151,
        "include_agent": True,
        "include_report": True,
        "run_async": False,
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_id"] == "Kepler-10"
    assert data["classification"] == "exoplanet_transit"
    assert data["confidence"] >= 0.90
    assert "agent_output" in data
    assert data["agent_output"]["consensus_status"] == "CONFIRMED"
    assert "scientific_report" in data
    report = data["scientific_report"]
    assert "executive_summary" in report
    assert "photometric_analysis" in report
    assert "spectroscopic_analysis" in report
    assert len(report["recommended_followup"]) >= 1


def test_analyze_endpoint_asynchronous(client):
    """Verify /analyze with run_async=True enqueues background job with pollable status."""
    payload = {
        "star_id": "KIC_9700322",
        "kic_id": 9700322,
        "include_agent": True,
        "include_report": True,
        "run_async": True,
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "PENDING"
    task_id = data["task_id"]

    # Poll status
    time.sleep(0.5)
    task_resp = client.get(f"/tasks/{task_id}")
    assert task_resp.status_code == 200
    task_data = task_resp.json()
    assert task_data["task_id"] == task_id
    assert task_data["status"] in ("PENDING", "RUNNING", "SUCCESS")


# ---------------------------------------------------------------------------
# Rate Limiting Tests (Phase 9 Turn & Env Controls)
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_requests_with_headers(client):
    """Verify rate limiter allows requests and attaches rate limit headers."""
    headers = {"X-Turn-ID": "test_turn_headers"}
    response = client.get("/status", headers=headers)
    assert response.status_code == 200
    assert "x-ratelimit-limit" in response.headers
    assert "x-ratelimit-remaining" in response.headers
    assert int(response.headers["x-ratelimit-remaining"]) == rate_limiter.max_requests - 1


def test_rate_limiter_blocks_when_quota_exhausted(client, monkeypatch):
    """Verify rate limiter returns HTTP 429 when turn request limit is exceeded."""
    monkeypatch.setenv("RATE_LIMIT_PER_TURN", "3")
    headers = {"X-Turn-ID": "test_turn_exhaust"}

    # 3 allowed requests
    for i in range(3):
        res = client.get("/status", headers=headers)
        assert res.status_code == 200
        assert int(res.headers["x-ratelimit-remaining"]) == 3 - (i + 1)

    # 4th request must be blocked with HTTP 429
    blocked = client.get("/status", headers=headers)
    assert blocked.status_code == 429
    data = blocked.json()
    assert data["error"] == "Rate limit exceeded"
    assert "Maximum 3 requests per turn allowed" in data["detail"]
    assert blocked.headers["x-ratelimit-remaining"] == "0"
    assert "retry-after" in blocked.headers


def test_rate_limiter_turn_isolation(client, monkeypatch):
    """Verify separate turns (X-Turn-ID) have isolated quotas."""
    monkeypatch.setenv("RATE_LIMIT_PER_TURN", "2")

    # Exhaust turn A
    client.get("/status", headers={"X-Turn-ID": "turn_A"})
    client.get("/status", headers={"X-Turn-ID": "turn_A"})
    res_a_blocked = client.get("/status", headers={"X-Turn-ID": "turn_A"})
    assert res_a_blocked.status_code == 429

    # Turn B is fresh and allowed
    res_b = client.get("/status", headers={"X-Turn-ID": "turn_B"})
    assert res_b.status_code == 200
    assert res_b.headers["x-ratelimit-remaining"] == "1"


def test_rate_limiter_reads_env_dynamically(monkeypatch):
    """Verify rate_limiter.max_requests dynamically reflects RATE_LIMIT_PER_TURN from env."""
    monkeypatch.setenv("RATE_LIMIT_PER_TURN", "75")
    assert rate_limiter.max_requests == 75

    monkeypatch.setenv("RATE_LIMIT_PER_TURN", "50")
    assert rate_limiter.max_requests == 50


def test_health_endpoint_excluded_from_rate_limit(client, monkeypatch):
    """Verify health endpoint is excluded from consuming rate limit turns."""
    monkeypatch.setenv("RATE_LIMIT_PER_TURN", "1")
    headers = {"X-Turn-ID": "turn_health"}

    # Multiple calls to health should not be blocked or count against quota
    for _ in range(3):
        res = client.get("/health", headers=headers)
        assert res.status_code == 200
