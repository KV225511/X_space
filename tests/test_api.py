"""Unit and Integration Tests for AstroFoundation FastAPI Service."""

import os
import time
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(autouse=True)
def setup_test_env():
    """Run API tests with offline mock database flags enabled."""
    os.environ["ASTRO_OFFLINE_MODE"] = "1"
    yield
    os.environ.pop("ASTRO_OFFLINE_MODE", None)


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
