"""
Integration tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app, startup_event

@pytest.fixture(scope="module")
def client():
    startup_event()
    return TestClient(app)

def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["catalog_items_count"] > 0

def test_info_endpoint(client):
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.json()
    assert "app_name" in data
    assert "evaluation_summary" in data

def test_text_search_endpoint(client):
    payload = {
        "query": "black running shoes under $120",
        "top_k": 5
    }
    res = client.post("/api/search/text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) <= 5
    assert "timings" in data
    assert "extracted_constraints" in data
    assert data["extracted_constraints"]["color"] == "Black"
    assert data["extracted_constraints"]["max_price"] == 120.0

def test_baseline_search_endpoint(client):
    payload = {
        "query": "blue athletic jacket",
        "top_k": 5
    }
    res = client.post("/api/search/baseline", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) <= 5
