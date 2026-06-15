# test_health.py — tests for the CI/CD pipeline to execute

from fastapi.testclient import TestClient
import sys
import os

# Add the backend directory to Python's module search path
# so we can import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # import your FastAPI app

# TestClient is FastAPI's built-in test helper.
# It creates a fake HTTP client that calls your app directly,
# without needing a real server running.
client = TestClient(app)


def test_health_endpoint_returns_200():
    """
    Tests that GET /health returns HTTP 200.
    This is the most fundamental check — is the app alive?
    Every production service needs a health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200


def test_metrics_endpoint_exists():
    # Tests that GET /metrics returns HTTP 200 prometheus scrapes this endpoint. If it's broken, monitoring will fail.
    response = client.get("/metrics")
    assert response.status_code == 200


def test_health_response_has_status_field():
    #Tests that the /health response body contains a 'status' key.
    response = client.get("/health")
    data = response.json()
    assert "status" in data