"""Unit tests for FastAPI HTTP API endpoints."""

from fastapi.testclient import TestClient
from anydecision.core.engine import DecisionEngine
from anydecision.serving.app import create_app

client = TestClient(create_app(DecisionEngine(model="mock")))


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_model_info():
    res = client.get("/model")
    assert res.status_code == 200
    assert "model_name" in res.json()


def test_decide_endpoint():
    payload = {
        "question": {
            "text": "Is this action allowed?",
            "options": [
                {"key": "allow", "label": "allow"},
                {"key": "deny", "label": "deny"}
            ]
        },
        "level": "L0",
    }
    res = client.post("/decide", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] in ("allow", "deny")
    assert "probabilities" in data
    assert data["confidence"] > 0.0


def test_metrics_endpoint():
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "decision_count" in res.json()
