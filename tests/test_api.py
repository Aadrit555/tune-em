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


def test_model_flags_mock_backend_explicitly():
    res = client.get("/model")
    body = res.json()
    assert body["backend"] == "mock"
    assert body["is_mock"] is True
    assert "warning" in body


def test_decide_rejects_oversized_question():
    from anydecision.serving.app import ServiceLimits
    assert ServiceLimits().max_question_chars > 0
    payload = {
        "question": {
            "text": "x" * (ServiceLimits().max_question_chars + 1),
            "options": [
                {"key": "allow", "label": "allow"},
                {"key": "deny", "label": "deny"},
            ],
        },
        "level": "L0",
    }
    res = client.post("/decide", json=payload)
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "invalid_request"


def test_batch_rejects_oversized_batch():
    from anydecision.serving.app import ServiceLimits
    n = ServiceLimits().max_batch_size + 1
    payload = {
        "questions": [
            {"text": "ok?", "options": [{"key": "a", "label": "a"}, {"key": "b", "label": "b"}]}
            for _ in range(n)
        ],
        "level": "L0",
    }
    res = client.post("/batch", json=payload)
    assert res.status_code == 422


def test_calibrate_requires_admin_key():
    from fastapi.testclient import TestClient as TC
    from anydecision.serving.app import create_app
    admin_app = TC(create_app(DecisionEngine(model="mock"), admin_api_key="secret"))
    payload = {"dataset": [], "method": "temperature"}
    # No key -> 403
    res = admin_app.post("/calibrate", json=payload)
    assert res.status_code == 403
    # Wrong key -> 403
    res = admin_app.post("/calibrate", json=payload, headers={"X-API-Key": "wrong"})
    assert res.status_code == 403
    # Disabled when no key configured -> 403 admin_disabled
    res = client.post("/calibrate", json=payload)
    assert res.status_code == 403
    assert res.json()["error"]["code"] == "admin_disabled"


def test_internal_errors_do_not_leak():
    # Unknown level is a client error with safe envelope, not a raw traceback.
    payload = {
        "question": {
            "text": "Is this allowed?",
            "options": [{"key": "allow", "label": "allow"}, {"key": "deny", "label": "deny"}],
        },
        "level": "L99",
    }
    res = client.post("/decide", json=payload)
    assert res.status_code == 422
    assert "error" in res.json()
    assert "request_id" in res.json()["error"]

