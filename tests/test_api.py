import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_happy_path_returns_expected_fields():
    response = client.post("/predict", json={"text": "I forgot my password and need to reset it"})
    assert response.status_code == 200
    body = response.json()
    assert "intent" in body
    assert "confidence" in body
    assert "status" in body
    assert "department" in body
    assert "priority" in body
    assert "requires_human" in body


def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_whitespace_only_text():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


def test_predict_fraud_keyword_forces_review_even_if_confident():
    response = client.post(
        "/predict",
        json={"text": "Someone hacked my account and stole money, this is fraud"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REVIEW"
    assert body["requires_human"] is True
    assert body["priority"] == "high"
