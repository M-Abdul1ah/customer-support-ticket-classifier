import requests


BASE_URL = "http://127.0.0.1:8000"


def test_root_endpoint():
    response = requests.get(f"{BASE_URL}/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Customer Support Ticket Classifier"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


def test_health_endpoint():
    response = requests.get(f"{BASE_URL}/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "model" in data
    assert "threshold" in data


def test_valid_prediction():
    payload = {
        "text": "I forgot my password"
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["ticket"] == payload["text"]
    assert data["intent"] == "recover_password"

    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["threshold"] <= 1

    assert data["status"] in ["AUTO", "REVIEW"]


def test_payment_prediction():
    payload = {
        "text": "Why was my payment declined?"
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "payment_issue"


def test_shipping_address_prediction():
    payload = {
        "text": "I need to change my delivery address"
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "change_shipping_address"


def test_empty_text_rejected():
    payload = {
        "text": ""
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )

    assert response.status_code == 422


def test_missing_text_rejected():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={}
    )

    assert response.status_code == 422


def test_short_text_rejected():
    payload = {
        "text": "hi"
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )

    assert response.status_code == 422