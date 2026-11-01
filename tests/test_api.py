
from fastapi.testclient import TestClient

from careflow.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "careflow-ai"
    assert isinstance(data["model_available"], bool)


def test_model_info():
    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "careflow-admission-xgboost"
    assert data["model_version"] == "1.0.0"
    assert data["threshold"] == 0.20
    assert data["research_only"] is True
    assert len(data["features"]) == 14
    assert data["model_available"] is True



import math


VALID_REQUEST = {
    "age": 65,
    "sex": 1,
    "residence": 1,
    "triage_immediacy": 3,
    "temperature_f": 98.6,
    "pulse": 92,
    "respiratory_rate": 20,
    "systolic_bp": 140,
    "diastolic_bp": 85,
    "oxygen_saturation": 97,
    "pain_scale": 4,
    "visit_month": 11,
    "visit_day": 1,
    "arrival_time": "1430",
}


def test_predict_success():
    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200

    data = response.json()

    assert math.isfinite(data["admission_probability"])
    assert 0 <= data["admission_probability"] <= 1
    assert data["threshold"] == 0.20
    assert data["research_only"] is True
    assert data["predicted_admission"] == (
        data["admission_probability"] >= data["threshold"]
    )


def test_predict_invalid_age():
    payload = {**VALID_REQUEST, "age": -5}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_missing_optional_values():
    payload = {
        **VALID_REQUEST,
        "temperature_f": None,
        "pulse": None,
        "triage_immediacy": None,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert 0 <= response.json()["admission_probability"] <= 1


def test_predict_rejects_unknown_fields():
    payload = {**VALID_REQUEST, "patient_name": "Example"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_invalid_arrival_time():
    payload = {**VALID_REQUEST, "arrival_time": "2965"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422