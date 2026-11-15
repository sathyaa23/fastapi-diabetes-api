"""Tests for the API endpoints using FastAPI's test client."""
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

PATIENT = {
    "age": 59, "sex": 2, "bmi": 32.1, "blood_pressure": 101,
    "total_cholesterol": 157, "ldl": 93.2, "hdl": 38,
    "cholesterol_hdl_ratio": 4, "log_triglycerides": 4.86, "blood_sugar": 87,
}


def test_health():
    """Health check returns 200 and confirms the model is loaded."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model_loaded": True}


def test_model_info():
    """Model info lists the 10 features and the test metrics."""
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert len(body["features"]) == 10
    assert "r2" in body["test_metrics"]


def test_predict_valid():
    """A valid patient gets a score and a risk level."""
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["progression_score"] > 0
    assert body["risk_level"] in {"low", "moderate", "high"}


def test_predict_missing_field():
    """Leaving out a field returns a 422 validation error."""
    incomplete = {k: v for k, v in PATIENT.items() if k != "bmi"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_predict_out_of_range():
    """An unrealistic age is rejected."""
    assert client.post("/predict", json={**PATIENT, "age": 200}).status_code == 422


def test_predict_invalid_sex():
    """Sex must be coded as 1 or 2."""
    assert client.post("/predict", json={**PATIENT, "sex": 3}).status_code == 422


def test_batch_predict():
    """A batch returns one prediction per patient."""
    r = client.post("/predict/batch", json={"patients": [PATIENT, {**PATIENT, "bmi": 22.0}]})
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_batch_empty():
    """An empty batch is rejected."""
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422
