import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock all models before importing app
mock_xgb = MagicMock()
mock_xgb.predict_proba.return_value = np.array([[0.95, 0.05]])

mock_autoencoder = MagicMock()
mock_autoencoder.predict.return_value = np.zeros((1, 32))

mock_scaler = MagicMock()
mock_scaler.transform.return_value = np.array([[0.5]])

mock_feature_config = {
    "features": [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled", "Amount_log", "Hour"],
    "ae_threshold": 1.6544,
    "xgb_weight": 0.7,
    "ae_weight": 0.3,
    "ensemble_threshold": 0.5
}

with patch("joblib.load", side_effect=[mock_xgb, mock_scaler, mock_feature_config]), \
     patch("tensorflow.keras.models.load_model", return_value=mock_autoencoder):
    from src.api.main import app

client = TestClient(app)

SAMPLE_TX = {
    "Time": 406, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
    "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
    "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
    "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_TX)
    assert response.status_code == 200

def test_predict_response_has_required_fields():
    response = client.post("/predict", json=SAMPLE_TX)
    data = response.json()
    for field in ["transaction_id", "fraud_probability", "xgb_score",
                  "ae_score", "ensemble_score", "is_fraud", "risk_level",
                  "processing_time_ms", "model"]:
        assert field in data

def test_predict_invalid_input_returns_422():
    response = client.post("/predict", json={"Time": 406})
    assert response.status_code == 422

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200

def test_unique_transaction_ids():
    r1 = client.post("/predict", json=SAMPLE_TX)
    r2 = client.post("/predict", json=SAMPLE_TX)
    assert r1.json()["transaction_id"] != r2.json()["transaction_id"]