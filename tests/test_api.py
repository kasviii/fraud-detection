import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.main import app

client = TestClient(app)

# Sample legitimate transaction
LEGITIMATE_TX = {
    "Time": 406, "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": 2.53634673796914, "V4": 1.37815522427443, "V5": -0.338320769942518,
    "V6": 0.462387777762292, "V7": 0.239598554061257, "V8": 0.0986979012610507,
    "V9": 0.363786969611213, "V10": 0.0907941719789316, "V11": -0.551599533260813,
    "V12": -0.617800855762348, "V13": -0.991389847235408, "V14": -0.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478, "V17": 0.207971241929242,
    "V18": 0.0257905801521169, "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899, "V23": -0.110473910188767,
    "V24": 0.0669280749146731, "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215, "Amount": 149.62
}

# Sample fraudulent transaction (high anomaly values)
FRAUD_TX = {
    "Time": 406, "V1": -3.0, "V2": -3.0, "V3": -3.0, "V4": -3.0,
    "V5": -3.0, "V6": -3.0, "V7": -3.0, "V8": -3.0, "V9": -3.0,
    "V10": -3.0, "V11": -3.0, "V12": -3.0, "V13": -3.0, "V14": -3.0,
    "V15": -3.0, "V16": -3.0, "V17": -3.0, "V18": -3.0, "V19": -3.0,
    "V20": -3.0, "V21": -3.0, "V22": -3.0, "V23": -3.0, "V24": -3.0,
    "V25": -3.0, "V26": -3.0, "V27": -3.0, "V28": -3.0, "Amount": 9999.99
}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["version"] == "2.0.0"
    print("✅ test_root passed")

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["xgboost_loaded"] == True
    assert data["autoencoder_loaded"] == True
    print("✅ test_health passed")

def test_predict_legitimate():
    response = client.post("/predict", json=LEGITIMATE_TX)
    assert response.status_code == 200
    data = response.json()
    assert "transaction_id" in data
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_level" in data
    assert data["is_fraud"] == False
    assert data["risk_level"] == "LOW"
    assert 0.0 <= data["fraud_probability"] <= 1.0
    print(f"✅ test_predict_legitimate passed — prob={data['fraud_probability']}")

def test_predict_response_fields():
    response = client.post("/predict", json=LEGITIMATE_TX)
    assert response.status_code == 200
    data = response.json()
    required_fields = [
        "transaction_id", "fraud_probability", "xgb_score",
        "ae_score", "ensemble_score", "is_fraud",
        "risk_level", "processing_time_ms", "model"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
    print("✅ test_predict_response_fields passed")

def test_predict_invalid_input():
    response = client.post("/predict", json={"Time": 406, "Amount": 100})
    assert response.status_code == 422  # validation error
    print("✅ test_predict_invalid_input passed")

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "fraud_predictions_total" in response.text
    print("✅ test_metrics_endpoint passed")

def test_processing_time_reasonable():
    response = client.post("/predict", json=LEGITIMATE_TX)
    data = response.json()
    assert data["processing_time_ms"] < 5000  # should be under 5 seconds
    print(f"✅ test_processing_time passed — {data['processing_time_ms']}ms")

def test_transaction_id_unique():
    r1 = client.post("/predict", json=LEGITIMATE_TX)
    r2 = client.post("/predict", json=LEGITIMATE_TX)
    assert r1.json()["transaction_id"] != r2.json()["transaction_id"]
    print("✅ test_transaction_id_unique passed")