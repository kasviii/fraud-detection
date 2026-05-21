import os
import uuid
import time
import logging

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
logger.info("Loading models...")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_fraud.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_config = joblib.load(os.path.join(MODELS_DIR, "feature_config.pkl"))
autoencoder = tf.keras.models.load_model(os.path.join(MODELS_DIR, "autoencoder_fraud.keras"))
FEATURES = feature_config["features"]
AE_THRESHOLD = feature_config["ae_threshold"]
XGB_WEIGHT = feature_config["xgb_weight"]
AE_WEIGHT = feature_config["ae_weight"]

logger.info("All models loaded — XGBoost + Autoencoder ensemble ready")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of predictions made",
    ["result", "risk_level"],
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Time taken to produce a prediction",
)
FRAUD_SCORE_HIST = Histogram(
    "fraud_score_distribution",
    "Distribution of ensemble fraud scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection using an "
        "XGBoost + Autoencoder ensemble with Prometheus monitoring."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    xgb_score: float
    ae_score: float
    ensemble_score: float
    is_fraud: bool
    risk_level: str
    processing_time_ms: float
    model: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def engineer_features(data: dict) -> pd.DataFrame:
    data["Amount_scaled"] = float(scaler.transform([[data["Amount"]]])[0][0])
    data["Time_scaled"] = float((data["Time"] - 94813) / 47488)
    data["Amount_log"] = float(np.log1p(data["Amount"]))
    data["Hour"] = float((data["Time"] // 3600) % 24)
    return pd.DataFrame([{f: data.get(f, 0.0) for f in FEATURES}])


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    if prob < 0.6:
        return "MEDIUM"
    if prob < 0.8:
        return "HIGH"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Fraud Detection API",
        "status": "running",
        "version": "2.0.0",
        "models": ["XGBoost", "Autoencoder"],
        "ensemble_weights": {
            "xgboost": XGB_WEIGHT,
            "autoencoder": AE_WEIGHT,
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "xgboost_loaded": xgb_model is not None,
        "autoencoder_loaded": autoencoder is not None,
        "ae_threshold": AE_THRESHOLD,
        "features": len(FEATURES),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    start_time = time.time()

    data = transaction.model_dump()
    df = engineer_features(data)

    # XGBoost score
    xgb_score = float(xgb_model.predict_proba(df)[0][1])

    # Autoencoder anomaly score
    reconstructed = autoencoder.predict(df, verbose=0)
    ae_error = float(np.mean(np.power(df.values - reconstructed, 2)))
    ae_score = float(min(ae_error / (AE_THRESHOLD * 3), 1.0))

    # Ensemble
    ensemble_score = XGB_WEIGHT * xgb_score + AE_WEIGHT * ae_score
    is_fraud = ensemble_score > 0.5
    risk_level = get_risk_level(ensemble_score)
    processing_time = (time.time() - start_time) * 1000

    # Metrics
    PREDICTIONS_TOTAL.labels(
        result="fraud" if is_fraud else "legitimate",
        risk_level=risk_level,
    ).inc()
    PREDICTION_LATENCY.observe(processing_time / 1000)
    FRAUD_SCORE_HIST.observe(ensemble_score)

    logger.info(
        "Prediction: fraud=%s ensemble=%.4f xgb=%.4f ae=%.4f time=%.1fms",
        is_fraud, ensemble_score, xgb_score, ae_score, processing_time,
    )

    return PredictionResponse(
        transaction_id=str(uuid.uuid4()),
        fraud_probability=round(ensemble_score, 4),
        xgb_score=round(xgb_score, 4),
        ae_score=round(ae_score, 4),
        ensemble_score=round(ensemble_score, 4),
        is_fraud=is_fraud,
        risk_level=risk_level,
        processing_time_ms=round(processing_time, 2),
        model="XGBoost + Autoencoder Ensemble",
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()