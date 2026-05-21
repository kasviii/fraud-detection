# 🔍 Real-Time Fraud Detection System

![CI](https://github.com/kasviii/fraud-detection/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Docker](https://img.shields.io/badge/docker-compose-blue)

A production-grade real-time credit card fraud detection system using an **XGBoost + Autoencoder ensemble**, served via **FastAPI**, monitored with **Prometheus + Grafana**, containerized with **Docker Compose**, and tested with **GitHub Actions CI/CD**.

---

## 🏗️ Architecture

```
Credit Card Transaction
        │
        ▼
  FastAPI REST API
  (real-time inference)
        │
   ┌────┴────┐
   │         │
XGBoost   Autoencoder
(0.70)    (0.30)
   │         │
   └────┬────┘
        │
   Ensemble Score
        │
   ┌────┴────────────┐
   │                 │
Prometheus      Prediction
 Metrics        Response
   │
Grafana
Dashboard
```

---

## 📊 Model Performance

| Model | ROC-AUC | Precision (Fraud) | Recall (Fraud) |
|---|---|---|---|
| XGBoost | 0.9842 | 0.59 | 0.86 |
| Autoencoder | 0.9331 | — | — |
| **Ensemble** | **0.9705** | **0.72** | **0.85** |

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- 492 fraud cases (0.17%)
- Class imbalance handled with SMOTE

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost, TensorFlow/Keras Autoencoder |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| API | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| Drift Detection | Evidently AI |
| Containerization | Docker Compose |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Language | Python 3.11 |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11
- Git

### 1. Clone the repository
```bash
git clone https://github.com/kasviii/fraud-detection.git
cd fraud-detection
```

### 2. Download model files
Download the pre-trained models from Google Drive and place them in the `models/` folder:
- `xgboost_fraud.pkl`
- `autoencoder_fraud.keras`
- `scaler.pkl`
- `feature_config.pkl`

> To retrain from scratch, open `notebooks/Untitled1.ipynb` in Google Colab and run all cells.

### 3. Start all services
```bash
docker-compose up --build
```

This starts:
- **API** at http://localhost:8000
- **Prometheus** at http://localhost:9090
- **Grafana** at http://localhost:3000 (admin/admin)

### 4. Test the API
Visit http://localhost:8000/docs for the interactive Swagger UI.

Sample prediction request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406, "V1": -1.36, "V2": -0.07, "V3": 2.54,
    "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24,
    "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55,
    "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
    "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11,
    "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13,
    "V28": -0.02, "Amount": 149.62
  }'
```

Sample response:
```json
{
  "transaction_id": "d3cae45f-d6ca-4910-9ff5-91ca558e24fa",
  "fraud_probability": 0.0305,
  "xgb_score": 0.0002,
  "ae_score": 0.101,
  "ensemble_score": 0.0305,
  "is_fraud": false,
  "risk_level": "LOW",
  "processing_time_ms": 83.5,
  "model": "XGBoost + Autoencoder Ensemble"
}
```

---

## 📁 Project Structure

```
fraud-detection/
├── .github/
│   └── workflows/
│       └── tests.yml          # GitHub Actions CI/CD
├── data/
│   ├── eda_overview.png       # EDA charts
│   ├── xgboost_results.png    # XGBoost evaluation
│   └── ensemble_results.png   # Ensemble comparison
├── models/                    # Model files (not in git — see setup)
│   ├── xgboost_fraud.pkl
│   ├── autoencoder_fraud.keras
│   ├── scaler.pkl
│   └── feature_config.pkl
├── notebooks/
│   └── Untitled1.ipynb        # Training notebook (Google Colab)
├── reports/                   # Evidently drift reports (generated)
├── scripts/
│   └── simulate_traffic.py    # Traffic simulator
├── src/
│   ├── api/
│   │   └── main.py            # FastAPI application
│   ├── monitoring/
│   │   └── drift.py           # Evidently drift detection
│   └── pipeline/              # Feature engineering
├── tests/
│   ├── test_api.py            # Full tests (requires model files)
│   └── test_api_ci.py         # CI tests (mocked models)
├── docker-compose.yml
├── dockerfile
├── prometheus.yml
└── requirements.txt
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info and status |
| GET | `/health` | Health check + model status |
| POST | `/predict` | Real-time fraud prediction |
| GET | `/metrics` | Prometheus metrics |

---

## 📈 Monitoring

### Grafana Dashboard
After starting with Docker Compose:
1. Go to http://localhost:3000 (admin/admin)
2. Add Prometheus data source: `http://prometheus:9090`
3. Import the dashboard

Metrics tracked:
- Total predictions (fraud vs legitimate)
- Prediction latency (ms)
- Fraud score distribution
- Predictions per minute

### Traffic Simulator
Simulate realistic transaction traffic:
```bash
python scripts/simulate_traffic.py
```
Sends 20 transactions/minute with 15% fraud rate for 5 minutes.

### Drift Detection
Generate an Evidently data drift report:
```bash
python src/monitoring/drift.py
```
Opens an interactive HTML report comparing current vs reference distributions.

---

## 🧪 Running Tests

```bash
# With model files present
pytest tests/test_api.py -v

# Without model files (CI mode)
pytest tests/test_api_ci.py -v
```

8 tests covering: root, health, predict, response fields, validation, metrics, latency, unique IDs.

---

## 🔬 Model Details

### XGBoost
- 300 estimators, max depth 6, learning rate 0.05
- Trained on SMOTE-balanced data (227k vs 227k)
- Evaluated on Area Under Precision-Recall Curve (AUCPR)
- Top features: V14, V10, V4, V12 (PCA-transformed)

### Autoencoder
- Architecture: 32→64→32→16→32→64→32
- Trained on legitimate transactions only
- Anomaly score = MSE reconstruction error
- Threshold: 1.6544 (95th percentile of legitimate errors)

### Ensemble
- Weighted combination: XGBoost (70%) + Autoencoder (30%)
- Final score > 0.5 → fraud
- Risk levels: LOW (<0.3), MEDIUM (0.3-0.6), HIGH (0.6-0.8), CRITICAL (>0.8)

---

## ⚠️ Limitations & Future Work

**Current limitations:**
- Model files not versioned in git (stored separately)
- No authentication on API endpoints
- Single-instance deployment

**Planned improvements:**
- [ ] React frontend dashboard
- [ ] Model versioning with MLflow or DVC
- [ ] Authentication with JWT tokens
- [ ] Kubernetes deployment
- [ ] Batch prediction endpoint
- [ ] Alert system for high-risk transactions

---

## 🤖 AI Tools Used

Claude (Anthropic) assisted with pipeline design, code structure, and documentation. All model training, evaluation, and system integration was executed and verified manually.

---

## 📄 License

MIT License
