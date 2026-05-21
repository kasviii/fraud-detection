import requests
import random
import time
import json
from datetime import datetime

API_URL = "http://localhost:8000/predict"

# Real fraud patterns from the dataset
LEGITIMATE_TEMPLATES = [
    {"V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
     "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
     "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
     "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
     "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
     "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62},
    {"V1": 1.19, "V2": 0.27, "V3": 0.17, "V4": 0.45, "V5": 0.06,
     "V6": -0.08, "V7": -0.08, "V8": 0.09, "V9": 0.59, "V10": 0.29,
     "V11": -0.26, "V12": -0.08, "V13": -0.43, "V14": 0.01, "V15": 0.41,
     "V16": -0.09, "V17": 0.80, "V18": 0.09, "V19": 0.09, "V20": 0.01,
     "V21": -0.01, "V22": 0.13, "V23": -0.01, "V24": 0.14, "V25": 0.08,
     "V26": 0.09, "V27": 0.00, "V28": 0.01, "Amount": 2.69},
]

FRAUD_TEMPLATES = [
    {"V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99, "V5": -0.52,
     "V6": -1.43, "V7": -2.77, "V8": -2.77, "V9": -0.33, "V10": -2.67,
     "V11": -0.07, "V12": -3.54, "V13": 1.92, "V14": -4.29, "V15": 0.39,
     "V16": -1.14, "V17": -2.83, "V18": -0.17, "V19": 0.84, "V20": 0.40,
     "V21": 0.86, "V22": -0.13, "V23": -0.18, "V24": 0.13, "V25": -0.34,
     "V26": 0.17, "V27": 0.13, "V28": -0.02, "Amount": 378.66},
]

def add_noise(template, noise_level=0.1):
    """Add small random noise to a transaction template"""
    noisy = template.copy()
    for key in noisy:
        if key != "Amount":
            noisy[key] += random.uniform(-noise_level, noise_level)
    noisy["Amount"] = max(0.01, noisy["Amount"] * random.uniform(0.5, 2.0))
    return noisy

def send_transaction(tx_type="legitimate"):
    """Send a single transaction to the API"""
    if tx_type == "fraud":
        template = random.choice(FRAUD_TEMPLATES)
    else:
        template = random.choice(LEGITIMATE_TEMPLATES)

    transaction = add_noise(template)
    transaction["Time"] = random.uniform(0, 172792)

    try:
        response = requests.post(API_URL, json=transaction, timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = "🚨 FRAUD" if data["is_fraud"] else "✅ OK"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} | "
                  f"Risk: {data['risk_level']:8s} | "
                  f"Score: {data['ensemble_score']:.4f} | "
                  f"Time: {data['processing_time_ms']:.0f}ms")
            return data
        else:
            print(f"❌ API error: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection error: {e}")

def simulate(
    duration_seconds=300,
    rate_per_minute=30,
    fraud_rate=0.1
):
    """
    Simulate realistic traffic with configurable fraud rate.
    duration_seconds: how long to run
    rate_per_minute: transactions per minute
    fraud_rate: fraction of transactions that are fraud (0.1 = 10%)
    """
    delay = 60 / rate_per_minute
    end_time = time.time() + duration_seconds

    total = 0
    frauds = 0

    print(f"🚀 Starting traffic simulation")
    print(f"   Duration: {duration_seconds}s")
    print(f"   Rate: {rate_per_minute} tx/min")
    print(f"   Fraud rate: {fraud_rate*100:.0f}%")
    print("-" * 60)

    while time.time() < end_time:
        tx_type = "fraud" if random.random() < fraud_rate else "legitimate"
        result = send_transaction(tx_type)

        if result:
            total += 1
            if result["is_fraud"]:
                frauds += 1

        time.sleep(delay)

    print("-" * 60)
    print(f"✅ Simulation complete!")
    print(f"   Total transactions: {total}")
    print(f"   Detected fraud: {frauds}")
    print(f"   Detection rate: {frauds/max(total,1)*100:.1f}%")

if __name__ == "__main__":
    simulate(
        duration_seconds=300,  # run for 5 minutes
        rate_per_minute=20,    # 20 transactions per minute
        fraud_rate=0.15        # 15% fraud rate
    )