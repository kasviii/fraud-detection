import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

REFERENCE_STATS = {
    "V1": (-0.0, 1.96), "V2": (0.0, 1.65), "V3": (0.0, 1.52),
    "V4": (0.0, 1.42), "V5": (0.0, 1.38), "V6": (0.0, 1.33),
    "V7": (0.0, 1.24), "V8": (0.0, 1.19), "V9": (0.0, 1.10),
    "V10": (0.0, 1.09), "V11": (0.0, 1.02), "V12": (0.0, 0.99),
    "V13": (0.0, 0.99), "V14": (0.0, 0.96), "V15": (0.0, 0.92),
    "V16": (0.0, 0.88), "V17": (0.0, 0.85), "V18": (0.0, 0.84),
    "V19": (0.0, 0.81), "V20": (0.0, 0.77), "V21": (0.0, 0.73),
    "V22": (0.0, 0.73), "V23": (0.0, 0.62), "V24": (0.0, 0.61),
    "V25": (0.0, 0.52), "V26": (0.0, 0.48), "V27": (0.0, 0.40),
    "V28": (0.0, 0.33), "Amount": (88.35, 250.12)
}

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

def generate_reference_data(n=1000):
    data = {}
    for feature, (mean, std) in REFERENCE_STATS.items():
        data[feature] = np.random.normal(mean, std, n)
    data["Amount"] = np.abs(data["Amount"])
    return pd.DataFrame(data)

def generate_drift_report(current_data: pd.DataFrame, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)

    reference_df = generate_reference_data(len(current_data))
    current_df = current_data[FEATURES].copy() if all(f in current_data.columns for f in FEATURES) else generate_reference_data(len(current_data))

    data_definition = DataDefinition(
        numerical_columns=FEATURES
    )

    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_dataset, current_dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")

    my_eval.save_html(html_path)

    print(f"{'='*50}")
    print(f"DRIFT REPORT — {timestamp}")
    print(f"{'='*50}")
    print(f"HTML report saved: {html_path}")
    print(f"Open it in your browser to see detailed drift analysis!")
    print(f"{'='*50}")

    return {"report_path": html_path, "timestamp": timestamp}

if __name__ == "__main__":
    print("Generating drift report with sample data...")
    sample_current = generate_reference_data(500)
    # Introduce drift
    sample_current["Amount"] = sample_current["Amount"] * 3 + 500
    sample_current["V1"] = sample_current["V1"] - 2.0
    sample_current["V14"] = sample_current["V14"] - 1.5

    result = generate_drift_report(sample_current)
    print(f"\nOpen this file in your browser:")
    print(f"  {result['report_path']}")