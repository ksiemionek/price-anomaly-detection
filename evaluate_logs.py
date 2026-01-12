import json

import pandas as pd

LOG_FILE = "prediction_logs.jsonl"


def evaluate_logs():
    data = []
    print(f"Reading logs from {LOG_FILE}...")

    try:
        with open(LOG_FILE, "r") as fh:
            for line in fh:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print("Log file not found.")
        return

    if not data:
        print("No log entries found.")
        return

    df = pd.DataFrame(data)
    print("\n" + "-" * 10 + "A/B TEST RESULTS" + "-" * 10)
    print(f"total predictions: {len(df)}")

    group_a = df[df["ab_test_group"] == "A"]  # Baseline
    print(f"Group A (Baseline) predictions: {len(group_a)}")
    group_b = df[df["ab_test_group"] == "B"]  # Target
    print(f"Group B (Target) predictions: {len(group_b)}")

    rate_a = group_a["prediction"].mean() if not group_a.empty else 0
    rate_b = group_b["prediction"].mean() if not group_b.empty else 0

    print("-" * 10 + "PREDICTION RATES" + "-" * 10)
    print(f"Baseline model: {rate_a:.2%}% suspicious predictions")
    print(f"Target model: {rate_b:.2%}% suspicious predictions")


if __name__ == "__main__":
    evaluate_logs()
