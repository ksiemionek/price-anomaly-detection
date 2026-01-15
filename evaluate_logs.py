import json

import pandas as pd
from sklearn.metrics import classification_report

LOG_FILE = "logs/simulation_results.jsonl"


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

    groups = sorted(df["ab_test_group"].unique())
    for group in groups:
        group_df = df[df["ab_test_group"] == group]
        if group_df.empty:
            continue

        model = group_df["model_used"].iloc[0]

        print(f"\n--- GROUP {group} ({model}) ---")
        print(f"Prediction count: {len(group_df)}")
        print(
            classification_report(
                group_df["is_suspicious"],
                group_df["prediction"],
                labels=[0, 1],
                target_names=["safe", "suspicious"],
                zero_division=0,
            )
        )


if __name__ == "__main__":
    evaluate_logs()
