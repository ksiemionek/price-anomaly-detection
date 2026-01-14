import json

import pandas as pd
from sklearn.metrics import classification_report

from utilities import create_labels_advanced, spank

LOG_FILE = "logs/prediction_logs.jsonl"


def prepare_data(df):
    if "availability_365" not in df.columns:
        df["availability_365"] = 0
        df["availability_rate"] = 0
    else:
        df["availability_rate"] = df["availability_365"] / 365.0

    required_columns = [
        "number_of_reviews",
        "avg_price_calendar",
        "max_price_calendar",
        "price_vs_neighbourhood",
        "session_count",
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    return df


def evaluate_logs():
    data = []
    print(f"Reading logs from {LOG_FILE}...")

    try:
        with open(LOG_FILE, "r") as fh:
            for line in fh:
                if line.strip():
                    entry = json.loads(line)
                    flat_entry = entry.copy()
                    if "input_data" in flat_entry:
                        input_data = flat_entry["input_data"]
                        del flat_entry["input_data"]
                        flat_entry.update(input_data)
                    data.append(flat_entry)
    except FileNotFoundError:
        print("Log file not found. 🥺")
        for i in range(2):
            spank()
        return
    if not data:
        print("No log entries found. 😭")
        return

    try:
        df = pd.DataFrame(data)
        df = prepare_data(df)
        df = create_labels_advanced(df)
    except Exception as e:
        print(f"Error creating DataFrame: {e} 😭😭😭")
        return

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
                target_names=["crewmate", "impostor"],
                zero_division=0,
            )
        )

        tp = ((group_df["prediction"] == 1) & (group_df["is_suspicious"] == 1)).sum()
        fp = ((group_df["prediction"] == 1) & (group_df["is_suspicious"] == 0)).sum()

        total_impostors = group_df["is_suspicious"].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_impostors if total_impostors > 0 else 0.0

        print(f"- Total interventions: {tp + fp}")
        print(f"- Actual sus count: {total_impostors}")
        print(f"- Valid (TP): {tp}")
        print(f"- Invalid (FP): {fp}")
        print(f"- Missed (FN): {total_impostors - tp}")
        print(f"= Precision: {precision:.2%}")
        print(f"= Recall: {recall:.2%}")


if __name__ == "__main__":
    evaluate_logs()
