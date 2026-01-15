import os
import pickle
import utilities
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score


OUTPUT_DIR = "plots"


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def plot_correlation_heatmap(df):
    cols_to_corr = [
        "is_suspicious",
        "listing_price",
        "number_of_reviews",
        "session_count",
        "price_vs_neighbourhood",
        "availability_365",
    ]

    existing_cols = [c for c in cols_to_corr if c in df.columns]

    if len(existing_cols) < 2:
        print("Not enough features for correlation heatmap")
        return

    ensure_output_dir()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[existing_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")

    save_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Correlation plot saved to {save_path}")
    plt.close()


def load_models():
    models = {}
    path_base = "models/model_baseline.pkl"
    path_target = "models/model_target.pkl"

    with open(path_base, "rb") as f:
        models["baseline"] = pickle.load(f)
    with open(path_target, "rb") as f:
        models["target"] = pickle.load(f)
    return models


def run_simulation_step(models, impostor_rate, tricky_rate, samples=500):
    data = []
    y_true = []

    for _ in range(samples):
        rand = np.random.random()
        if rand < impostor_rate:
            pack = utilities.generate_random_suspicious()
            data.append(pack["data"])
            y_true.append(1)
        elif rand < (impostor_rate + tricky_rate):
            pack = utilities.generate_random_tricky()
            data.append(pack["data"])
            y_true.append(0)
        else:
            pack = utilities.generate_random_safe()
            data.append(pack["data"])
            y_true.append(0)

    df = pd.DataFrame(data)

    results = {}

    for name, model in models.items():
        y_pred = model.predict(df)
        results[f"{name}_precision"] = precision_score(y_true, y_pred, zero_division=0)
        results[f"{name}_recall"] = recall_score(y_true, y_pred, zero_division=0)

    return results


def plot_sensitivity_analysis():
    models = load_models()
    ensure_output_dir()

    print("Running sensitivity simulation (this may take a moment)...")

    impostor_rates = [0.5, 0.2, 0.1, 0.05, 0.01]
    tricky_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    metrics = [
        "baseline_precision",
        "target_precision",
        "baseline_recall",
        "target_recall",
    ]
    heatmaps = {
        m: pd.DataFrame(index=impostor_rates, columns=tricky_rates, dtype=float)
        for m in metrics
    }

    for i_rate in impostor_rates:
        for t_rate in tricky_rates:
            res = run_simulation_step(models, i_rate, t_rate)
            for m in metrics:
                heatmaps[m].loc[i_rate, t_rate] = res[m]

    plot_cfg = [
        ("baseline_precision", "Baseline Model: PRECISION", "RdYlGn"),
        ("target_precision", "Target Model (Random Forest): PRECISION", "RdYlGn"),
        ("baseline_recall", "Baseline Model: RECALL", "Blues"),
        ("target_recall", "Target Model (Random Forest): RECALL", "Blues"),
    ]

    print("Generating individual plots...")

    for metric, title, cmap in plot_cfg:
        plt.figure(figsize=(9, 7))

        sns.heatmap(
            heatmaps[metric],
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        plt.title(title)
        plt.xlabel("Tricky Rate [%]")
        plt.ylabel("Suspicious Rate [%]")

        filename = f"{metric}_heatmap.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot: {save_path}")

        plt.close()


if __name__ == "__main__":
    try:
        listings, sessions = utilities.load_data()
        df = utilities.prepare_data(listings, sessions)
        df = utilities.create_labels_advanced(df)
        plot_correlation_heatmap(df)
    except Exception as e:
        print(f"data not found: {e}")

    try:
        plot_sensitivity_analysis()
    except Exception as e:
        print(f"models not found: {e}")