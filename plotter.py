import os

import matplotlib.pyplot as plt
import seaborn as sns

from utilities import load_data, prepare_data


def plot_correlation_heatmap(df):
    cols_to_corr = [
        "max_price_calendar",
        "number_of_reviews",
        "review_scores_rating",
        "session_count",
        "price_vs_neighbourhood",
        "availability_365",
    ]

    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols_to_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")

    plt.savefig(
        os.path.join(output_dir, "correlation_heatmap.png"),
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    listings, calendar, sessions = load_data()
    df = prepare_data(listings, calendar, sessions)

    plot_correlation_heatmap(df)
