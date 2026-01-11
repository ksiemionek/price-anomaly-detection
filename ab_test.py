import pickle

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

import utilities as siemionek_kacper


def run_ab_tests():
    listings, calendar, sessions = siemionek_kacper.load_data()
    df = siemionek_kacper.prepare_data(listings, calendar, sessions)

    df = siemionek_kacper.create_labels_advanced(df)

    with open("model_target.pkl", "rb") as fh:
        model = pickle.load(fh)

    group_ctrl, group_exp = train_test_split(df, test_size=0.5, random_state=67)

    features = [
        "number_of_reviews",
        "review_scores_rating",
        "host_is_superhost_int",
        "availability_365",
        "avg_price_calendar",
        "std_price_calendar",
        "session_count",
        "price_vs_neighbourhood",
        "room_type",
    ]

    X_exp = group_exp[features]
    y_true = group_exp["is_suspicious"]

    y_pred = model.predict(X_exp)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print("-" * 10 + "RESULTS" + "-" * 10)
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")


if __name__ == "__main__":
    run_ab_tests()
