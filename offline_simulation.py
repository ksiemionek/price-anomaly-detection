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

    # group A, group B
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

    print("-" * 10 + "ANALYTICAL RESULTS" + "-" * 10)
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")

    current_bookings = group_exp["session_count"].sum()
    recovered_bookings = 0

    normal_sessions = df[df["is_suspicious"] == 0]["session_count"].mean()

    y_true_arr = y_true.to_numpy()

    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true_arr[i] == 1:
            # We assume that 25% of notified offers will regulate their price and thus result in a recovered sale
            recovered_bookings += normal_sessions * 0.25

    projected_bookings = current_bookings + recovered_bookings

    growth = projected_bookings - current_bookings
    growth_pct = growth / current_bookings

    print("-" * 10 + "BUSSINESS RESULTS" + "-" * 10)
    print(f"Current bookings (sessions): {current_bookings}")
    print(f"Projected growth: {growth_pct:.2%}")


if __name__ == "__main__":
    run_ab_tests()
