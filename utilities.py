import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest


def load_data():
    listings = pd.read_csv(f"data/listings.csv.zst")
    calendar = pd.read_csv(f"data/calendar.csv.zst")
    sessions = pd.read_csv(f"data/sessions.csv.zst")
    return listings, calendar, sessions


def prepare_data(listings, calendar, sessions):
    listings = listings.copy()
    calendar = calendar.copy()

    listings["price"] = (
        listings["price"].replace(r"[\$,]", "", regex=True).astype(float)
    )
    listings = listings.rename(columns={"id": "listing_id"})
    avg_rating = listings["review_scores_rating"].mean()
    listings["review_scores_rating"] = listings["review_scores_rating"].fillna(
        avg_rating
    )
    listings["host_is_superhost_int"] = (
        listings["host_is_superhost"].map({"t": 1, "f": 0}).fillna(0)
    )
    listings["availability_365"] = listings["availability_365"].fillna(0)

    calendar["price"] = (
        calendar["price"].replace(r"[\$,]", "", regex=True).astype(float)
    )
    calendar["available_int"] = calendar["available"].map({"t": 1, "f": 0})
    calendar_stats = (
        calendar.groupby("listing_id")
        .agg({"price": ["mean", "max", "std"], "available_int": "mean"})
        .reset_index()
    )
    calendar_stats.columns = [
        "listing_id",
        "avg_price_calendar",
        "max_price_calendar",
        "std_price_calendar",
        "availability_rate",
    ]

    sessions_stats = (
        sessions.groupby("listing_id").size().reset_index(name="session_count")
    )

    df = pd.merge(listings, calendar_stats, on="listing_id", how="left")
    df = pd.merge(df, sessions_stats, on="listing_id", how="left")

    df["price"] = df["price"].fillna(df["avg_price_calendar"])
    df["avg_price_calendar"] = df["avg_price_calendar"].fillna(df["price"])
    df["max_price_calendar"] = df["max_price_calendar"].fillna(df["price"])
    df["std_price_calendar"] = df["std_price_calendar"].fillna(0)
    df["session_count"] = df["session_count"].fillna(0)

    grouped = df.groupby("neighbourhood_cleansed")["max_price_calendar"]
    df["price_vs_neighbourhood"] = (
        df["max_price_calendar"] - grouped.transform("mean")
    ) / (grouped.transform("std") + 1)
    df["price_vs_neighbourhood"] = df["price_vs_neighbourhood"].fillna(0)

    return df.dropna(subset=["price"])


def create_labels_advanced(df):
    labeling_features = [
        "max_price_calendar",
        "price_vs_neighbourhood",
        "session_count",
        "availability_rate",
    ]

    X_labeling = df[labeling_features].fillna(0)

    iso_forest = IsolationForest(n_estimators=100, contamination=0.075, random_state=42)
    predictions = iso_forest.fit_predict(X_labeling)
    df["is_suspicious"] = np.where(predictions == -1, 1, 0)

    df.loc[df["max_price_calendar"] < 500, "is_suspicious"] = 0
    # df.loc[df['price_vs_neighbourhood'] < 1.25, 'is_suspicious'] = 0
    df.loc[df["number_of_reviews"] > 15, "is_suspicious"] = 0
    df.loc[df["availability_rate"] < 0.3, "is_suspicious"] = 0

    print(f"{df['is_suspicious'].sum()} suspicious offers")
    return df


def get_model_pipeline(model_type):
    numeric_cols = [
        "number_of_reviews",
        "review_scores_rating",
        "host_is_superhost_int",
        "availability_365",
        "avg_price_calendar",
        "std_price_calendar",
        "session_count",
        "price_vs_neighbourhood",
    ]
    categorical_cols = ["room_type"]

    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    cat_pipe = OneHotEncoder()

    preprocessor = make_column_transformer(
        (num_pipe, numeric_cols), (cat_pipe, categorical_cols)
    )

    if model_type == 'baseline':
        model = LogisticRegression(class_weight='balanced')
    elif model_type == 'target':
        model = RandomForestClassifier(class_weight='balanced')
    else:
        raise ValueError("Wrong model type")

    return make_pipeline(preprocessor, model)
