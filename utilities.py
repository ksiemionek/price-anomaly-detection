import random
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data():
    listings = pd.read_csv("data/listings.csv.zst")
    sessions = pd.read_csv("data/sessions.csv.zst")
    return listings, sessions


def prepare_data(listings, sessions):
    listings = listings.copy()

    listings["listing_price"] = (
        listings["price"].replace(r"[\$,]", "", regex=True).astype(float)
    )
    listings = listings.rename(columns={"id": "listing_id"})
    listings["host_is_superhost_int"] = (
        listings["host_is_superhost"].map({"t": 1, "f": 0}).fillna(0)
    )

    sessions_stats = (
        sessions.groupby("listing_id").size().reset_index(name="session_count")
    )

    df = pd.merge(listings, sessions_stats, on="listing_id", how="left")

    avg_price_neighbourhood = df.groupby("neighbourhood_cleansed")[
        "listing_price"
    ].transform("mean")
    df["price_vs_neighbourhood"] = df["listing_price"] / avg_price_neighbourhood

    df["number_of_reviews"] = df["number_of_reviews"].fillna(0)
    df["session_count"] = df["session_count"].fillna(0)
    df["availability_365"] = df["availability_365"].fillna(0)

    return df.dropna(subset=["listing_price"])


def create_labels_advanced(df):
    labeling_features = [
        "listing_price",
        "price_vs_neighbourhood",
        "session_count",
        "availability_365",
        "number_of_reviews"
    ]

    X_labeling = df[labeling_features].fillna(0)

    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    predictions = iso_forest.fit_predict(X_labeling)
    df["is_suspicious"] = np.where(predictions == -1, 1, 0)

    df.loc[df["listing_price"] < 250, "is_suspicious"] = 0
    df.loc[df["number_of_reviews"] > 15, "is_suspicious"] = 0
    df.loc[df["availability_365"] < 150, "is_suspicious"] = 0

    return df


def get_model_pipeline(model_type):
    numeric_cols = [
        "number_of_reviews",
        "listing_price",
        "price_vs_neighbourhood",
        "host_is_superhost_int",
        "availability_365",
        "session_count",
    ]
    categorical_cols = ["room_type"]

    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    cat_pipe = OneHotEncoder()

    preprocessor = make_column_transformer(
        (num_pipe, numeric_cols), (cat_pipe, categorical_cols)
    )

    if model_type == "baseline":
        model = LogisticRegression(class_weight="balanced")
    elif model_type == "target":
        model = RandomForestClassifier(class_weight="balanced")
    else:
        raise ValueError("Wrong model type")

    return make_pipeline(preprocessor, model)


def generate_random_suspicious():
    return {
        "number_of_reviews": random.randint(0, 15),
        "listing_price": random.uniform(250, 12000),
        "host_is_superhost_int": 1 if random.random() > 0.75 else 0,
        "availability_365": random.randint(200, 365),
        "session_count": random.randint(0, 50),
        "price_vs_neighbourhood": random.uniform(2.0, 15.0),
        "room_type": random.choice(["Entire home/apt", "Private room", "Shared room"]),
        "is_suspicious": 1,
    }


def generate_random_tricky():
    return {
        "number_of_reviews": random.randint(6, 100),
        "listing_price": random.uniform(250, 2000),
        "host_is_superhost_int": 1 if random.random() > 0.3 else 0,
        "availability_365": random.randint(50, 300),
        "session_count": random.randint(5, 50),
        "price_vs_neighbourhood": random.uniform(1.5, 4.0),
        "room_type": random.choice(["Entire home/apt", "Private room", "Shared room"]),
        "is_suspicious": 0,
    }

def generate_random_safe():
    return {
        "number_of_reviews": random.randint(20, 500),
        "listing_price": random.uniform(50, 400),
        "host_is_superhost_int": 1 if random.random() > 0.6 else 0,
        "availability_365": random.randint(0, 200),
        "session_count": random.randint(10, 150),
        "price_vs_neighbourhood": random.uniform(0.01, 1.35),
        "room_type": random.choice(["Entire home/apt", "Private room", "Shared room"]),
        "is_suspicious": 0,
    }