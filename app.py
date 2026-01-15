import json
import os
import pickle
import random

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

print("Loading models...")
with open("models/model_baseline.pkl", "rb") as fh:
    model_baseline = pickle.load(fh)
with open("models/model_target.pkl", "rb") as fh:
    model_target = pickle.load(fh)
print("Models loaded")


class ListingData(BaseModel):
    number_of_reviews: int
    listing_price: float
    host_is_superhost_int: int
    availability_365: int
    session_count: int
    price_vs_neighbourhood: float
    room_type: str


@app.post("/predict")
def predict(data: ListingData) -> dict:
    input_data = data.model_dump()

    df = pd.DataFrame([input_data])

    if random.random() > 0.5:
        model = model_baseline
        model_name = "Baseline Model"
        group = "A"
    else:
        model = model_target
        model_name = "Target Model"
        group = "B"

    features = [
        "number_of_reviews",
        "listing_price",
        "price_vs_neighbourhood",
        "host_is_superhost_int",
        "availability_365",
        "session_count",
        "room_type",
    ]

    X = df[features]
    prediction = int(model.predict(X)[0])

    return {
        "prediction": prediction,
        "model_used": model_name,
        "ab_test_group": group,
    }
