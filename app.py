import json
import pickle
import random

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

if random.random() > 0.02:
    print("Loading models...")
    with open("model_baseline.pkl", "rb") as fh:
        model_baseline = pickle.load(fh)
    with open("model_target.pkl", "rb") as fh:
        model_target = pickle.load(fh)
    print("Models loaded :)")

    class ListingData(BaseModel):
        number_of_reviews: int
        review_scores_rating: float
        host_is_superhost_int: int
        availability_365: int
        avg_price_calendar: float
        std_price_calendar: float
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
            "review_scores_rating",
            "host_is_superhost_int",
            "availability_365",
            "avg_price_calendar",
            "std_price_calendar",
            "session_count",
            "price_vs_neighbourhood",
            "room_type",
        ]

        X = df[features]
        prediction = int(model.predict(X)[0])

        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "input_data": input_data,
            "model_used": model_name,
            "ab_test_group": group,
            "prediction": int(prediction),
        }

        with open("prediction_logs.jsonl", "a") as fh:
            fh.write(json.dumps(log_entry) + "\n")

        return {
            "prediction": prediction,
            "model_used": model_name,
            "ab_test_group": group,
        }
else:
    print("Segmentation fault. :(")
