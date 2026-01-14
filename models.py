import os
import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import utilities

listings, calendar, sessions = utilities.load_data()
df = utilities.prepare_data(listings, calendar, sessions)
df = utilities.create_labels_advanced(df)

suspects = df[df["is_suspicious"] == 1].copy()

cols = [
    "listing_id",
    "neighbourhood_cleansed",
    "room_type",
    "max_price_calendar",
    "price_vs_neighbourhood",
    "number_of_reviews",
    "reviews_per_month",
    "session_count",
]
os.makedirs("logs", exist_ok=True)
suspects[cols].to_csv("logs/suspects.csv", index=False)

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

target = "is_suspicious"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print("Base model (Logistic Regression)")
model_base = utilities.get_model_pipeline("baseline")
model_base.fit(X_train, y_train)
print(classification_report(y_test, model_base.predict(X_test), zero_division=0))

print("\nTarget model (Random Forest)")
model_rf = utilities.get_model_pipeline("target")
model_rf.fit(X_train, y_train)
print(classification_report(y_test, model_rf.predict(X_test), zero_division=0))

os.makedirs("models", exist_ok=True)
with open("models/model_baseline.pkl", "wb") as f:
    pickle.dump(model_base, f)
with open("models/model_target.pkl", "wb") as f:
    pickle.dump(model_rf, f)
