import os
import pickle
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import utilities


listings, sessions = utilities.load_data()
df = utilities.prepare_data(listings, sessions)
df = utilities.create_labels_advanced(df)

suspects = df[df["is_suspicious"] == 1].copy()

cols = [
    "listing_id",
    "number_of_reviews",
    "listing_price",
    "price_vs_neighbourhood",
    "host_is_superhost_int",
    "availability_365",
    "session_count",
    "room_type",
]


available_cols = [c for c in cols if c in suspects.columns]

os.makedirs("logs", exist_ok=True)
suspects[available_cols].to_csv("logs/suspects.csv", index=False)
print(f"Saved {len(suspects)} suspects to logs/suspects.csv")

synthetic_data = [utilities.generate_random_suspicious() for _ in range(80)]
synthetic_data += [utilities.generate_random_tricky() for _ in range(80)]

df_synthetic = pd.DataFrame(synthetic_data)

for col in df.columns:
    if col not in df_synthetic.columns:
        df_synthetic[col] = 0 if pd.api.types.is_numeric_dtype(df[col]) else "Unknown"

df = pd.concat([df, df_synthetic], ignore_index=True)

features = [
    "number_of_reviews",
    "listing_price",
    "price_vs_neighbourhood",
    "host_is_superhost_int",
    "availability_365",
    "session_count",
    "room_type",
]

target = "is_suspicious"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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
