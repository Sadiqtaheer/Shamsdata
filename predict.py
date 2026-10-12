"""
Small inference helper for the transaction success model.

Example:
    python predict.py
"""
import joblib
import pandas as pd
import numpy as np
from config import MODEL_DIR

bundle = joblib.load(MODEL_DIR / "transaction_success.joblib")
model, features = bundle["model"], bundle["features"]

# Replace this sample with values from a new transaction.
sample = {
    "amount": 1000,
    "vending_medium": "Datastation",
    "transaction_channel": np.nan,
    "service": "data",
    "balance_before": 5000,
    "hour": 12,
    "dayofweek": 2,
    "day": 15,
    "month": 9,
    "is_weekend": 0,
}
row = pd.DataFrame([{c: sample.get(c, np.nan) for c in features}])
# Match saved preprocessing types, including fields omitted from this example.
for name, transformer, columns in model.named_steps["preprocess"].transformers_:
    if name == "num":
        row[columns] = row[columns].apply(pd.to_numeric, errors="coerce")
    elif name == "cat":
        row[columns] = row[columns].astype(object).where(row[columns].notna(), np.nan)
model.named_steps["model"].n_jobs = 1
classes = list(model.classes_)
p = model.predict_proba(row)[0, classes.index(1)] if 1 in classes else 0.0
print(f"Predicted success probability: {p:.2%}")
print(f"Predicted failure probability: {1-p:.2%}")
