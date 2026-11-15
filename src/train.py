"""Train the diabetes progression model and save it with its metrics.

Run from the project root with: python -m src.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data import FEATURES, load_data, split_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def fit_model(X_train, y_train):
    """Fit a random forest regressor on the training data."""
    model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, random_state=42)
    return model.fit(X_train, y_train)


def evaluate(model, X_test, y_test) -> dict:
    """Return R squared and mean absolute error on the test set."""
    pred = model.predict(X_test)
    return {
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 2),
    }


def risk_thresholds(y_train) -> dict:
    """Use the 33rd and 67th percentiles of the training target as cut offs
    for low, moderate and high progression."""
    low, high = np.percentile(y_train, [33, 67])
    return {"low_below": round(float(low), 1), "high_above": round(float(high), 1)}


def main():
    """Train, evaluate and save the model and metrics."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "model_type": "RandomForestRegressor",
            "features": list(FEATURES.values()),
            "test_metrics": metrics,
            "risk_thresholds": risk_thresholds(y_train),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }, f, indent=2)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
