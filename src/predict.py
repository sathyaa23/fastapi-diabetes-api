"""Load the saved model once and make predictions."""
import json
from functools import lru_cache

import joblib

from src.train import METRICS_PATH, MODEL_PATH


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the model and metrics from disk the first time, then reuse them.

    The original lab reloaded the model file on every request.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run python -m src.train first.")
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        meta = json.load(f)
    return model, meta


def risk_level(score: float, thresholds: dict) -> str:
    """Turn a progression score into a low, moderate or high label."""
    if score < thresholds["low_below"]:
        return "low"
    if score > thresholds["high_above"]:
        return "high"
    return "moderate"


def predict_data(rows: list) -> list:
    """Predict progression scores and risk levels for a list of feature rows."""
    model, meta = load_artifacts()
    scores = model.predict(rows)
    return [
        {"progression_score": round(float(s), 1),
         "risk_level": risk_level(float(s), meta["risk_thresholds"])}
        for s in scores
    ]
