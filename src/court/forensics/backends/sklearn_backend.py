from pathlib import Path
from typing import Any

import joblib

Estimator = Any


def load(model_path: Path) -> Estimator:
    return joblib.load(model_path)


def predict_proba(model: Estimator, text: str, positive_index: int) -> float:
    return float(model.predict_proba([text])[0, positive_index])
