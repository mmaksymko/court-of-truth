from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    return {
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 8),
        "positive_recall": round(float(recall_score(truth, predicted, zero_division=0)), 8),
        "positive_precision": round(float(precision_score(truth, predicted, zero_division=0)), 8),
        "negative_recall": round(float(recall_score(~truth, ~predicted, zero_division=0)), 8),
    }
