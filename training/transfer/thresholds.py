from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score

from training.transfer.config import ThresholdPolicy


def choose_threshold(
    scores: np.ndarray,
    truth: np.ndarray,
    policy: ThresholdPolicy,
) -> tuple[float, dict[str, float]]:
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    thresholds = _candidate_thresholds(scores, policy)
    rows = []
    for threshold in thresholds:
        predicted = scores >= threshold
        rows.append(
            (
                float(threshold),
                float(f1_score(truth, predicted, average="macro", zero_division=0)),
                float(recall_score(truth, predicted, zero_division=0)),
            )
        )

    if policy.objective == "target_recall":
        eligible = [row for row in rows if row[2] >= policy.target_recall]
        candidates = eligible or rows
    else:
        candidates = rows
    threshold, macro_f1, recall = max(
        candidates,
        key=lambda row: (row[1], row[2], -row[0]),
    )
    return threshold, {
        "macro_f1": round(macro_f1, 8),
        "positive_recall": round(recall, 8),
    }


def _candidate_thresholds(scores: np.ndarray, policy: ThresholdPolicy) -> np.ndarray:
    if policy.candidates == "probability_grid":
        count = round((policy.grid_max - policy.grid_min) / policy.grid_step)
        return np.round(
            policy.grid_min + np.arange(count + 1) * policy.grid_step,
            12,
        )
    unique = np.unique(scores)
    if len(unique) == 1:
        return np.asarray([np.nextafter(unique[0], -np.inf), np.nextafter(unique[0], np.inf)])
    middle = (unique[:-1] + unique[1:]) / 2
    return np.concatenate(
        (
            [np.nextafter(unique[0], -np.inf)],
            middle,
            [np.nextafter(unique[-1], np.inf)],
        )
    )
