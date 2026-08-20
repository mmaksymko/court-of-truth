from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import recall_score
from training.config import DetectorConfig
from training.evaluation import calibrate, evaluate, split_data, tune_threshold


def _config() -> DetectorConfig:
    return DetectorConfig(
        id="tiny",
        source_csv=Path("tiny.csv"),
        text_field="text",
        label_field="label",
        scope="body",
        positive_label="positive",
        analyzer="word",
        ngram=(1, 1),
        min_df=1,
    )


def _frame() -> pd.DataFrame:
    negative = [f"neutral report item {index}" for index in range(30)]
    positive = [f"urgent sponsored offer {index}" for index in range(30)]
    text = negative + positive
    return pd.DataFrame(
        {
            "text": text,
            "_x": text,
            "label": ["negative"] * len(negative) + ["positive"] * len(positive),
        }
    )


def test_calibration_uses_cross_validation_before_threshold_tuning():
    cfg = _config()
    train, val, test = split_data(cfg, _frame(), seed=42)
    model = calibrate(cfg, train, seed=42)

    assert len(model.calibrated_classifiers_) == 1
    positive_index = list(model.classes_).index(cfg.positive_label)
    threshold = tune_threshold(model, val, cfg, positive_index)
    metrics, recall = evaluate(model, test, cfg, positive_index, threshold)

    assert 0.2 <= threshold <= 0.8
    assert 0 <= metrics["brier_score"] <= 1
    assert set(recall) == {"negative", "positive"}


def test_predefined_split_rejects_unknown_partition():
    cfg = _config().model_copy(update={"split_field": "split"})
    frame = _frame()
    frame["split"] = "unknown"

    with pytest.raises(ValueError, match="train/val/test"):
        split_data(cfg, frame, seed=42)


def test_target_recall_threshold_is_selected_only_from_validation():
    cfg = _config().model_copy(
        update={
            "threshold_objective": "target_recall",
            "target_recall": 1.0,
            "threshold_candidates": "exact",
        }
    )
    train, val, _ = split_data(cfg, _frame(), seed=42)
    model = calibrate(cfg, train, seed=42)
    positive_index = list(model.classes_).index(cfg.positive_label)

    threshold = tune_threshold(model, val, cfg, positive_index)
    truth = val[cfg.label_field].eq(cfg.positive_label)
    predicted = model.predict_proba(val["_x"])[:, positive_index] >= threshold

    assert recall_score(truth, predicted) == 1.0
