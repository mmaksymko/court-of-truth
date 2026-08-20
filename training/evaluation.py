from itertools import pairwise
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from court.forensics.text_transforms import transform_for
from training.config import DetectorConfig
from training.pipelines import build_pipeline
from training.transfer.splits import seeded_outlet_split


def split_data(
    cfg: DetectorConfig, frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cfg.group_field and cfg.group_field in frame.columns:
        grouped, _ = seeded_outlet_split(
            frame,
            group_field=cfg.group_field,
            seed=seed,
        )
        parts = tuple(grouped[name] for name in ("train", "val", "test"))
    elif cfg.split_field and cfg.split_field in frame.columns:
        by = frame[cfg.split_field]
        actual = {str(value) for value in by.unique()}
        if actual != {"train", "val", "test"}:
            raise ValueError(f"{cfg.id}: expected train/val/test splits, got {sorted(actual)}")
        parts = (frame[by == "train"], frame[by == "val"], frame[by == "test"])
    else:
        train, holdout = train_test_split(
            frame, test_size=0.3, stratify=frame[cfg.label_field], random_state=seed
        )
        val, test = train_test_split(
            holdout, test_size=0.5, stratify=holdout[cfg.label_field], random_state=seed
        )
        parts = (train, val, test)
    for name, part in zip(("train", "val", "test"), parts, strict=True):
        if part[cfg.label_field].nunique() != 2:
            raise ValueError(f"{cfg.id}: {name} split must contain both labels")
    return parts


def calibrate(cfg: DetectorConfig, train: pd.DataFrame, seed: int) -> CalibratedClassifierCV:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(
        build_pipeline(cfg, seed),
        cv=cv,
        ensemble=False,
    ).fit(train["_x"], train[cfg.label_field])


def tune_threshold(
    model: CalibratedClassifierCV,
    frame: pd.DataFrame,
    cfg: DetectorConfig,
    positive_index: int,
) -> float:
    proba = model.predict_proba(frame["_x"])[:, positive_index]
    truth = (frame[cfg.label_field] == cfg.positive_label).to_numpy()
    if cfg.threshold_candidates == "exact":
        ordered = sorted(set(proba))
        grid = [
            0.0,
            *((left + right) / 2 for left, right in pairwise(ordered)),
            1.0,
        ]
    else:
        grid = [round(0.2 + 0.025 * i, 3) for i in range(25)]
    rows = [
        (
            threshold,
            f1_score(truth, proba >= threshold, average="macro"),
            recall_score(truth, proba >= threshold),
        )
        for threshold in grid
    ]
    if cfg.threshold_objective == "target_recall":
        eligible = [row for row in rows if row[2] >= cfg.target_recall]
        rows = eligible or rows
    return max(rows, key=lambda row: (row[1], row[2], -row[0]))[0]


def evaluate(
    model: CalibratedClassifierCV,
    frame: pd.DataFrame,
    cfg: DetectorConfig,
    positive_index: int,
    threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    proba = model.predict_proba(frame["_x"])[:, positive_index]
    truth = (frame[cfg.label_field] == cfg.positive_label).to_numpy()
    predicted = proba >= threshold
    negative = next(
        label for label in frame[cfg.label_field].unique() if label != cfg.positive_label
    )
    metrics = {
        "f1_macro": round(f1_score(truth, predicted, average="macro"), 4),
        "brier_score": round(brier_score_loss(truth, proba), 4),
    }
    recall = {
        cfg.positive_label: round(recall_score(truth, predicted, pos_label=True), 4),
        negative: round(recall_score(truth, predicted, pos_label=False), 4),
    }
    return metrics, recall


def gold_recall(
    model: CalibratedClassifierCV,
    cfg: DetectorConfig,
    positive_index: int,
    threshold: float,
    data_root: Path,
) -> float:
    gold = pd.read_csv(data_root / "jeansa_gold.csv")
    text = gold[cfg.text_field].fillna("").map(lambda value: transform_for(cfg.id, value))
    text = text[text.str.len() > 0]
    proba = model.predict_proba(text)[:, positive_index]
    return round(float((proba >= threshold).mean()), 4)
