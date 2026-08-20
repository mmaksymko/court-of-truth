from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from training.transfer.config import EstimatorFactory, ModelConfig


def built_in_factory(config: ModelConfig) -> EstimatorFactory:
    def factory(seed: int) -> Any:
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer=config.analyzer,
                        ngram_range=(config.ngram_min, config.ngram_max),
                        min_df=config.min_df,
                        max_features=config.max_features,
                        sublinear_tf=config.sublinear_tf,
                        norm=config.norm,
                    ),
                ),
                (
                    "svc",
                    LinearSVC(
                        C=config.c,
                        class_weight=config.class_weight,
                        dual=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
        if config.calibration == "sigmoid_cv":
            return CalibratedClassifierCV(pipeline, cv=5, ensemble=False)
        return pipeline

    return factory


def positive_scores(model: Any, text: pd.Series, positive_label: str) -> np.ndarray:
    classes = [str(value) for value in model.classes_]
    if positive_label not in classes:
        raise ValueError(f"positive label {positive_label!r} absent from {classes}")
    positive_index = classes.index(positive_label)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(text)[:, positive_index], dtype=float)

    decision = np.asarray(model.decision_function(text), dtype=float)
    if decision.ndim == 1:
        return decision if positive_index == 1 else -decision
    return decision[:, positive_index]


def default_threshold(model: Any) -> float:
    return 0.5 if hasattr(model, "predict_proba") else 0.0
