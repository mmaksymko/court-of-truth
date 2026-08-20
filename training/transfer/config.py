from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_LOG = REPO_ROOT / "experiments" / "transfer_log.jsonl"
SEED = 42
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

EstimatorFactory = Callable[[int], Any]


@dataclass(frozen=True)
class ModelConfig:
    analyzer: Literal["word", "char", "char_wb"] = "char_wb"
    ngram_min: int = 3
    ngram_max: int = 5
    min_df: int = 3
    max_features: int = 200_000
    c: float = 2.0
    class_weight: str | None = None
    calibration: Literal["none", "sigmoid_cv"] = "none"
    sublinear_tf: bool = True
    norm: Literal["l1", "l2"] | None = "l2"


@dataclass(frozen=True)
class ThresholdPolicy:
    objective: Literal["macro_f1", "target_recall"] = "macro_f1"
    target_recall: float = 0.95
    candidates: Literal["exact", "probability_grid"] = "exact"
    grid_min: float = 0.20
    grid_max: float = 0.80
    grid_step: float = 0.025


@dataclass(frozen=True)
class DedupConfig:
    fuzzy_threshold: float = 0.90
    shingle_size: int = 3
    max_shingle_df: int = 25
    min_shared_shingles: int = 2
    min_length_ratio: float = 0.50
    audit_examples: int = 20


class ExternalLeakageError(RuntimeError):
    pass


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_JEANSA_MODEL_CONFIG = ModelConfig(
    c=1.0,
    class_weight="balanced",
    calibration="sigmoid_cv",
)
DEFAULT_THRESHOLD_POLICY = ThresholdPolicy()
DEFAULT_JEANSA_THRESHOLD_POLICY = ThresholdPolicy(candidates="probability_grid")
DEFAULT_DEDUP_CONFIG = DedupConfig()
