from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.transfer.config import (
    DATA_ROOT,
    DEFAULT_DEDUP_CONFIG,
    DEFAULT_JEANSA_MODEL_CONFIG,
    DEFAULT_JEANSA_THRESHOLD_POLICY,
    DEFAULT_LOG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_THRESHOLD_POLICY,
    SEED,
    SPLIT_NAMES,
    DedupConfig,
    EstimatorFactory,
    ExternalLeakageError,
    ModelConfig,
    ThresholdPolicy,
)
from training.transfer.dedup import (
    decontaminate_against_external,
    deduplicate_cross_split,
    find_fuzzy_matches,
)
from training.transfer.metrics import binary_metrics
from training.transfer.models import built_in_factory, default_threshold, positive_scores
from training.transfer.splits import (
    assert_group_isolation,
    predefined_split,
    prepare_frame,
    seeded_outlet_split,
    validate_binary_parts,
)
from training.transfer.thresholds import choose_threshold


def evaluate_clickbait(
    frame: pd.DataFrame,
    *,
    estimator_factory: EstimatorFactory,
    model_config: Mapping[str, Any],
    threshold_policy: ThresholdPolicy = DEFAULT_THRESHOLD_POLICY,
    dedup_config: DedupConfig = DEFAULT_DEDUP_CONFIG,
    seed: int = SEED,
) -> dict[str, Any]:
    prepared = prepare_frame(frame, detector_id="clickbait")
    grouped, split_audit = seeded_outlet_split(prepared, seed=seed)
    grouped, dedup_audit = deduplicate_cross_split(
        grouped,
        text_field="text",
        config=dedup_config,
    )
    assert_group_isolation(grouped, "outlet")

    model = estimator_factory(seed)
    model.fit(grouped["train"]["_x"], grouped["train"]["label"])
    val_scores = positive_scores(model, grouped["val"]["_x"], "clickbait")
    val_truth = grouped["val"]["label"].eq("clickbait").to_numpy()
    threshold, val_selection = choose_threshold(val_scores, val_truth, threshold_policy)
    test_scores = positive_scores(model, grouped["test"]["_x"], "clickbait")
    test_truth = grouped["test"]["label"].eq("clickbait").to_numpy()

    baseline_threshold = default_threshold(model)
    result = {
        "detector": "clickbait",
        "protocol": "outlet-held-out",
        "seed": seed,
        "model_config": dict(model_config),
        "threshold_policy": asdict(threshold_policy),
        "decision_threshold": round(threshold, 10),
        "threshold_selected_on": "outlet-held-out val only",
        "val_selection_metrics": val_selection,
        "ood_test": binary_metrics(test_truth, test_scores >= threshold),
        "baseline_default_threshold": round(baseline_threshold, 10),
        "baseline_default_test": binary_metrics(test_truth, test_scores >= baseline_threshold),
        "split_audit": split_audit,
        "dedup_audit": dedup_audit,
        "n_train": len(grouped["train"]),
        "n_val": len(grouped["val"]),
        "n_test": len(grouped["test"]),
    }

    if set(frame["split"].astype(str).unique()) == set(SPLIT_NAMES):
        id_parts = predefined_split(prepared)
        id_parts, id_dedup = deduplicate_cross_split(
            id_parts,
            text_field="text",
            config=dedup_config,
        )
        id_model = estimator_factory(seed)
        id_model.fit(id_parts["train"]["_x"], id_parts["train"]["label"])
        id_val_scores = positive_scores(id_model, id_parts["val"]["_x"], "clickbait")
        id_threshold, id_selection = choose_threshold(
            id_val_scores,
            id_parts["val"]["label"].eq("clickbait").to_numpy(),
            threshold_policy,
        )
        id_test_scores = positive_scores(id_model, id_parts["test"]["_x"], "clickbait")
        result["in_distribution"] = {
            **binary_metrics(
                id_parts["test"]["label"].eq("clickbait").to_numpy(),
                id_test_scores >= id_threshold,
            ),
            "decision_threshold": round(id_threshold, 10),
            "threshold_selected_on": "predefined internal val only",
            "val_selection_metrics": id_selection,
            "split_rows": {name: len(part) for name, part in id_parts.items()},
            "dedup_audit": id_dedup,
        }
    return result


def evaluate_jeansa(
    frame: pd.DataFrame,
    gold_path: Path,
    *,
    estimator_factory: EstimatorFactory,
    model_config: Mapping[str, Any],
    threshold_policy: ThresholdPolicy,
    dedup_config: DedupConfig = DEFAULT_DEDUP_CONFIG,
    seed: int = SEED,
    include_ablations: bool = True,
) -> dict[str, Any]:
    prepared = prepare_frame(frame, detector_id="jeansa")
    parts = predefined_split(prepared)
    historical_parts = {name: part.copy() for name, part in parts.items()}
    parts, dedup_audit = deduplicate_cross_split(
        parts,
        text_field="text",
        config=dedup_config,
    )
    parts_before_gold_dedup = {name: part.copy() for name, part in parts.items()}

    gold_text_only = pd.read_csv(gold_path, usecols=["text"]).dropna()
    gold_text_only["text"] = gold_text_only["text"].astype(str)
    parts, gold_dedup_audit = decontaminate_against_external(
        parts,
        gold_text_only[["text"]],
        text_field="text",
        config=dedup_config,
    )
    validate_binary_parts(parts, "label")

    model = estimator_factory(seed)
    model.fit(parts["train"]["_x"], parts["train"]["label"])
    val_scores = positive_scores(model, parts["val"]["_x"], "sponsored")
    threshold, val_selection = choose_threshold(
        val_scores,
        parts["val"]["label"].eq("sponsored").to_numpy(),
        threshold_policy,
    )
    internal_scores = positive_scores(model, parts["test"]["_x"], "sponsored")
    internal_truth = parts["test"]["label"].eq("sponsored").to_numpy()

    gold = pd.read_csv(gold_path)
    if set(gold["label"].dropna().astype(str).unique()) != {"sponsored"}:
        raise ValueError("jeansa gold must contain positives only")
    gold = prepare_frame(gold, detector_id="jeansa")
    internal = pd.concat([parts[name] for name in SPLIT_NAMES], axis=0)
    gold_overlap = find_fuzzy_matches(
        gold,
        internal,
        text_field="text",
        config=dedup_config,
    )
    if gold_overlap:
        raise ExternalLeakageError(
            "jeansa gold overlaps internal data; refusing to report a contaminated metric"
        )
    gold_scores = positive_scores(model, gold["_x"], "sponsored")

    result = {
        "detector": "jeansa",
        "protocol": "external-positive-gold",
        "seed": seed,
        "model_config": dict(model_config),
        "threshold_policy": asdict(threshold_policy),
        "decision_threshold": round(threshold, 10),
        "threshold_selected_on": "predefined internal val only; gold loaded afterwards",
        "val_selection_metrics": val_selection,
        "in_distribution": binary_metrics(internal_truth, internal_scores >= threshold),
        "gold_recall": round(float(np.mean(gold_scores >= threshold)), 8),
        "n_gold": len(gold),
        "gold_overlap_audit": {
            "method": "same normalized SHA-256 + fuzzy matcher as internal splits",
            "matches": 0,
            "action_if_nonzero": "raise ExternalLeakageError",
        },
        "dedup_audit": dedup_audit,
        "gold_dedup_audit": gold_dedup_audit,
        "split_rows": {name: len(part) for name, part in parts.items()},
    }
    if include_ablations:
        result["gold_train_overlap_ablation"] = _jeansa_contamination_ablation(
            parts_before_gold_dedup,
            gold,
            estimator_factory=estimator_factory,
            threshold_policy=threshold_policy,
            seed=seed,
            purpose="no gold-protected train-row removal; internal cross-split dedup kept",
        )
        result["historical_no_dedup_baseline"] = _jeansa_contamination_ablation(
            historical_parts,
            gold,
            estimator_factory=estimator_factory,
            threshold_policy=threshold_policy,
            seed=seed,
            purpose="baseline reproduction; no internal/gold dedup",
        )
    return result


def _jeansa_contamination_ablation(
    parts: Mapping[str, pd.DataFrame],
    gold: pd.DataFrame,
    *,
    estimator_factory: EstimatorFactory,
    threshold_policy: ThresholdPolicy,
    seed: int,
    purpose: str,
) -> dict[str, Any]:
    model = estimator_factory(seed)
    model.fit(parts["train"]["_x"], parts["train"]["label"])
    val_scores = positive_scores(model, parts["val"]["_x"], "sponsored")
    threshold, selection = choose_threshold(
        val_scores,
        parts["val"]["label"].eq("sponsored").to_numpy(),
        threshold_policy,
    )
    test_scores = positive_scores(model, parts["test"]["_x"], "sponsored")
    gold_scores = positive_scores(model, gold["_x"], "sponsored")
    return {
        "purpose": purpose,
        "selection_firewall": "threshold selected on internal val, never gold",
        "decision_threshold": round(threshold, 10),
        "val_selection_metrics": selection,
        "in_distribution": binary_metrics(
            parts["test"]["label"].eq("sponsored").to_numpy(),
            test_scores >= threshold,
        ),
        "gold_recall": round(float(np.mean(gold_scores >= threshold)), 8),
    }


def run_experiment(
    *,
    experiment_id: str,
    clickbait_config: ModelConfig = DEFAULT_MODEL_CONFIG,
    jeansa_config: ModelConfig = DEFAULT_JEANSA_MODEL_CONFIG,
    clickbait_policy: ThresholdPolicy = DEFAULT_THRESHOLD_POLICY,
    jeansa_policy: ThresholdPolicy = DEFAULT_JEANSA_THRESHOLD_POLICY,
    data_root: Path = DATA_ROOT,
    log_path: Path | None = DEFAULT_LOG,
    notes: str = "",
    include_ablations: bool = True,
) -> dict[str, Any]:
    clickbait = evaluate_clickbait(
        pd.read_csv(data_root / "clickbait.csv"),
        estimator_factory=built_in_factory(clickbait_config),
        model_config=asdict(clickbait_config),
        threshold_policy=clickbait_policy,
    )
    jeansa = evaluate_jeansa(
        pd.read_csv(data_root / "jeansa.csv"),
        data_root / "jeansa_gold.csv",
        estimator_factory=built_in_factory(jeansa_config),
        model_config=asdict(jeansa_config),
        threshold_policy=jeansa_policy,
        include_ablations=include_ablations,
    )
    record = {
        "run_id": str(uuid.uuid4()),
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "notes": notes,
        "clickbait": clickbait,
        "jeansa": jeansa,
    }
    if log_path is not None:
        append_jsonl(log_path, record)
    return record


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if "\n" in encoded:
        raise ValueError("JSONL record unexpectedly contains a literal newline")
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
