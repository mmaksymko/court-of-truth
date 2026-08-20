from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.transfer.config import DATA_ROOT, DEFAULT_LOG, ModelConfig, ThresholdPolicy
from training.transfer.experiments import run_experiment


def _load_model_config(raw: str | None, default: ModelConfig) -> ModelConfig:
    if raw is None:
        return default
    payload = json.loads(raw) if raw.lstrip().startswith("{") else json.loads(Path(raw).read_text())
    return ModelConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="transfer-baseline")
    parser.add_argument("--clickbait-config", help="JSON object or JSON file")
    parser.add_argument("--jeansa-config", help="JSON object or JSON file")
    parser.add_argument("--jeansa-target-recall", type=float)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="skip extra historical/leakage models during broad searches",
    )
    args = parser.parse_args()

    clickbait_config = _load_model_config(args.clickbait_config, ModelConfig())
    jeansa_config = _load_model_config(
        args.jeansa_config,
        ModelConfig(c=1.0, class_weight="balanced", calibration="sigmoid_cv"),
    )
    if args.jeansa_target_recall is None:
        jeansa_policy = ThresholdPolicy(candidates="probability_grid")
    else:
        jeansa_policy = ThresholdPolicy(
            objective="target_recall",
            target_recall=args.jeansa_target_recall,
            candidates="probability_grid",
        )
    record = run_experiment(
        experiment_id=args.experiment_id,
        clickbait_config=clickbait_config,
        jeansa_config=jeansa_config,
        jeansa_policy=jeansa_policy,
        data_root=args.data_root,
        log_path=None if args.no_log else args.log,
        notes=args.notes,
        include_ablations=not args.skip_ablations,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
