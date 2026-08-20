import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy
import scipy
import sklearn

from court.forensics.hashing import path_sha256
from court.forensics.manifest import Manifest
from training.config import DetectorConfig

_HYPERPARAMS = (
    "analyzer",
    "ngram",
    "min_df",
    "max_features",
    "c",
    "class_weight",
    "group_field",
    "threshold_objective",
    "target_recall",
    "threshold_candidates",
    "model_name",
    "epochs",
    "learning_rate",
    "batch_size",
    "max_length",
    "weight_decay",
    "threshold_plateau_tolerance",
)


def build_manifest(
    cfg: DetectorConfig,
    labels: list[str],
    sizes: tuple[int, int, int],
    threshold: float,
    metrics: dict[str, float],
    per_class_recall: dict[str, float],
    source: Path,
    model_path: Path,
    repo_root: Path,
    seed: int,
    extra_hyperparams: dict[str, object] | None = None,
) -> Manifest:
    now = datetime.now(UTC)
    if cfg.backend == "transformers":
        implementation = {
            "positive_index": 1,
            "calibration": "softmax",
            "threshold_selection": (
                "highest threshold within "
                f"{cfg.threshold_plateau_tolerance} of validation macro-F1 maximum"
            ),
        }
    else:
        implementation = {
            "sublinear_tf": True,
            "dual": True,
            "calibration": "CalibratedClassifierCV(cv=5, ensemble=False)",
            "threshold_selection": (
                f"{cfg.threshold_candidates}, {cfg.threshold_objective} on validation only"
            ),
        }
    return Manifest(
        detector_id=cfg.id,
        version=now.strftime("%Y-%m-%dT%H%M%SZ"),
        trained_at=now.isoformat(),
        backend=cfg.backend,
        sklearn_version=sklearn.__version__,
        numpy_version=numpy.__version__,
        scipy_version=scipy.__version__,
        python_version=platform.python_version(),
        seed=seed,
        hyperparams={
            **cfg.model_dump(mode="json", include=set(_HYPERPARAMS)),
            **implementation,
            **(extra_hyperparams or {}),
        },
        source_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model_sha256=path_sha256(model_path),
        split_field=cfg.split_field,
        n_train=sizes[0],
        n_val=sizes[1],
        n_test=sizes[2],
        labels=(labels[0], labels[1]),
        label_map={label: index for index, label in enumerate(labels)},
        positive_label=cfg.positive_label,
        scope=cfg.scope,
        decision_threshold=threshold,
        metrics=metrics,
        per_class_recall=per_class_recall,
        language_note="~44% ru" if cfg.id == "manipulation" else "",
        git_commit=_git_commit(repo_root),
    )


def _git_commit(repo_root: Path) -> str:
    if not (repo_root / ".git").exists():
        return ""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
