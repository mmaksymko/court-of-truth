import hashlib
import platform
from pathlib import Path

import joblib
import numpy
import scipy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from training.config import DetectorConfig
from training.pipelines import build_pipeline

from court.forensics.manifest import Manifest


def tiny_model():
    config = DetectorConfig(
        id="tiny",
        source_csv=Path("x"),
        text_field="t",
        label_field="l",
        scope="body",
        positive_label="pos",
        analyzer="word",
        ngram=(1, 1),
        min_df=1,
    )
    x = ["clean neutral news report"] * 8 + ["buy now spam offer deal"] * 8
    y = ["neg"] * 8 + ["pos"] * 8
    pipeline = build_pipeline(config, 42).fit(x, y)
    return CalibratedClassifierCV(FrozenEstimator(pipeline)).fit(x, y)


def manifest(
    sklearn_version: str,
    model_sha256: str = "0" * 64,
    python_version: str = "",
) -> Manifest:
    return Manifest(
        detector_id="tiny",
        version="test",
        trained_at="now",
        backend="sklearn",
        sklearn_version=sklearn_version,
        numpy_version=numpy.__version__,
        scipy_version=scipy.__version__,
        python_version=python_version or platform.python_version(),
        seed=42,
        hyperparams={},
        source_csv_sha256="x",
        model_sha256=model_sha256,
        n_train=16,
        n_val=4,
        n_test=4,
        labels=("neg", "pos"),
        label_map={"neg": 0, "pos": 1},
        positive_label="pos",
        scope="body",
        decision_threshold=0.5,
        metrics={},
        per_class_recall={},
    )


def write_artifact(tmp: Path, artifact_manifest: Manifest, *, valid_hash: bool = True) -> None:
    (tmp / "tiny").mkdir(parents=True)
    model_path = tmp / "tiny" / "model.joblib"
    joblib.dump(tiny_model(), model_path)
    if valid_hash:
        artifact_manifest = artifact_manifest.model_copy(
            update={"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
        )
    (tmp / "tiny" / "manifest.json").write_text(artifact_manifest.model_dump_json())
