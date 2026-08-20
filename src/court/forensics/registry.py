import platform
from collections.abc import Callable, Collection, Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import numpy
import scipy
import sklearn

from court.forensics import occlusion, promo_signals
from court.forensics.backends import sklearn_backend, transformers_backend
from court.forensics.hashing import path_sha256
from court.forensics.manifest import Manifest
from court.forensics.schemas import DetectorMeta

_EVIDENCE_TOP_K = 5


class LoadedDetector(NamedTuple):
    meta: DetectorMeta
    predict: Callable[[str], float]
    evidence: Callable[[str], list[str]]


def load_registry(
    artifacts_dir: Path, enabled_detectors: Collection[str] = ()
) -> Mapping[str, LoadedDetector]:
    detectors: dict[str, LoadedDetector] = {}
    for manifest_path in sorted(artifacts_dir.glob("*/manifest.json")):
        manifest = Manifest.model_validate_json(manifest_path.read_text())
        if enabled_detectors and manifest.detector_id not in enabled_detectors:
            continue
        _check_versions(manifest)
        detectors[manifest.detector_id] = _build(manifest, manifest_path.parent)
    if not detectors:
        raise RuntimeError(f"no detectors found under {artifacts_dir}")
    return MappingProxyType(detectors)


def _build(manifest: Manifest, artifact_dir: Path) -> LoadedDetector:
    if manifest.backend == "transformers":
        return _build_transformer(manifest, artifact_dir)

    model_path = artifact_dir / "model.joblib"
    _check_model_hash(manifest, model_path)
    model = sklearn_backend.load(model_path)
    classes = list(model.classes_)
    if manifest.positive_label not in classes:
        raise RuntimeError(f"{manifest.detector_id}: positive_label not in {classes}")
    positive_index = classes.index(manifest.positive_label)

    meta = DetectorMeta(
        id=manifest.detector_id,
        scope=manifest.scope,
        backend=manifest.backend,
        labels=manifest.labels,
        positive_label=manifest.positive_label,
        threshold=manifest.decision_threshold,
        version=manifest.version,
        metrics=manifest.metrics,
    )
    predict = partial(sklearn_backend.predict_proba, model, positive_index=positive_index)
    return LoadedDetector(meta=meta, predict=predict, evidence=_evidence(manifest, predict))


def _evidence(manifest: Manifest, predict: Callable[[str], float]) -> Callable[[str], list[str]]:
    if manifest.detector_id == "jeansa":
        # Promotional exhibits read better than an occlusion word-list for jeansa.
        return partial(promo_signals.promo_evidence, k=_EVIDENCE_TOP_K)
    return partial(occlusion.occlusion_evidence, predict, k=_EVIDENCE_TOP_K)


def _build_transformer(manifest: Manifest, artifact_dir: Path) -> LoadedDetector:
    model_path = artifact_dir / "model"
    _check_model_hash(manifest, model_path)
    positive_index = int(str(manifest.hyperparams.get("positive_index", 1)))
    max_length = int(str(manifest.hyperparams.get("max_length", 64)))
    model = transformers_backend.load(
        str(model_path),
        max_length=max_length,
        positive_index=positive_index,
    )
    meta = DetectorMeta(
        id=manifest.detector_id,
        scope=manifest.scope,
        backend=manifest.backend,
        labels=manifest.labels,
        positive_label=manifest.positive_label,
        threshold=manifest.decision_threshold,
        version=manifest.version,
        metrics=manifest.metrics,
    )
    predict = partial(transformers_backend.predict_proba, model)
    return LoadedDetector(meta=meta, predict=predict, evidence=_evidence(manifest, predict))


def _check_model_hash(manifest: Manifest, model_path: Path) -> None:
    actual = path_sha256(model_path)
    if actual != manifest.model_sha256:
        raise RuntimeError(f"{manifest.detector_id}: model hash mismatch")


def _check_versions(manifest: Manifest) -> None:
    runtime = (sklearn.__version__, numpy.__version__, scipy.__version__)
    trained = (manifest.sklearn_version, manifest.numpy_version, manifest.scipy_version)
    if runtime != trained:
        raise RuntimeError(
            f"{manifest.detector_id}: version mismatch trained={trained} runtime={runtime}"
        )
    if manifest.python_version.split(".")[:2] != platform.python_version().split(".")[:2]:
        raise RuntimeError(f"{manifest.detector_id}: python minor mismatch")
