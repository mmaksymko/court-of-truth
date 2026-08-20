from collections.abc import Collection, Mapping
from pathlib import Path
from types import MappingProxyType

from court.forensics.registry import LoadedDetector, load_registry

EXPECTED_DETECTORS = ("ai_generated", "clickbait", "jeansa", "mt_translation")


def ensure_registry(
    artifacts_dir: Path,
    expected_detectors: Collection[str],
    enabled_detectors: Collection[str] = (),
) -> Mapping[str, LoadedDetector]:
    return _validated_registry(artifacts_dir, expected_detectors, enabled_detectors)


def _validated_registry(
    artifacts_dir: Path,
    expected_detectors: Collection[str],
    enabled_detectors: Collection[str] = (),
) -> Mapping[str, LoadedDetector]:
    registry = load_registry(artifacts_dir, ())
    actual = set(registry)
    expected = set(expected_detectors)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"artifact set mismatch: missing={missing}, extra={extra}")
    selected = set(enabled_detectors or expected_detectors)
    unknown = sorted(selected - expected)
    if unknown:
        raise RuntimeError(f"unknown enabled detectors: {unknown}")
    return MappingProxyType(
        {
            detector_id: detector
            for detector_id, detector in registry.items()
            if detector_id in selected
        }
    )
