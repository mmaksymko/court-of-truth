from court.forensics.registry import LoadedDetector
from court.forensics.schemas import DetectorMeta


def make_detector(
    detector_id: str,
    scope: str,
    labels: tuple[str, str],
    positive: str,
    probability: float,
    threshold: float = 0.5,
) -> LoadedDetector:
    meta = DetectorMeta(
        id=detector_id,
        scope=scope,
        backend="sklearn",
        labels=labels,
        positive_label=positive,
        threshold=threshold,
        version="test",
        metrics={},
    )
    return LoadedDetector(
        meta=meta,
        predict=lambda _text: probability,
        evidence=lambda _text: ["мітка"],
    )
