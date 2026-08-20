from court.forensics.registry import LoadedDetector
from court.forensics.schemas import AnalyzeRequest, DetectorMeta, OkResult, SkippedResult
from court.forensics.text_transforms import transform_for

DEFAULT_MARGIN = 0.1
_TITLE_MIN = 4
_BODY_MIN = {"ai_generated": 100, "mt_translation": 100, "jeansa": 60}
_BODY_DEFAULT = 50


def run_detector(
    detector: LoadedDetector,
    request: AnalyzeRequest,
    margin: float = DEFAULT_MARGIN,
) -> OkResult | SkippedResult:
    meta = detector.meta
    gate = transform_for(meta.id, request.title if meta.scope == "title" else request.text)
    gate_words = len(gate.split())
    if gate_words == 0:
        return SkippedResult(id=meta.id, scope=meta.scope, reason="no_text")

    text = transform_for(meta.id, _model_text(meta, request))
    probability = detector.predict(text)
    flagged = probability >= meta.threshold
    negative_label = next(label for label in meta.labels if label != meta.positive_label)

    floor = _min_words(meta)
    caveats: list[str] = []
    if gate_words < floor:
        # below the shortest text the detector was trained on: still report, but flag it
        caveats.append(
            f"текст коротший за навчальний мінімум ({gate_words} < {floor} слів); "
            "вердикт поза розподілом навчання і менш надійний"
        )
    return OkResult(
        id=meta.id,
        scope=meta.scope,
        label=meta.positive_label if flagged else negative_label,
        probability=probability,
        flagged=flagged,
        low_confidence=bool(caveats) or abs(probability - meta.threshold) < margin,
        caveats=caveats,
        evidence=detector.evidence(text) if flagged else [],
    )


def _model_text(meta: DetectorMeta, request: AnalyzeRequest) -> str:
    if meta.scope == "title":
        return request.title
    if not request.title:
        return request.text
    return f"{request.title} {request.text}"


def _min_words(meta: DetectorMeta) -> int:
    if meta.scope == "title":
        return _TITLE_MIN
    return _BODY_MIN.get(meta.id, _BODY_DEFAULT)
