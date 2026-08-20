from court.forensics.predict import _model_text, run_detector
from court.forensics.schemas import AnalyzeRequest, OkResult, SkippedResult
from tests.fakes import make_detector

LONG_BODY = "речення про новини та важливі події дня " * 20
LONG_TITLE = "названо просту звичку яка змінить усе назавжди"


def test_flagged_when_probability_above_threshold():
    detector = make_detector("jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.9)
    result = run_detector(detector, AnalyzeRequest(text=LONG_BODY))
    assert isinstance(result, OkResult)
    assert result.flagged
    assert result.label == "sponsored"
    assert result.evidence


def test_flag_boundary_is_inclusive():
    detector = make_detector(
        "jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.5, threshold=0.5
    )
    result = run_detector(detector, AnalyzeRequest(text=LONG_BODY))
    assert result.flagged


def test_not_flagged_keeps_negative_label_and_no_evidence():
    detector = make_detector("jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.1)
    result = run_detector(detector, AnalyzeRequest(text=LONG_BODY))
    assert not result.flagged
    assert result.label == "editorial"
    assert result.evidence == []


def test_not_low_confidence_when_far_from_threshold():
    detector = make_detector("jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.9)
    result = run_detector(detector, AnalyzeRequest(text=LONG_BODY), margin=0.1)
    assert isinstance(result, OkResult)
    assert not result.low_confidence


def test_low_confidence_near_threshold():
    detector = make_detector("clickbait", "title", ("clickbait", "neutral"), "clickbait", 0.55)
    result = run_detector(detector, AnalyzeRequest(title=LONG_TITLE, text=LONG_BODY), margin=0.1)
    assert isinstance(result, OkResult)
    assert result.low_confidence


def test_short_body_is_low_confidence_not_skipped():
    detector = make_detector("jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.9)
    result = run_detector(detector, AnalyzeRequest(text="занадто коротко"))
    assert isinstance(result, OkResult)
    assert result.low_confidence
    assert result.caveats
    assert "коротший" in result.caveats[0]


def test_per_detector_floor_differs():
    body = "слово " * 70  # above jeansa floor (60), below ai floor (100)
    ai = make_detector("ai_generated", "body", ("ai_generated", "human_news"), "ai_generated", 0.9)
    jeansa = make_detector("jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.9)
    ai_result = run_detector(ai, AnalyzeRequest(text=body))
    jeansa_result = run_detector(jeansa, AnalyzeRequest(text=body))
    assert ai_result.caveats and ai_result.low_confidence
    assert not jeansa_result.caveats


def test_empty_scope_is_skipped():
    detector = make_detector("clickbait", "title", ("clickbait", "neutral"), "clickbait", 0.9)
    result = run_detector(detector, AnalyzeRequest(text=LONG_BODY))
    assert isinstance(result, SkippedResult)
    assert result.reason == "no_text"


def test_title_folded_into_body_input():
    request = AnalyzeRequest(title="Заголовок", text="тіло статті")
    jeansa = make_detector("jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.9).meta
    assert _model_text(jeansa, request) == "Заголовок тіло статті"


def test_gate_counts_body_only_not_title():
    # 58-word body is below jeansa floor 60; a title must NOT rescue it (builders gated on body)
    body = "слово " * 58
    detector = make_detector("jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.9)
    with_title = AnalyzeRequest(title="один два три чотири пять шість", text=body)
    result = run_detector(detector, with_title)
    assert isinstance(result, OkResult)
    assert result.caveats  # below floor because the body (not title) is counted
