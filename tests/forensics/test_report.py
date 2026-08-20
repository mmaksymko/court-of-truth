from concurrent.futures import ThreadPoolExecutor

import pytest

from court.forensics.report import analyze, analyze_async, risk_summary
from court.forensics.schemas import AnalyzeRequest, OkResult, SkippedResult
from tests.fakes import make_detector

LONG = "речення про новини та важливі події дня " * 20


def _registry():
    return {
        "jeansa": make_detector("jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.9),
        "ai_generated": make_detector(
            "ai_generated", "body", ("ai_generated", "human_news"), "ai_generated", 0.1
        ),
    }


def test_analyze_returns_result_per_detector():
    report = analyze(AnalyzeRequest(text=LONG), _registry())
    assert {r.id for r in report.results} == {"jeansa", "ai_generated"}
    assert report.text_chars == len(LONG)


def test_risk_summary_counts_only_flagged():
    report = analyze(AnalyzeRequest(text=LONG), _registry())
    assert report.risk.flagged_ids == ["jeansa"]
    assert report.risk.flagged_count == 1


def test_risk_summary_ignores_skipped():
    results = [
        OkResult(
            id="a", scope="body", label="x", probability=0.9, flagged=True, low_confidence=False
        ),
        SkippedResult(id="b", scope="body", reason="no_text"),
    ]
    summary = risk_summary(results)
    assert summary.flagged_ids == ["a"]
    assert "b" not in summary.flagged_ids


def test_title_present_reflects_input():
    registry = _registry()
    assert analyze(AnalyzeRequest(title="Заголовок", text=LONG), registry).title_present is True
    assert analyze(AnalyzeRequest(text=LONG), registry).title_present is False
    assert analyze(AnalyzeRequest(title="   ", text=LONG), registry).title_present is False


def test_low_confidence_ids_populated():
    registry = {
        "clickbait": make_detector(
            "clickbait", "title", ("clickbait", "neutral"), "clickbait", 0.55, threshold=0.5
        )
    }
    report = analyze(
        AnalyzeRequest(title="один два три чотири назва", text=LONG), registry, margin=0.1
    )
    assert report.risk.low_confidence_ids == ["clickbait"]


@pytest.mark.asyncio
async def test_analyze_async_matches_sync():
    request = AnalyzeRequest(title="Заголовок", text=LONG)
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel = await analyze_async(request, _registry(), executor)
    assert parallel == analyze(request, _registry())
