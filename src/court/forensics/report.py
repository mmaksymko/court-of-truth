import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor

from court.forensics.predict import DEFAULT_MARGIN, run_detector
from court.forensics.registry import LoadedDetector
from court.forensics.schemas import (
    AnalyzeRequest,
    ForensicReport,
    OkResult,
    RiskSummary,
    SkippedResult,
)


def analyze(
    request: AnalyzeRequest,
    registry: Mapping[str, LoadedDetector],
    margin: float = DEFAULT_MARGIN,
) -> ForensicReport:
    results = [run_detector(detector, request, margin) for _, detector in sorted(registry.items())]
    return _report(request, results)


async def analyze_async(
    request: AnalyzeRequest,
    registry: Mapping[str, LoadedDetector],
    executor: Executor,
    margin: float = DEFAULT_MARGIN,
) -> ForensicReport:
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(executor, run_detector, detector, request, margin)
        for _, detector in sorted(registry.items())
    ]
    results = await asyncio.gather(*tasks)
    return _report(request, results)


def _report(request: AnalyzeRequest, results: Sequence[OkResult | SkippedResult]) -> ForensicReport:
    return ForensicReport(
        title_present=bool(request.title.strip()),
        text_chars=len(request.text),
        results=list(results),
        risk=risk_summary(results),
    )


def risk_summary(results: Sequence[OkResult | SkippedResult]) -> RiskSummary:
    flagged = [r.id for r in results if isinstance(r, OkResult) and r.flagged]
    low_confidence = [r.id for r in results if isinstance(r, OkResult) and r.low_confidence]
    return RiskSummary(
        flagged_count=len(flagged),
        flagged_ids=flagged,
        low_confidence_ids=low_confidence,
    )
