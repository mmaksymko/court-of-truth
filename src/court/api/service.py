from collections.abc import Mapping
from concurrent.futures import Executor

import httpx

from court.api.schemas import AnalyzeInput, AnalyzeResult
from court.config import Settings
from court.forensics.registry import LoadedDetector
from court.forensics.report import analyze_async
from court.forensics.schemas import AnalyzeRequest, ForensicReport
from court.ingest.fetch import fetch_article
from court.tribunal.deliberation import deliberate
from court.tribunal.llm import LLMClient
from court.tribunal.schemas import ReviewResponse


async def analyze_input(
    body: AnalyzeInput,
    *,
    http: httpx.AsyncClient,
    settings: Settings,
    registry: Mapping[str, LoadedDetector],
    executor: Executor,
) -> AnalyzeResult:
    if body.url is not None:
        article = await fetch_article(str(body.url), http, settings, executor=executor)
        title, text = article.title, article.text
    else:
        title, text = body.title, body.text
    report = await analyze_async(
        AnalyzeRequest(title=title, text=text),
        registry,
        executor,
        settings.low_confidence_margin,
    )
    return AnalyzeResult(**report.model_dump(), source_title=title, source_text=text)


async def review_input(  # noqa: PLR0913
    body: AnalyzeInput,
    *,
    http: httpx.AsyncClient,
    settings: Settings,
    registry: Mapping[str, LoadedDetector],
    executor: Executor,
    llm: LLMClient,
) -> ReviewResponse:
    analyzed = await analyze_input(
        body,
        http=http,
        settings=settings,
        registry=registry,
        executor=executor,
    )
    report = ForensicReport.model_validate(
        analyzed.model_dump(exclude={"source_title", "source_text"})
    )
    deliberation = await deliberate(
        analyzed.source_title,
        analyzed.source_text,
        report,
        llm,
        source_url=str(body.url) if body.url is not None else None,
    )
    return ReviewResponse(report=report, deliberation=deliberation)
