from collections.abc import Mapping
from concurrent.futures import Executor

import httpx

from court.api.schemas import AnalyzeInput, AnalyzeResult
from court.config import Settings
from court.forensics.registry import LoadedDetector
from court.forensics.report import analyze_async
from court.forensics.schemas import AnalyzeRequest
from court.ingest.fetch import fetch_article


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
