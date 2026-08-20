import asyncio
from concurrent.futures import Executor
from urllib.parse import urljoin

import httpx
from pydantic import HttpUrl

from court.config import Settings
from court.ingest.content import check_response, extract, read_limited
from court.ingest.errors import IngestError
from court.ingest.schemas import Article, ReviewRequest
from court.ingest.url_security import (
    Resolver,
    resolve_host,
    validate_public_url,
)

_REDIRECTS = {301, 302, 303, 307, 308}
_SUCCESS_MIN = 200
_SUCCESS_MAX = 300
_MAX_ARTICLE_CHARS = 100_000


async def resolve(
    request: ReviewRequest,
    client: httpx.AsyncClient,
    settings: Settings,
    resolver: Resolver | None = None,
    executor: Executor | None = None,
) -> Article:
    if request.url is None:
        return Article(title=(request.title or "").strip(), text=(request.text or "").strip())
    return await fetch_article(str(request.url), client, settings, resolver, executor)


async def fetch_article(
    url: str,
    client: httpx.AsyncClient,
    settings: Settings,
    resolver: Resolver | None = None,
    executor: Executor | None = None,
) -> Article:
    lookup = resolver or resolve_host
    current = url
    for redirect_count in range(settings.fetch_max_redirects + 1):
        await validate_public_url(current, lookup)
        async with client.stream("GET", current, follow_redirects=False) as response:
            if response.status_code in _REDIRECTS:
                current = _redirect(response, current, redirect_count, settings.fetch_max_redirects)
                continue
            if not _SUCCESS_MIN <= response.status_code < _SUCCESS_MAX:
                raise IngestError(
                    "fetch_failed",
                    f"source returned HTTP {response.status_code}",
                    502,
                )
            check_response(response, settings.fetch_max_bytes)
            raw = await read_limited(response, settings.fetch_max_bytes)
            content_type = response.headers.get("content-type", "")
            loop = asyncio.get_running_loop()
            title, text = await loop.run_in_executor(executor, extract, raw, content_type)
            if not title and not text:
                raise IngestError("extraction_failed", "source contained no article text", 422)
            return Article(
                title=title[:1000],
                text=text[:_MAX_ARTICLE_CHARS],
                source_url=HttpUrl(current),
            )
    raise IngestError("too_many_redirects", "too many redirects", 502)


def _redirect(
    response: httpx.Response,
    current: str,
    redirect_count: int,
    max_redirects: int,
) -> str:
    location = response.headers.get("location")
    if not location:
        raise IngestError("bad_redirect", "redirect has no location", 502)
    if redirect_count >= max_redirects:
        raise IngestError("too_many_redirects", "too many redirects", 502)
    return urljoin(current, location)
