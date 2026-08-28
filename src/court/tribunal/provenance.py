import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit, urlunsplit

from court.tribunal.errors import TribunalError
from court.tribunal.schemas import Argument
from court.tribunal.telemetry import SearchRecord, SearchSource

if TYPE_CHECKING:
    from agents.items import ModelResponse

logger = logging.getLogger("court.tribunal")


class SearchResult(Protocol):
    raw_responses: list["ModelResponse"]


def validate_search_provenance(
    argument: Argument,
    result: SearchResult,
    extra_allowed_urls: set[str] | None = None,
) -> None:
    allowed = {_canonical(url) for url in _search_urls(result)}
    if extra_allowed_urls:
        allowed |= {_canonical(url) for url in extra_allowed_urls}
    submitted = {_canonical(str(source.url)) for source in argument.sources}
    if not submitted.issubset(allowed):
        logger.warning(
            "tribunal evidence not grounded in search: ungrounded=%s allowed=%s",
            sorted(submitted - allowed),
            sorted(allowed),
        )
        raise TribunalError(
            502,
            "tribunal_unverified_evidence",
            "tribunal returned evidence not grounded in web search metadata",
        )


def search_records(result: SearchResult) -> list[SearchRecord]:
    """Capture the executed web searches (query and returned sources) from one run.

    Reuses the same response walk as provenance validation; the output feeds the
    experiment telemetry and the frozen search transcript.
    """
    records: list[SearchRecord] = []
    for response in getattr(result, "raw_responses", []):
        for item in getattr(response, "output", []):
            if _get(item, "type") != "web_search_call":
                continue
            action = _get(item, "action")
            query = _get(action, "query", "")
            raw_sources = _get(action, "sources", [])
            sources: list[SearchSource] = []
            if isinstance(raw_sources, list):
                for source in raw_sources:
                    url = _get(source, "url")
                    if url:
                        title = _get(source, "title", "") or ""
                        sources.append(SearchSource(url=str(url), title=str(title)))
            records.append(SearchRecord(query=str(query or ""), sources=tuple(sources)))
    return records


def _search_urls(result: SearchResult) -> set[str]:
    urls: set[str] = set()
    for response in getattr(result, "raw_responses", []):
        for item in getattr(response, "output", []):
            if _get(item, "type") != "web_search_call":
                continue
            action = _get(item, "action")
            sources = _get(action, "sources", [])
            if not isinstance(sources, list):
                continue
            for source in sources:
                url = _get(source, "url")
                if url:
                    urls.add(str(url))
    return urls


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _canonical(value: str) -> str:
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    port = f":{parts.port}" if parts.port and parts.port not in {80, 443} else ""
    netloc = f"{host}{port}"
    path = parts.path.rstrip("/") or "/"
    return str(urlunsplit((parts.scheme.lower(), netloc, path, parts.query, "")))
