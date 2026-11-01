import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol
from urllib.parse import unquote, urlsplit, urlunsplit

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
    """Drop any source whose URL is not grounded in the executed web search.

    A single ungrounded URL used to fail the whole review with a 502; that made a
    minor citation slip (URL normalization drift, a cleaned tracking parameter)
    poison an otherwise valid deliberation. Instead we silently discard the
    ungrounded sources and keep the grounded ones - the party simply argues with a
    smaller, fully verified evidence set. If every source is ungrounded the party
    is left with no external evidence, which the Judge scores as thin support
    (typically questionable), not as a service failure.
    """
    allowed = {_canonical(url) for url in _search_urls(result)}
    if extra_allowed_urls:
        allowed |= {_canonical(url) for url in extra_allowed_urls}
    grounded = [source for source in argument.sources if _canonical(str(source.url)) in allowed]
    dropped = [source for source in argument.sources if source not in grounded]
    if dropped:
        logger.warning(
            "tribunal dropping ungrounded sources: dropped=%s allowed=%s",
            sorted(str(source.url) for source in dropped),
            sorted(allowed),
        )
        argument.sources = grounded


def search_records(
    result: SearchResult,
    response_times: Mapping[str, str] | None = None,
    fallback_time: str | None = None,
) -> list[SearchRecord]:
    """Capture the executed web searches (query, sources, UTC time) from one run.

    Reuses the same response walk as provenance validation; the output feeds the
    experiment telemetry and the frozen search transcript. ``response_times`` maps
    a model response key (see ``response_key``) to the UTC moment it arrived; a
    search inside an unmatched response gets ``fallback_time``.
    """
    times = response_times or {}
    records: list[SearchRecord] = []
    for response in getattr(result, "raw_responses", []):
        executed_at = times.get(response_key(response), fallback_time)
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
            records.append(
                SearchRecord(
                    query=str(query or ""),
                    sources=tuple(sources),
                    executed_at=executed_at,
                )
            )
    return records


def response_key(response: object) -> str:
    """Stable key of one model response: its API id, else its object identity."""
    response_id = _get(response, "response_id")
    return str(response_id) if response_id else f"obj:{id(response)}"


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
    # Normalize aggressively so that cosmetic differences between the search-result
    # URL and the model-submitted URL do not falsely mark a real source ungrounded:
    # lowercase host, drop a leading "www.", percent-decode the path (Cyrillic slugs
    # arrive both encoded and decoded), strip the trailing slash and the query string
    # (models routinely clean tracking parameters).
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = f":{parts.port}" if parts.port and parts.port not in {80, 443} else ""
    netloc = f"{host}{port}"
    path = unquote(parts.path).rstrip("/") or "/"
    return str(urlunsplit((parts.scheme.lower(), netloc, path, "", "")))
