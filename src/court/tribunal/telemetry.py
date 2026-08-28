"""Per-call telemetry for the tribunal: token usage and the web-search trace.

These records let the experiment runner persist cost, latency and the exact
evidence a paid run produced, and they are the basis of the frozen search
transcript used for offline replay. Telemetry is opt-in: the API `/review`
path never enables it, so a long-lived server accumulates nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class SearchSource:
    url: str
    title: str = ""


@dataclass(frozen=True)
class SearchRecord:
    query: str
    sources: tuple[SearchSource, ...] = ()


@dataclass(frozen=True)
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class CallTelemetry:
    kind: str
    usage: Usage
    searches: tuple[SearchRecord, ...] = ()


def aggregate_usage(calls: Iterable[CallTelemetry]) -> Usage:
    total = Usage()
    for call in calls:
        total = Usage(
            requests=total.requests + call.usage.requests,
            input_tokens=total.input_tokens + call.usage.input_tokens,
            output_tokens=total.output_tokens + call.usage.output_tokens,
            total_tokens=total.total_tokens + call.usage.total_tokens,
        )
    return total


def usage_as_dict(usage: Usage) -> dict[str, int]:
    return {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def searches_as_dicts(calls: Iterable[CallTelemetry]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for call in calls:
        for search in call.searches:
            records.append(
                {
                    "kind": call.kind,
                    "query": search.query,
                    "sources": [
                        {"url": source.url, "title": source.title} for source in search.sources
                    ],
                }
            )
    return records
