"""Frozen web-search transcript and its offline replay provider (decision E-014).

A live capture run records, per query, the sources the hosted search returned.
Saving that transcript freezes the evidence pool so the same paired run can be
replayed deterministically and at no cost. The provider is pure and offline; the
only key-gated step is letting the live tribunal draw searches from it, which is
confirmed during the pilot.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from court.tribunal.telemetry import SearchRecord, SearchSource

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path


class SearchTranscript:
    """A query-keyed snapshot of executed web searches."""

    def __init__(self, records: Iterable[SearchRecord]) -> None:
        self._by_query: dict[str, tuple[SearchSource, ...]] = {}
        for record in records:
            merged = list(self._by_query.get(record.query, ()))
            seen = {source.url for source in merged}
            for source in record.sources:
                if source.url not in seen:
                    merged.append(source)
                    seen.add(source.url)
            self._by_query[record.query] = tuple(merged)

    def lookup(self, query: str) -> tuple[SearchSource, ...]:
        return self._by_query.get(query, ())

    def queries(self) -> list[str]:
        return list(self._by_query)

    def all_urls(self) -> set[str]:
        return {source.url for sources in self._by_query.values() for source in sources}

    def as_lines(self) -> list[str]:
        lines: list[str] = []
        for query in sorted(self._by_query):
            payload = {
                "query": query,
                "sources": [
                    {"url": source.url, "title": source.title} for source in self._by_query[query]
                ],
            }
            lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return lines

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.as_lines()) + "\n", encoding="utf-8")

    def sha256(self) -> str:
        return hashlib.sha256("\n".join(self.as_lines()).encode("utf-8")).hexdigest()

    @classmethod
    def from_records(cls, records: Iterable[SearchRecord]) -> SearchTranscript:
        return cls(records)

    @classmethod
    def from_captured(cls, captured: Iterable[Sequence[dict[str, object]]]) -> SearchTranscript:
        """Build a transcript from serialized ``RunRecord.searches`` entries."""
        records: list[SearchRecord] = []
        for run in captured:
            for entry in run:
                raw_sources = entry.get("sources", [])
                sources: list[SearchSource] = []
                if isinstance(raw_sources, list):
                    for source in raw_sources:
                        if isinstance(source, dict) and source.get("url"):
                            sources.append(
                                SearchSource(
                                    url=str(source["url"]),
                                    title=str(source.get("title", "")),
                                )
                            )
                records.append(
                    SearchRecord(query=str(entry.get("query", "")), sources=tuple(sources))
                )
        return cls(records)

    @classmethod
    def load(cls, path: Path) -> SearchTranscript:
        records: list[SearchRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            raw_sources = data.get("sources", [])
            sources = tuple(
                SearchSource(url=str(item["url"]), title=str(item.get("title", "")))
                for item in raw_sources
                if item.get("url")
            )
            records.append(SearchRecord(query=str(data["query"]), sources=sources))
        return cls(records)


class RecordedSearchProvider:
    """Serves recorded search results offline and records queries with no capture."""

    def __init__(self, transcript: SearchTranscript) -> None:
        self._transcript = transcript
        self.misses: list[str] = []

    def search(self, query: str) -> tuple[SearchSource, ...]:
        sources = self._transcript.lookup(query)
        if not sources:
            self.misses.append(query)
        return sources

    def served_urls(self) -> set[str]:
        return self._transcript.all_urls()
