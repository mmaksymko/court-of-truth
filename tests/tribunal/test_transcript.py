from pathlib import Path

import pytest

from court.tribunal.errors import TribunalError
from court.tribunal.provenance import validate_search_provenance
from court.tribunal.telemetry import SearchRecord, SearchSource
from court.tribunal.transcript import RecordedSearchProvider, SearchTranscript
from tests.tribunal.llm_support import Result, argument


def _records() -> list[SearchRecord]:
    return [
        SearchRecord(
            query="перевірка факту",
            sources=(SearchSource(url="https://a.example/story", title="A"),),
        ),
        SearchRecord(
            query="перевірка факту",
            sources=(SearchSource(url="https://b.example/story", title="B"),),
        ),
    ]


def test_transcript_merges_sources_for_repeated_query():
    transcript = SearchTranscript.from_records(_records())
    sources = transcript.lookup("перевірка факту")
    assert {source.url for source in sources} == {
        "https://a.example/story",
        "https://b.example/story",
    }
    assert transcript.all_urls() == {"https://a.example/story", "https://b.example/story"}


def test_transcript_sha256_is_stable_and_roundtrips(tmp_path: Path):
    transcript = SearchTranscript.from_records(_records())
    digest = transcript.sha256()
    path = tmp_path / "trace.jsonl"
    transcript.save(path)
    reloaded = SearchTranscript.load(path)
    assert reloaded.sha256() == digest
    assert reloaded.lookup("перевірка факту")


def test_transcript_from_captured_run_records():
    captured = [
        [
            {
                "kind": "prosecutor",
                "query": "запит",
                "sources": [{"url": "https://a.example/x", "title": "A"}],
            }
        ]
    ]
    transcript = SearchTranscript.from_captured(captured)
    assert transcript.lookup("запит")[0].url == "https://a.example/x"


def test_recorded_provider_hits_and_tracks_misses():
    provider = RecordedSearchProvider(SearchTranscript.from_records(_records()))
    assert provider.search("перевірка факту")
    assert provider.search("невідомий запит") == ()
    assert provider.misses == ["невідомий запит"]


def test_provenance_accepts_urls_from_replay_allowed_set():
    arg = argument("https://replayed.example/only")
    result = Result(arg, source_urls=())
    with pytest.raises(TribunalError):
        validate_search_provenance(arg, result)
    validate_search_provenance(arg, result, extra_allowed_urls={"https://replayed.example/only"})
