from types import SimpleNamespace

import agents
import pytest

from court.tribunal.provenance import response_key, search_records
from court.tribunal.telemetry import (
    CallTelemetry,
    SearchRecord,
    SearchSource,
    Usage,
    searches_as_dicts,
)
from court.tribunal.transcript import SearchTranscript
from tests.tribunal.llm_support import URL, Result, argument, client


def _search_response(response_id: str | None, query: str) -> SimpleNamespace:
    item = {"type": "web_search_call", "action": {"query": query, "sources": [{"url": URL}]}}
    return SimpleNamespace(output=[item], response_id=response_id)


def test_search_records_take_time_of_their_response_or_fallback():
    matched = _search_response("resp_1", "перший")
    unmatched = _search_response(None, "другий")
    result = SimpleNamespace(raw_responses=[matched, unmatched])
    times = {response_key(matched): "2026-09-27T10:00:00.000Z"}

    records = search_records(result, times, fallback_time="2026-09-27T10:00:05.000Z")

    assert [record.executed_at for record in records] == [
        "2026-09-27T10:00:00.000Z",
        "2026-09-27T10:00:05.000Z",
    ]
    assert response_key(matched) == "resp_1"
    assert response_key(unmatched).startswith("obj:")


def test_search_dicts_add_time_only_when_known():
    call = CallTelemetry(
        kind="prosecutor",
        usage=Usage(),
        searches=(
            SearchRecord("q1", (SearchSource(URL),), executed_at="2026-09-27T10:00:00.000Z"),
            SearchRecord("q2", (SearchSource(URL),)),
        ),
    )
    first, second = searches_as_dicts([call])
    assert first["executed_at"] == "2026-09-27T10:00:00.000Z"
    assert "executed_at" not in second


def test_transcript_hash_ignores_search_time():
    timed = SearchRecord("q", (SearchSource(URL),), executed_at="2026-09-27T10:00:00.000Z")
    untimed = SearchRecord("q", (SearchSource(URL),))
    assert SearchTranscript([timed]).sha256() == SearchTranscript([untimed]).sha256()


@pytest.mark.asyncio
async def test_live_client_stamps_searches_via_run_hooks(monkeypatch):
    async def fake_run(starting_agent, user, *, max_turns, run_config, hooks):
        result = Result(argument(), (URL,))
        response = result.raw_responses[0]
        response.response_id = "resp_live"
        await hooks.on_llm_end(None, starting_agent, response)
        return result

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    tribunal = client()
    tribunal.enable_telemetry()
    await tribunal.argue("prosecutor", "user")
    (call,) = tribunal.drain_telemetry()
    executed_at = call.searches[0].executed_at
    assert executed_at is not None
    assert executed_at.endswith("Z")
    assert executed_at[:4].isdigit()
