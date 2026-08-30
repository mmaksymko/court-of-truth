from unittest.mock import AsyncMock

import agents
import pytest

from court.tribunal.errors import TribunalError
from tests.tribunal.llm_support import Result, argument, client


@pytest.mark.asyncio
async def test_ungrounded_source_is_dropped_and_unexpected_output_is_rejected(monkeypatch):
    results = iter([Result(argument()), Result("wrong")])

    async def fake_run(starting_agent, user, *, max_turns, run_config):
        return next(results)

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    llm = client()
    # An ungrounded source is silently discarded (no search metadata backs it),
    # leaving the party with an empty evidence set instead of failing the review.
    output = await llm.argue("prosecutor", "user")
    assert output.sources == []
    # A structurally wrong output is still a hard failure.
    with pytest.raises(TribunalError, match="invalid output"):
        await llm.judge("user")


@pytest.mark.asyncio
async def test_model_behavior_error_is_stable_and_client_closes(monkeypatch):
    async def fake_run(starting_agent, user, *, max_turns, run_config):
        raise agents.ModelBehaviorError("sensitive provider detail")

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    llm = client()
    llm._client.close = AsyncMock()
    with pytest.raises(TribunalError) as caught:
        await llm.judge("user")
    assert caught.value.code == "tribunal_failed"
    assert "sensitive" not in caught.value.message
    await llm.aclose()
    llm._client.close.assert_awaited_once()
