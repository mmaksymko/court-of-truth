from unittest.mock import AsyncMock

import agents
import pytest

from court.tribunal.errors import TribunalError
from tests.tribunal.llm_support import Result, argument, client


@pytest.mark.asyncio
async def test_unverified_source_and_unexpected_output_are_rejected(monkeypatch):
    results = iter([Result(argument()), Result("wrong")])

    async def fake_run(starting_agent, user, *, max_turns, run_config):
        return next(results)

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    llm = client()
    with pytest.raises(TribunalError, match="not grounded"):
        await llm.argue("prosecutor", "user")
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
