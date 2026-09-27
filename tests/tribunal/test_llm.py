import agents
import pytest

from court.tribunal.instructions import PROSECUTOR
from court.tribunal.schemas import Argument, Verdict, VerdictProbabilities
from tests.tribunal.llm_support import URL, Result, argument, client


@pytest.mark.asyncio
async def test_argument_uses_private_bounded_search_run(monkeypatch):
    captured = {}

    async def fake_run(starting_agent, user, *, max_turns, run_config, hooks=None):
        captured.update(agent=starting_agent, turns=max_turns, config=run_config)
        return Result(argument(), (URL,))

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    result = await client().argue("prosecutor", "user")
    agent = captured["agent"]
    assert result.role == "prosecutor"
    assert [tool.name for tool in agent.tools] == ["web_search"]
    assert agent.output_type is Argument
    assert agent.instructions == PROSECUTOR
    assert agent.model_settings.store is False
    assert agent.model_settings.tool_choice == "auto"
    assert "web_search_call.action.sources" in agent.model_settings.response_include
    assert agent.model_settings.reasoning is not None
    assert agent.model_settings.reasoning.effort == "medium"
    assert captured["turns"] == 8
    assert captured["config"].tracing_disabled


def test_b3_party_has_no_web_search_tool():
    from court.tribunal.llm import OpenAIAgentsClient
    from tests.fakes import make_settings

    client = OpenAIAgentsClient(
        make_settings(openai_api_key="sk-test-not-real"), search_context=None
    )
    prosecutor = client._agents.prosecutor
    assert prosecutor.tools == []
    assert prosecutor.model_settings.tool_choice != "required"
    assert "Без зовнішнього пошуку" in prosecutor.instructions


@pytest.mark.asyncio
async def test_telemetry_captures_usage_and_search_trace(monkeypatch):
    async def fake_run(starting_agent, user, *, max_turns, run_config, hooks=None):
        return Result(argument(), (URL,))

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    tribunal = client()
    assert tribunal.drain_telemetry() == []
    tribunal.enable_telemetry()
    await tribunal.argue("prosecutor", "user")
    drained = tribunal.drain_telemetry()
    assert len(drained) == 1
    assert drained[0].kind == "prosecutor"
    assert drained[0].searches[0].sources[0].url == URL
    # Draining twice yields nothing new.
    assert tribunal.drain_telemetry() == []


@pytest.mark.asyncio
async def test_judge_has_no_tools_and_private_output(monkeypatch):
    captured = {}
    verdict = Verdict(
        label="reliable",
        probabilities=VerdictProbabilities(reliable=0.7, questionable=0.2, unreliable=0.1),
        rationale="ok",
        source_assessment="обмежена доказова база",
    )

    async def fake_run(starting_agent, user, *, max_turns, run_config, hooks=None):
        captured["agent"] = starting_agent
        return Result(verdict)

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    result = await client().judge("user")
    assert result.label == "reliable"
    assert captured["agent"].tools == []
    assert captured["agent"].model_settings.store is False
