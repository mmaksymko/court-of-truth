import asyncio

import pytest

from court.tribunal.deliberation import deliberate
from court.tribunal.errors import TribunalError
from court.tribunal.llm import OpenAIAgentsClient, build_llm
from court.tribunal.schemas import Deliberation
from tests.fakes import FakeLLM, make_settings
from tests.tribunal.support import report


@pytest.mark.asyncio
async def test_deliberate_runs_arguments_concurrently():
    llm = FakeLLM(delay=0.01)
    result = await deliberate("Назва статті", "текст статті", report(), llm)
    assert isinstance(result, Deliberation)
    assert result.prosecutor.role == "prosecutor"
    assert result.advocate.role == "advocate"
    assert result.prosecutor.sources[0].excerpt
    assert result.verdict.source_assessment
    assert llm.max_active_arguments == 2


@pytest.mark.asyncio
async def test_swap_order_reverses_party_presentation_to_judge():
    from court.tribunal.deliberation import deliberate_mode

    class SpyJudge(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.judge_prompt = ""

        async def judge(self, user: str):
            self.judge_prompt = user
            return await super().judge(user)

    default = SpyJudge()
    await deliberate_mode("Назва", "текст", report(), default)
    swapped = SpyJudge()
    await deliberate_mode("Назва", "текст", report(), swapped, swap_order=True)

    assert default.judge_prompt.index('role="prosecution"') < default.judge_prompt.index(
        'role="defence"'
    )
    assert swapped.judge_prompt.index('role="defence"') < swapped.judge_prompt.index(
        'role="prosecution"'
    )


@pytest.mark.asyncio
async def test_deliberate_cancels_sibling_when_one_argument_fails():
    class FailingLLM(FakeLLM):
        cancelled = False

        async def argue(self, role, user):
            if role == "prosecutor":
                await asyncio.sleep(0)
                raise TribunalError(502, "tribunal_failed", "failed")
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    llm = FailingLLM()
    with pytest.raises(TribunalError):
        await deliberate("Назва", "Текст", report(), llm)
    assert llm.cancelled


def test_build_llm_none_without_key():
    assert build_llm(make_settings(openai_api_key="")) is None


def test_build_llm_constructs_with_key():
    client = build_llm(make_settings(openai_api_key="sk-test-not-real"))
    assert isinstance(client, OpenAIAgentsClient)


def test_unknown_detector_reference_is_rejected():
    class BadLLM(FakeLLM):
        async def argue(self, role, user):
            argument = await super().argue(role, user)
            argument.cited_detectors = ["unknown"]
            return argument

    with pytest.raises(TribunalError, match="absent"):
        asyncio.run(deliberate("Назва", "Текст", report(), BadLLM()))


def test_source_referencing_unknown_claim_is_rejected():
    class DanglingSourceLLM(FakeLLM):
        async def argue(self, role, user):
            argument = await super().argue(role, user)
            argument.sources[0].claim_id = "не-існує"
            return argument

    with pytest.raises(TribunalError, match="claim absent from the argument"):
        asyncio.run(deliberate("Назва", "Текст", report(), DanglingSourceLLM()))
