import logging
from typing import Any, Protocol

from court.config import Settings
from court.tribunal.agents import AgentBundle, SearchContext, build_agents
from court.tribunal.errors import TribunalError
from court.tribunal.provenance import search_records, validate_search_provenance
from court.tribunal.schemas import Argument, Role, Verdict
from court.tribunal.telemetry import CallTelemetry, Usage

logger = logging.getLogger("court.tribunal")


class LLMClient(Protocol):
    async def argue(self, role: Role, user: str) -> Argument: ...

    async def judge(self, user: str) -> Verdict: ...

    async def aclose(self) -> None: ...

    def enable_telemetry(self) -> None: ...

    def drain_telemetry(self) -> list[CallTelemetry]: ...


def build_llm(
    settings: Settings,
    *,
    search_context: SearchContext | None = "low",
    include_forensics: bool = True,
    adversarial: bool = True,
) -> LLMClient | None:
    if not settings.openai_api_key.get_secret_value():
        return None
    return OpenAIAgentsClient(
        settings,
        search_context=search_context,
        include_forensics=include_forensics,
        adversarial=adversarial,
    )


class OpenAIAgentsClient:
    def __init__(
        self,
        settings: Settings,
        *,
        search_context: SearchContext | None = "low",
        include_forensics: bool = True,
        adversarial: bool = True,
    ) -> None:
        from agents import OpenAIResponsesModel  # noqa: PLC0415
        from openai import AsyncOpenAI  # noqa: PLC0415

        self._client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        model = OpenAIResponsesModel(model=settings.model, openai_client=self._client)
        self._agents: AgentBundle = build_agents(
            model,
            search_context=search_context,
            include_forensics=include_forensics,
            adversarial=adversarial,
        )
        self._max_turns = settings.llm_max_turns
        self._telemetry: list[CallTelemetry] | None = None

    def enable_telemetry(self) -> None:
        self._telemetry = []

    def drain_telemetry(self) -> list[CallTelemetry]:
        drained = self._telemetry or []
        if self._telemetry is not None:
            self._telemetry = []
        return drained

    async def argue(self, role: Role, user: str) -> Argument:
        agent = getattr(self._agents, role)
        result = await self._run(agent, user)
        self._record(role, result)
        output = result.final_output
        if not isinstance(output, Argument):
            raise TribunalError(502, "tribunal_invalid_output", "tribunal returned invalid output")
        output.role = role
        validate_search_provenance(output, result)
        return output

    async def judge(self, user: str) -> Verdict:
        result = await self._run(self._agents.judge, user)
        self._record("judge", result)
        output = result.final_output
        if not isinstance(output, Verdict):
            raise TribunalError(502, "tribunal_invalid_output", "tribunal returned invalid output")
        return output

    def _record(self, kind: str, result: Any) -> None:  # noqa: ANN401
        if self._telemetry is None:
            return
        usage = result.context_wrapper.usage
        self._telemetry.append(
            CallTelemetry(
                kind=kind,
                usage=Usage(
                    requests=usage.requests,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                ),
                searches=tuple(search_records(result)),
            )
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def _run(self, agent: Any, user: str) -> Any:  # noqa: ANN401
        from agents import ModelBehaviorError, RunConfig, Runner  # noqa: PLC0415
        from agents.exceptions import MaxTurnsExceeded  # noqa: PLC0415
        from openai import APITimeoutError, OpenAIError  # noqa: PLC0415

        try:
            result = await Runner.run(
                agent,
                user,
                max_turns=self._max_turns,
                run_config=RunConfig(tracing_disabled=True),
            )
        except APITimeoutError as exc:
            raise TribunalError(504, "tribunal_timeout", "tribunal request timed out") from exc
        except (MaxTurnsExceeded, ModelBehaviorError, OpenAIError) as exc:
            raise TribunalError(502, "tribunal_failed", "tribunal request failed") from exc
        usage = result.context_wrapper.usage
        logger.info(
            "tribunal usage requests=%d input_tokens=%d output_tokens=%d total_tokens=%d",
            usage.requests,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )
        return result
