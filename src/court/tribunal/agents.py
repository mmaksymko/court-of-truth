from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from court.tribunal.instructions import judge_instructions, party_instructions
from court.tribunal.schemas import Argument, Role, Verdict

if TYPE_CHECKING:
    from agents import Model

SearchContext = Literal["low", "medium", "high"]


class AgentBundle(NamedTuple):
    prosecutor: Any
    advocate: Any
    neutral: Any
    judge: Any


def build_agents(
    model: "Model",
    *,
    search_context: "SearchContext | None" = "low",
    include_forensics: bool = True,
    adversarial: bool = True,
) -> AgentBundle:
    from agents import Agent, ModelSettings, WebSearchTool  # noqa: PLC0415
    from openai.types.shared import Reasoning  # noqa: PLC0415

    search = search_context is not None
    if search_context is not None:
        research_settings = ModelSettings(
            store=False,
            tool_choice="required",
            reasoning=Reasoning(effort="medium"),
            response_include=["web_search_call.action.sources"],
        )
        tools: list[Any] = [WebSearchTool(search_context_size=search_context)]
    else:
        # B3: no external search - the sides argue from the article and detectors only.
        research_settings = ModelSettings(store=False, reasoning=Reasoning(effort="medium"))
        tools = []

    def researcher(name: str, instructions: str) -> Any:  # noqa: ANN401
        return Agent(
            name=name,
            instructions=instructions,
            model=model,
            model_settings=research_settings,
            output_type=Argument,
            tools=tools,
        )

    def party(name: str, role: Role) -> Any:  # noqa: ANN401
        return researcher(
            name,
            party_instructions(role, include_forensics=include_forensics, include_search=search),
        )

    return AgentBundle(
        prosecutor=party("Прокурор", "prosecutor"),
        advocate=party("Адвокат", "advocate"),
        neutral=party("Дослідник", "neutral"),
        judge=Agent(
            name="Суддя",
            instructions=judge_instructions(
                adversarial=adversarial, include_forensics=include_forensics
            ),
            model=model,
            model_settings=ModelSettings(store=False, reasoning=Reasoning(effort="medium")),
            output_type=Verdict,
        ),
    )
