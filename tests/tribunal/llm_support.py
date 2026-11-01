from types import SimpleNamespace

from agents.usage import Usage
from pydantic import HttpUrl

from court.tribunal.llm import OpenAIAgentsClient
from court.tribunal.schemas import Argument, Claim, Source
from tests.fakes import make_settings

URL = "https://example.org/evidence"


class Result:
    def __init__(self, output: object, source_urls: tuple[str, ...] = ()) -> None:
        self.final_output = output
        sources = [{"url": url} for url in source_urls]
        item = {"type": "web_search_call", "action": {"sources": sources}}
        self.raw_responses = [SimpleNamespace(output=[item])] if sources else []
        self.context_wrapper = SimpleNamespace(usage=Usage())


def argument(url: str = URL) -> Argument:
    return Argument(
        role="advocate",
        thesis="теза",
        claims=[Claim(id="c1", text="пункт")],
        sources=[
            Source(
                claim_id="claim-1",
                url=HttpUrl(url),
                excerpt="Неперевірене резюме.",
                supports="підтримує пункт",
            )
        ],
    )


def client() -> OpenAIAgentsClient:
    return OpenAIAgentsClient(make_settings(openai_api_key="sk-test-not-real"))
