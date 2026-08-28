import pytest
from pydantic import HttpUrl, ValidationError

from court.tribunal.schemas import Argument, Claim, Source


def _source(claim_id: str, url: str) -> Source:
    return Source(
        claim_id=claim_id,
        url=HttpUrl(url),
        title="Джерело",
        excerpt="Перевірений фрагмент.",
        supports="підтверджує тезу",
    )


def _argument(sources: list[Source]) -> Argument:
    return Argument(
        role="prosecutor",
        thesis="теза",
        claims=[Claim(id="c1", text="твердження")],
        sources=sources,
    )


def test_claim_may_cite_two_independent_sources():
    argument = _argument(
        [
            _source("c1", "https://a.example/story"),
            _source("c1", "https://b.example/story"),
        ]
    )
    assert len(argument.sources) == 2


def test_exact_duplicate_source_is_rejected():
    with pytest.raises(ValidationError):
        _argument(
            [
                _source("c1", "https://a.example/story"),
                _source("c1", "https://a.example/story"),
            ]
        )
