import asyncio

import pytest
from pydantic import ValidationError

from court.tribunal.deliberation import deliberate
from court.tribunal.evidence import build_evidence
from court.tribunal.schemas import Objection, Verdict, VerdictProbabilities
from tests.fakes import FakeLLM
from tests.tribunal.support import report


def _verdict(**overrides) -> dict:
    base = {
        "label": "questionable",
        "probabilities": VerdictProbabilities(reliable=0.2, questionable=0.6, unreliable=0.2),
        "rationale": "ok",
        "source_assessment": "нейтрально",
    }
    base.update(overrides)
    return base


def test_confidence_is_derived_from_the_probability_of_the_label():
    verdict = Verdict(**_verdict())
    assert verdict.confidence == 0.6
    assert verdict.schema_version == 1


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1"):
        Verdict(
            **_verdict(
                probabilities=VerdictProbabilities(reliable=0.2, questionable=0.2, unreliable=0.2)
            )
        )


def test_label_must_equal_the_most_probable_class():
    with pytest.raises(ValidationError, match="most probable"):
        Verdict(
            **_verdict(
                label="reliable",
                probabilities=VerdictProbabilities(reliable=0.2, questionable=0.6, unreliable=0.2),
            )
        )


def test_duplicate_reference_ids_are_rejected():
    with pytest.raises(ValidationError, match="used_evidence_ids"):
        Verdict(**_verdict(used_evidence_ids=["p1", "p1"]))
    with pytest.raises(ValidationError, match="cited_detectors"):
        Verdict(**_verdict(cited_detectors=["jeansa", "jeansa"]))


def test_build_evidence_assigns_stable_ids_and_sides():
    async def arguments():
        llm = FakeLLM()
        return await llm.argue("prosecutor", "u"), await llm.argue("advocate", "u")

    prosecution, defence = asyncio.run(arguments())
    evidence = build_evidence([prosecution, defence])
    # Evidence ids live in their own namespace ("ep"/"ea"), distinct from claim ids.
    assert [record.id for record in evidence] == ["ep1", "ea1"]
    assert [record.side for record in evidence] == ["prosecutor", "advocate"]


def test_verdict_citing_unknown_evidence_is_trimmed():
    class GhostEvidenceJudge(FakeLLM):
        async def judge(self, _user: str) -> Verdict:
            return Verdict(**_verdict(used_evidence_ids=["ghost"]))

    result = asyncio.run(deliberate("Назва", "Текст", report(), GhostEvidenceJudge()))
    assert result.verdict.used_evidence_ids == []


def test_verdict_citing_unknown_detector_is_trimmed():
    class GhostDetectorJudge(FakeLLM):
        async def judge(self, _user: str) -> Verdict:
            return Verdict(**_verdict(cited_detectors=["nope"]))

    result = asyncio.run(deliberate("Назва", "Текст", report(), GhostDetectorJudge()))
    assert result.verdict.cited_detectors == []


def test_verdict_objecting_to_unknown_claim_is_trimmed():
    class GhostObjectionJudge(FakeLLM):
        async def judge(self, _user: str) -> Verdict:
            return Verdict(**_verdict(objections=[Objection(claim_id="zzz", reason="без підстав")]))

    result = asyncio.run(deliberate("Назва", "Текст", report(), GhostObjectionJudge()))
    assert result.verdict.objections == []


def test_verdict_objecting_to_a_real_claim_passes():
    class RealObjectionJudge(FakeLLM):
        async def judge(self, _user: str) -> Verdict:
            return Verdict(
                **_verdict(objections=[Objection(claim_id="pc1", reason="слабко підтверджено")])
            )

    result = asyncio.run(deliberate("Назва", "Текст", report(), RealObjectionJudge()))
    assert result.verdict.objections[0].claim_id == "pc1"
