from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator

from court.forensics.schemas import ForensicReport

Stance = Literal["prosecutor", "advocate"]
Role = Literal["prosecutor", "advocate", "neutral"]
Label = Literal["reliable", "questionable", "unreliable"]


class Source(BaseModel):
    claim_id: str = Field(min_length=1, max_length=80)
    url: HttpUrl
    title: str = Field(default="", max_length=500)
    excerpt: str = Field(
        min_length=1,
        max_length=800,
        description="Unverified model summary; the URL alone is grounded in search metadata.",
    )
    supports: str = Field(min_length=1, max_length=500)
    verification: Literal["search_url_only"] = "search_url_only"


class Claim(BaseModel):
    id: str = Field(min_length=1, max_length=8)
    text: str = Field(min_length=1, max_length=800)


class Argument(BaseModel):
    role: Role
    thesis: str = Field(min_length=1, max_length=1200)
    claims: list[Claim] = Field(min_length=1, max_length=8)
    cited_detectors: list[str] = Field(default_factory=list, max_length=5)
    sources: list[Source] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _unique_references(self) -> "Argument":
        # A claim may cite several independent sources, so uniqueness is enforced on the
        # (claim_id, url) pair: distinct URLs for one claim are allowed, exact repeats are not.
        source_keys = [(source.claim_id, str(source.url)) for source in self.sources]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("duplicate (claim_id, url) sources are not allowed")
        claim_ids = [claim.id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim id values must be unique")
        if len(self.cited_detectors) != len(set(self.cited_detectors)):
            raise ValueError("cited_detectors values must be unique")
        return self


class EvidenceRecord(BaseModel):
    """Program-verified evidence assembled from the parties' sources, not authored by the Judge."""

    id: str = Field(min_length=1, max_length=8)
    side: Role
    claim_id: str = Field(min_length=1, max_length=80)
    url: HttpUrl
    title: str = Field(default="", max_length=500)
    excerpt: str = Field(min_length=1, max_length=800)
    supports: str = Field(min_length=1, max_length=500)
    verification: Literal["search_url_only"] = "search_url_only"


class Objection(BaseModel):
    claim_id: str = Field(min_length=1, max_length=8)
    reason: str = Field(min_length=1, max_length=500)


class VerdictProbabilities(BaseModel):
    reliable: float = Field(ge=0, le=1)
    questionable: float = Field(ge=0, le=1)
    unreliable: float = Field(ge=0, le=1)

    def as_map(self) -> dict[Label, float]:
        return {
            "reliable": self.reliable,
            "questionable": self.questionable,
            "unreliable": self.unreliable,
        }


_PROBABILITY_TOLERANCE = 0.02


class Verdict(BaseModel):
    label: Label
    probabilities: VerdictProbabilities
    confidence: float = Field(default=0.0, ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=2000)
    key_signals: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=8
    )
    source_assessment: str = Field(min_length=1, max_length=1000)
    used_evidence_ids: list[str] = Field(default_factory=list, max_length=16)
    objections: list[Objection] = Field(default_factory=list, max_length=8)
    cited_detectors: list[str] = Field(default_factory=list, max_length=5)
    schema_version: int = 1

    @model_validator(mode="after")
    def _derive_and_validate(self) -> "Verdict":
        probabilities = self.probabilities.as_map()
        if abs(sum(probabilities.values()) - 1.0) > _PROBABILITY_TOLERANCE:
            raise ValueError("verdict probabilities must sum to 1")
        if self.label != max(probabilities, key=lambda label: probabilities[label]):
            raise ValueError("verdict label must equal the most probable class")
        self.confidence = probabilities[self.label]
        if len(self.used_evidence_ids) != len(set(self.used_evidence_ids)):
            raise ValueError("used_evidence_ids values must be unique")
        if len(self.cited_detectors) != len(set(self.cited_detectors)):
            raise ValueError("cited_detectors values must be unique")
        return self


class Deliberation(BaseModel):
    prosecutor: Argument
    advocate: Argument
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    verdict: Verdict


class ReviewResponse(BaseModel):
    report: ForensicReport
    deliberation: Deliberation
