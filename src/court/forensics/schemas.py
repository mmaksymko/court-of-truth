from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Scope = Literal["title", "body"]
Backend = Literal["sklearn", "transformers"]

MAX_TEXT_CHARS = 100_000


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=1000)
    text: str = Field(default="", max_length=MAX_TEXT_CHARS)

    @model_validator(mode="after")
    def _not_empty(self) -> "AnalyzeRequest":
        if not self.title.strip() and not self.text.strip():
            raise ValueError("title or text is required")
        return self


class OkResult(BaseModel):
    status: Literal["ok"] = "ok"
    id: str
    scope: Scope
    label: str
    probability: float = Field(ge=0, le=1, description="P(positive class), regardless of label")
    flagged: bool
    low_confidence: bool
    caveats: list[str] = []
    evidence: list[str] = []


class SkippedResult(BaseModel):
    status: Literal["skipped"] = "skipped"
    id: str
    scope: Scope
    reason: str


DetectorResult = Annotated[OkResult | SkippedResult, Field(discriminator="status")]


class RiskSummary(BaseModel):
    kind: Literal["heuristic"] = "heuristic"
    flagged_count: int
    flagged_ids: list[str]
    low_confidence_ids: list[str]


class ForensicReport(BaseModel):
    title_present: bool
    text_chars: int
    results: list[DetectorResult]
    risk: RiskSummary


class DetectorMeta(BaseModel):
    id: str
    scope: Scope
    backend: Backend
    labels: tuple[str, str]
    positive_label: str
    threshold: float = Field(ge=0, le=1)
    version: str
    metrics: dict[str, float]
