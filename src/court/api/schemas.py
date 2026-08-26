from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from court.forensics.schemas import MAX_TEXT_CHARS, ForensicReport


class AnalyzeResult(ForensicReport):
    """ForensicReport плюс проаналізований вхід (заголовок і тіло після завантаження)."""

    source_title: str = ""
    source_text: str = ""


class AnalyzeInput(BaseModel):
    """Вхід /analyze: або URL для завантаження статті, або готові title/text."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl | None = None
    title: str = Field(default="", max_length=1000)
    text: str = Field(default="", max_length=MAX_TEXT_CHARS)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.url is not None:
            if self.title.strip() or self.text.strip():
                raise ValueError("url cannot be combined with title or text")
            return self
        if not self.title.strip() and not self.text.strip():
            raise ValueError("provide either url or non-empty title/text")
        return self


class ComponentState(BaseModel):
    ready: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, ComponentState]
    detectors: dict[str, str]


class LiveResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None = None
