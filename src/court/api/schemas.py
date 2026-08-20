from typing import Literal

from pydantic import BaseModel


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
