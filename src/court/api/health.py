from fastapi import APIRouter, HTTPException, Request

from court.api.dependencies import RegistryDep
from court.api.schemas import (
    ComponentState,
    HealthResponse,
    LiveResponse,
    ReadyResponse,
)
from court.forensics.schemas import DetectorMeta

router = APIRouter()


@router.get("/health/live", response_model=LiveResponse)
async def live() -> LiveResponse:
    return LiveResponse()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    registry = getattr(request.app.state, "registry", None)
    detector_ready = registry is not None
    llm_ready = getattr(request.app.state, "llm", None) is not None
    return HealthResponse(
        status="ok" if detector_ready else "degraded",
        components={
            "forensics": ComponentState(
                ready=detector_ready,
                detail="loaded" if detector_ready else "registry unavailable",
            ),
            "tribunal": ComponentState(
                ready=llm_ready,
                detail="configured" if llm_ready else "OpenAI API key not configured",
            ),
        },
        detectors={
            detector_id: detector.meta.version for detector_id, detector in (registry or {}).items()
        },
    )


@router.get("/health/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    if getattr(request.app.state, "registry", None) is None:
        raise HTTPException(503, "detector registry is not ready")
    return ReadyResponse()


@router.get("/detectors", response_model=list[DetectorMeta])
async def detectors(registry: RegistryDep) -> list[DetectorMeta]:
    return [detector.meta for detector in registry.values()]
