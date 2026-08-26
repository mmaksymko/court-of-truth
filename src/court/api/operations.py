from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, HTTPException, Request

from court.api.dependencies import RegistryDep, SettingsDep, rate_subject
from court.api.limits import LimitExceededError, OperationLimits
from court.api.schemas import AnalyzeInput, AnalyzeResult
from court.api.service import analyze_input

if TYPE_CHECKING:
    from concurrent.futures import Executor

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResult)
async def analyze_route(
    body: AnalyzeInput,
    request: Request,
    settings: SettingsDep,
    registry: RegistryDep,
) -> AnalyzeResult:
    subject = rate_subject(request)
    limits = cast("OperationLimits", request.app.state.operation_limits)
    executor = cast("Executor", request.app.state.detector_executor)
    try:
        await limits.check_rate(subject)
        async with limits.slot():
            return await analyze_input(
                body,
                http=request.app.state.http,
                settings=settings,
                registry=registry,
                executor=executor,
            )
    except LimitExceededError as exc:
        raise HTTPException(429, str(exc)) from exc
