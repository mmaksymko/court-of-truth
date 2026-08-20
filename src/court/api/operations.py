from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, HTTPException, Request

from court.api.dependencies import RegistryDep, SettingsDep, rate_subject
from court.api.limits import LimitExceededError, OperationLimits
from court.forensics.report import analyze_async
from court.forensics.schemas import AnalyzeRequest, ForensicReport

if TYPE_CHECKING:
    from concurrent.futures import Executor

router = APIRouter()


@router.post("/analyze", response_model=ForensicReport)
async def analyze_route(
    body: AnalyzeRequest,
    request: Request,
    settings: SettingsDep,
    registry: RegistryDep,
) -> ForensicReport:
    subject = rate_subject(request)
    limits = cast("OperationLimits", request.app.state.operation_limits)
    executor = cast("Executor", request.app.state.detector_executor)
    try:
        await limits.check_rate(subject)
        async with limits.slot():
            return await analyze_async(body, registry, executor, settings.low_confidence_margin)
    except LimitExceededError as exc:
        raise HTTPException(429, str(exc)) from exc
