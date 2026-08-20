from fastapi import APIRouter

from court.api.health import router as health_router
from court.api.operations import router as operations_router

router = APIRouter()
router.include_router(health_router)
router.include_router(operations_router)
