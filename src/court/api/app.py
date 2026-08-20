import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from court.api.errors import register_error_handlers
from court.api.limits import OperationLimits
from court.api.middleware import BodyLimitMiddleware, RequestIdMiddleware
from court.api.routes import router
from court.config import Settings, get_settings
from court.forensics.models import EXPECTED_DETECTORS, ensure_registry
from court.forensics.registry import LoadedDetector

RegistryLoader = Callable[[Settings], Mapping[str, LoadedDetector]]


def _registry(settings: Settings) -> Mapping[str, LoadedDetector]:
    return ensure_registry(
        settings.artifacts_dir,
        EXPECTED_DETECTORS,
        settings.enabled_detectors,
    )


def create_app(
    settings: Settings | None = None,
    *,
    registry_loader: RegistryLoader = _registry,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    configured = settings or get_settings()
    logging.basicConfig(
        level=configured.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = configured
        app.state.registry = await asyncio.to_thread(registry_loader, configured)
        app.state.detector_executor = ThreadPoolExecutor(
            max_workers=configured.detector_workers,
            thread_name_prefix="court-detector",
        )
        app.state.operation_limits = OperationLimits(
            configured.operation_rate_per_minute,
            configured.operation_concurrency,
        )
        timeout = httpx.Timeout(configured.fetch_timeout_s)
        app.state.http = httpx.AsyncClient(
            timeout=timeout,
            transport=transport,
            headers={"accept-encoding": "identity", "user-agent": "court-of-truth/0.2"},
            trust_env=False,
        )
        try:
            yield
        finally:
            try:
                await app.state.http.aclose()
            finally:
                app.state.detector_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Court Criminalist", version="0.2.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_bytes=configured.max_body_bytes)
    app.add_middleware(RequestIdMiddleware)
    if configured.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=configured.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type", "x-request-id"],
            expose_headers=["x-request-id"],
        )
    app.include_router(router, prefix="/v1")
    register_error_handlers(app)
    return app


app = create_app()
