import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from court.api.middleware import error_response
from court.ingest.fetch import IngestError

logger = logging.getLogger("court.api")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return error_response(422, "invalid_request", "request validation failed")

    @app.exception_handler(IngestError)
    async def ingest_error(_request: Request, exc: IngestError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = str(exc.detail)
        code = {
            404: "not_found",
            405: "method_not_allowed",
            429: "rate_limited",
            503: "not_ready",
        }.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, detail, exc.headers)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled request error", exc_info=exc)
        return error_response(
            500,
            "internal_error",
            "internal server error",
            request_id=getattr(request.state, "request_id", None),
        )
