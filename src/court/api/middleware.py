import re
import uuid
from collections import deque
from collections.abc import Mapping
from contextvars import ContextVar

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from court.api.schemas import ErrorResponse

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_BODY_METHODS = {"POST", "PUT", "PATCH"}


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return

        buffered: deque[Message] = deque()
        received = 0
        while True:
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    response = error_response(413, "request_too_large", "body too large")
                    await response(scope, receive, send)
                    return
                buffered.append(message)
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                buffered.append(message)
                break

        async def replay_receive() -> Message:
            return buffered.popleft() if buffered else await receive()

        await self.app(scope, replay_receive, send)


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = _header(scope, b"x-request-id")
        request_id = (
            incoming if incoming and _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex[:12]
        )
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["x-request-id"] = request_id
                headers["x-content-type-options"] = "nosniff"
                headers["cache-control"] = "no-store"
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            request_id_var.reset(token)


def error_response(
    status: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    request_id = request_id or request_id_var.get() or None
    body = ErrorResponse(code=code, message=message, request_id=request_id)
    response_headers = dict(headers or {})
    if request_id:
        response_headers["x-request-id"] = request_id
    return JSONResponse(status_code=status, content=body.model_dump(), headers=response_headers)


def _header(scope: Scope, name: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return ""
