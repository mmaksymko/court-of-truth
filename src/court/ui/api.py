from dataclasses import dataclass
from typing import Any

import httpx

JsonObject = dict[str, Any]
HTTP_ERROR_START = 400


@dataclass(frozen=True, slots=True)
class ApiError(Exception):
    message: str
    code: str = "network_error"
    status_code: int | None = None
    request_id: str | None = None

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class Readiness:
    forensics: bool
    tribunal: bool
    detail: str = ""


class CourtApi:
    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(30.0),
            transport=transport,
        )

    def health(self) -> Readiness:
        payload = self._request("GET", "/v1/health", timeout=10.0)
        components = payload.get("components", {})
        if not isinstance(components, dict):
            raise ApiError("API повернув некоректний стан компонентів.", code="invalid_response")
        forensics = components.get("forensics", {})
        tribunal = components.get("tribunal", {})
        return Readiness(
            forensics=bool(forensics.get("ready")) if isinstance(forensics, dict) else False,
            tribunal=bool(tribunal.get("ready")) if isinstance(tribunal, dict) else False,
            detail=str(tribunal.get("detail", "")) if isinstance(tribunal, dict) else "",
        )

    def analyze(self, payload: JsonObject) -> JsonObject:
        return self._request("POST", "/v1/analyze", json=payload, timeout=30.0)

    def review(self, payload: JsonObject) -> JsonObject:
        return self._request("POST", "/v1/review", json=payload, timeout=300.0)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: JsonObject | None = None,
        timeout: float,
    ) -> JsonObject:
        try:
            response = self._client.request(method, path, json=json, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise ApiError("Перевищено час очікування відповіді сервера.", code="timeout") from exc
        except httpx.HTTPError as exc:
            raise ApiError("Не вдалося встановити зв’язок із сервером.") from exc

        if response.status_code >= HTTP_ERROR_START:
            self._raise_response_error(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise ApiError(
                "Сервер повернув відповідь у невідомому форматі.",
                code="invalid_response",
                status_code=response.status_code,
                request_id=response.headers.get("x-request-id"),
            ) from exc
        if not isinstance(body, dict):
            raise ApiError(
                "Сервер повернув відповідь у невідомому форматі.",
                code="invalid_response",
                status_code=response.status_code,
                request_id=response.headers.get("x-request-id"),
            )
        return body

    @staticmethod
    def _raise_response_error(response: httpx.Response) -> None:
        try:
            body = response.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        request_id = body.get("request_id") or response.headers.get("x-request-id")
        message = body.get("message")
        code = body.get("code")
        raise ApiError(
            str(message) if message else _fallback_message(response.status_code),
            code=str(code) if code else f"http_{response.status_code}",
            status_code=response.status_code,
            request_id=str(request_id) if request_id else None,
        )


def _fallback_message(status_code: int) -> str:
    messages = {
        422: "Перевірте введені дані.",
        429: "Досягнуто обмеження частоти запитів. Спробуйте пізніше.",
        502: "Зовнішній сервіс не зміг завершити розгляд.",
        503: "Обраний компонент системи тимчасово недоступний.",
        504: "Зовнішній сервіс не відповів вчасно.",
    }
    return messages.get(status_code, "Сервер не зміг опрацювати запит.")
