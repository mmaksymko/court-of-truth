import httpx
import pytest

from court.ui.api import ApiError, CourtApi


def test_health_and_operation_paths_use_expected_contract():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/v1/health":
            return httpx.Response(
                200,
                json={
                    "components": {
                        "forensics": {"ready": True, "detail": "loaded"},
                        "tribunal": {"ready": True, "detail": "configured"},
                    }
                },
            )
        return httpx.Response(200, json={"ok": True})

    api = CourtApi("http://test", transport=httpx.MockTransport(handler))
    state = api.health()
    assert state.forensics and state.tribunal
    assert api.analyze({"title": "x"}) == {"ok": True}
    assert api.review({"title": "x"}) == {"ok": True}
    assert seen == [("GET", "/v1/health"), ("POST", "/v1/analyze"), ("POST", "/v1/review")]


@pytest.mark.parametrize("status", [422, 429, 502, 503, 504])
def test_error_envelope_is_preserved(status):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"code": "test_error", "message": "опис", "request_id": "req-17"},
        )

    api = CourtApi("http://test", transport=httpx.MockTransport(handler))
    with pytest.raises(ApiError) as caught:
        api.review({"title": "x"})
    assert caught.value.status_code == status
    assert caught.value.code == "test_error"
    assert caught.value.request_id == "req-17"


def test_malformed_success_response_is_rejected():
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, text="not-json"))
    api = CourtApi("http://test", transport=transport)
    with pytest.raises(ApiError, match="невідомому форматі") as caught:
        api.analyze({"title": "x"})
    assert caught.value.code == "invalid_response"


def test_network_error_is_translated():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed", request=request)

    api = CourtApi("http://test", transport=httpx.MockTransport(handler))
    with pytest.raises(ApiError, match="зв’язок") as caught:
        api.health()
    assert caught.value.code == "network_error"
