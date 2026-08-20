from starlette.testclient import TestClient

from court.config import Settings


def test_closed_request_shapes(client: TestClient):
    invalid = [
        {},
        {"title": "", "text": ""},
        {"text": ""},
        {"title": "заголовок", "junk": 1},
    ]
    for payload in invalid:
        response = client.post("/v1/analyze", json=payload)
        assert response.status_code == 422
        assert response.json()["code"] == "invalid_request"


def test_analyze_rejects_empty_and_extra(client: TestClient):
    assert client.post("/v1/analyze", json={}).status_code == 422
    assert client.post("/v1/analyze", json={"title": "x", "junk": 1}).status_code == 422


def test_not_found_and_method_errors_use_envelope(client: TestClient):
    assert client.get("/v1/missing").json()["code"] == "not_found"
    response = client.get("/v1/analyze")
    assert response.status_code == 405
    assert response.json()["code"] == "method_not_allowed"


def test_body_too_large_rejected(app_factory):
    configured = Settings(_env_file=None, max_body_bytes=64)
    with TestClient(app_factory(configured=configured)) as client:
        response = client.post("/v1/analyze", json={"text": "а" * 80})
    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"
    assert response.headers["x-request-id"]


def test_body_limit_counts_streamed_bytes_not_content_length(app_factory):
    configured = Settings(_env_file=None, max_body_bytes=64)
    with TestClient(app_factory(configured=configured)) as client:
        understated = client.post(
            "/v1/analyze",
            content=b'{"title":"' + b"x" * 80 + b'"}',
            headers={"content-length": "1", "content-type": "application/json"},
        )
        overstated = client.post(
            "/v1/analyze",
            content=b'{"title":"small"}',
            headers={"content-length": "999", "content-type": "application/json"},
        )
        chunked = client.post(
            "/v1/analyze",
            content=iter([b'{"title":"', b"small", b'"}']),
            headers={"content-type": "application/json"},
        )
    assert understated.status_code == 413
    assert overstated.status_code == 200
    assert chunked.status_code == 200


def test_error_carries_request_id(app_factory):
    application = app_factory()

    async def boom() -> None:
        raise RuntimeError("boom")

    application.add_api_route("/v1/_boom", boom, methods=["GET"])
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/v1/_boom", headers={"x-request-id": "trace-boom"})
    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert response.headers["x-request-id"] == "trace-boom"
