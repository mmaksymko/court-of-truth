import asyncio

from starlette.testclient import TestClient

from court.tribunal.errors import TribunalError
from tests.fakes import FakeLLM, make_settings


def test_rate_limit_is_enforced(app_factory):
    configured = make_settings(operation_rate_per_minute=1)
    with TestClient(app_factory(llm=FakeLLM(), configured=configured)) as client:
        assert client.post("/v1/review", json={"title": "Перша"}).status_code == 200
        response = client.post("/v1/review", json={"title": "Друга"})
    assert response.status_code == 429
    assert response.json()["code"] == "rate_limited"


def test_concurrency_limit_class():
    from court.api.limits import LimitExceededError, OperationLimits

    async def exercise():
        limits = OperationLimits(rate_per_minute=10, concurrency=1)
        async with limits.slot():
            try:
                async with limits.slot():
                    raise AssertionError("second slot should not be entered")
            except LimitExceededError:
                return
        raise AssertionError("limit was not raised")

    asyncio.run(exercise())


def test_expensive_endpoints_share_rate_limit(app_factory):
    configured = make_settings(operation_rate_per_minute=1)
    with TestClient(app_factory(llm=FakeLLM(), configured=configured)) as client:
        assert client.post("/v1/analyze", json={"title": "Перша новина дня"}).status_code == 200
        response = client.post("/v1/review", json={"title": "Друга новина дня"})
    assert response.status_code == 429


def test_tribunal_error_uses_stable_envelope(app_factory):
    class FailedJudge(FakeLLM):
        async def judge(self, _user: str):
            raise TribunalError(502, "llm_upstream_error", "tribunal provider failed")

    with TestClient(app_factory(llm=FailedJudge())) as client:
        response = client.post("/v1/review", json={"title": "Новина"})
    assert response.status_code == 502
    assert response.json()["code"] == "llm_upstream_error"
    assert response.json()["message"] == "tribunal provider failed"


def test_llm_client_is_closed_with_app(app_factory):
    llm = FakeLLM()
    with TestClient(app_factory(llm=llm)):
        assert not llm.closed
    assert llm.closed
