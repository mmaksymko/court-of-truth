from collections.abc import Callable, Iterator

import httpx
import pytest
from starlette.testclient import TestClient

from court.api.app import create_app
from court.config import Settings
from court.forensics.registry import LoadedDetector
from tests.fakes import make_detector


@pytest.fixture
def fake_registry() -> dict[str, LoadedDetector]:
    return {
        "jeansa": make_detector("jeansa", "body", ("editorial", "sponsored"), "sponsored", 0.9),
        "clickbait": make_detector(
            "clickbait", "title", ("clickbait", "neutral"), "clickbait", 0.2
        ),
    }


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        detector_workers=2,
        operation_rate_per_minute=10,
        operation_concurrency=2,
    )


@pytest.fixture
def app_factory(
    settings: Settings,
    fake_registry: dict[str, LoadedDetector],
) -> Callable[..., object]:
    def factory(
        *,
        configured: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        return create_app(
            configured or settings,
            registry_loader=lambda _settings: fake_registry,
            transport=transport,
        )

    return factory


@pytest.fixture
def client(app_factory) -> Iterator[TestClient]:
    with TestClient(app_factory()) as test_client:
        yield test_client
