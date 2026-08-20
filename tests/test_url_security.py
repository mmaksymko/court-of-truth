import httpx
import pytest

from court.config import Settings
from court.ingest.fetch import IngestError, fetch_article, validate_public_url


async def public_resolver(_host: str, _port: int) -> list[str]:
    return ["93.184.216.34"]


@pytest.mark.asyncio
async def test_private_and_unsafe_urls_are_rejected():
    async def private(_host: str, _port: int) -> list[str]:
        return ["127.0.0.1"]

    values = [
        "file:///etc/passwd",
        "http://user:pass@example.org/",
        "http://example.org:8080/",
        "http://localhost/",
    ]
    for value in values:
        with pytest.raises(IngestError, match=r"URL|http"):
            await validate_public_url(value, public_resolver)
    with pytest.raises(IngestError, match="non-public"):
        await validate_public_url("http://example.org", private)


@pytest.mark.asyncio
async def test_redirect_target_is_revalidated():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/metadata"})

    settings = Settings(_env_file=None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(IngestError, match="non-public"):
            await fetch_article("https://example.org", client, settings, public_resolver)
