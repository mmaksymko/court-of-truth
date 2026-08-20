import httpx
import pytest

from court.config import Settings
from court.ingest.fetch import IngestError, fetch_article


async def public_resolver(_host: str, _port: int) -> list[str]:
    return ["93.184.216.34"]


@pytest.mark.asyncio
async def test_fetch_extracts_article_and_follows_safe_redirect():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/article"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=(
                "<html><head><title>Новина</title><script>bad()</script></head>"
                "<body><article><p>Перший змістовний абзац матеріалу.</p>"
                "<p>Другий змістовний абзац із деталями.</p></article></body></html>"
            ).encode(),
        )

    settings = Settings(_env_file=None, fetch_max_redirects=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        article = await fetch_article(
            "https://example.org/start", client, settings, public_resolver
        )
    assert seen == ["https://example.org/start", "https://example.org/article"]
    assert article.title == "Новина"
    assert "bad()" not in article.text
    assert article.source_url


@pytest.mark.asyncio
async def test_declared_and_streamed_size_limits():
    settings = Settings(_env_file=None, fetch_max_bytes=10)

    def declared(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-length": "100"},
            content=b"x",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(declared)) as client:
        with pytest.raises(IngestError, match="too large"):
            await fetch_article("https://example.org", client, settings, public_resolver)

    def observed(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=b"01234567890",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(observed)) as client:
        with pytest.raises(IngestError, match="too large"):
            await fetch_article("https://example.org", client, settings, public_resolver)


@pytest.mark.asyncio
async def test_content_type_and_status_are_checked():
    settings = Settings(_env_file=None)

    def binary(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"png")

    async with httpx.AsyncClient(transport=httpx.MockTransport(binary)) as client:
        with pytest.raises(IngestError, match="HTML"):
            await fetch_article("https://example.org", client, settings, public_resolver)

    def failed(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, headers={"content-type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failed)) as client:
        with pytest.raises(IngestError, match="404"):
            await fetch_article("https://example.org", client, settings, public_resolver)
