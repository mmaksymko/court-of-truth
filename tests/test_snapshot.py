import hashlib

import httpx
import pytest

from court.ingest.snapshot import PageArchive
from tests.fakes import make_settings

HTML = (
    "<html><head><title>Джерело</title></head><body><article>"
    "<p>Перший абзац доказової сторінки з достатньою кількістю тексту для витягу.</p>"
    "</article></body></html>"
)


async def public_resolver(_host: str, _port: int) -> list[str]:
    return ["93.184.216.34"]


def _archive(tmp_path, handler) -> PageArchive:
    return PageArchive(
        tmp_path / "cache",
        make_settings(),
        transport=httpx.MockTransport(handler),
        resolver=public_resolver,
    )


@pytest.mark.asyncio
async def test_saves_text_copy_once_per_url_with_hash(tmp_path):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"}, content=HTML.encode())

    archive = _archive(tmp_path, handler)
    try:
        entries = await archive.snapshot_all(
            ["https://example.org/a", "https://example.org/a", "https://example.org/b"]
        )
    finally:
        await archive.aclose()

    assert [entry["url"] for entry in entries] == ["https://example.org/a", "https://example.org/b"]
    assert seen == ["/robots.txt", "/a", "/b"]
    first = entries[0]
    assert first["status"] == "saved"
    assert str(first["fetched_at"]).endswith("Z")
    saved = (tmp_path / "cache" / f"{first['sha256']}.txt").read_bytes()
    assert first["path"] == str(tmp_path / "cache" / f"{first['sha256']}.txt")
    assert hashlib.sha256(saved).hexdigest() == first["sha256"]
    assert saved.decode().startswith("Джерело\n\nПерший абзац")


@pytest.mark.asyncio
async def test_robots_disallow_skips_without_fetching_page(tmp_path):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, content=b"User-agent: *\nDisallow: /private\n")

    archive = _archive(tmp_path, handler)
    try:
        entry = (await archive.snapshot("https://example.org/private/x")).as_dict()
    finally:
        await archive.aclose()
    assert entry["status"] == "skipped"
    assert entry["reason"] == "robots_disallowed"
    assert entry["path"] is None
    assert seen == ["/robots.txt"]


@pytest.mark.asyncio
async def test_failures_are_recorded_not_raised(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "down.example.org":
            raise httpx.ConnectError("refused")
        if request.url.path == "/robots.txt":
            return httpx.Response(503)
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")

    archive = _archive(tmp_path, handler)
    try:
        robots = await archive.snapshot("https://example.org/doc")
        unsafe = await archive.snapshot("http://localhost/x")
        down = await archive.snapshot("https://down.example.org/x")
    finally:
        await archive.aclose()
    assert (robots.status, robots.reason) == ("failed", "robots_unavailable")
    assert (unsafe.status, unsafe.reason) == ("failed", "unsafe_url")
    assert (down.status, down.reason) == ("failed", "fetch_error")


@pytest.mark.asyncio
async def test_unsupported_content_and_write_errors(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/pdf":
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")
        return httpx.Response(200, headers={"content-type": "text/html"}, content=HTML.encode())

    blocker = tmp_path / "cache"
    blocker.write_text("not a directory")
    archive = _archive(tmp_path, handler)
    try:
        pdf = await archive.snapshot("https://example.org/pdf")
        page = await archive.snapshot("https://example.org/page")
    finally:
        await archive.aclose()
    assert (pdf.status, pdf.reason) == ("failed", "unsupported_content")
    assert (page.status, page.reason) == ("failed", "write_error")
