import gzip

import httpx
import pytest

from court.config import Settings
from court.ingest.fetch import IngestError, extract, fetch_article
from court.ingest.schemas import ReviewRequest


async def public_resolver(_host: str, _port: int) -> list[str]:
    return ["93.184.216.34"]


def test_plain_text_extract_and_input_normalization():
    assert extract(b"  one\n two ", "text/plain") == ("", "one two")
    assert extract("текст".encode(), "text/plain; charset=not-a-codec") == ("", "текст")
    request = ReviewRequest(title="  Заголовок  ")
    assert request.title == "  Заголовок  "


@pytest.mark.asyncio
async def test_compressed_source_is_rejected_before_reading():
    def compressed(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-encoding": "gzip"},
            content=gzip.compress(b"compressed"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(compressed)) as client:
        with pytest.raises(IngestError, match="compressed"):
            await fetch_article(
                "https://example.org",
                client,
                Settings(_env_file=None),
                public_resolver,
            )
