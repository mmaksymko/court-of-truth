import httpx
from bs4 import BeautifulSoup

from court.ingest.errors import IngestError

_ALLOWED_TYPES = {"application/xhtml+xml", "text/html", "text/plain"}
_MIN_PARAGRAPH_TEXT = 100


def extract(raw: bytes, content_type: str) -> tuple[str, str]:
    decoded = _decode(raw, _charset(content_type))
    if _media_type(content_type) == "text/plain":
        return "", _normalize(decoded)

    soup = BeautifulSoup(decoded, "html.parser")
    for node in soup(["script", "style", "noscript", "template", "svg"]):
        node.decompose()
    og_title = soup.find("meta", attrs={"property": "og:title"})
    title = ""
    if og_title and og_title.get("content"):
        title = str(og_title["content"])
    elif soup.title and soup.title.string:
        title = soup.title.string

    container = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [_normalize(node.get_text(" ", strip=True)) for node in container.find_all("p")]
    text = "\n\n".join(part for part in paragraphs if part)
    if len(text) < _MIN_PARAGRAPH_TEXT:
        text = _normalize(container.get_text(" ", strip=True))
    return _normalize(title), text


def check_response(response: httpx.Response, max_bytes: int) -> None:
    if _media_type(response.headers.get("content-type", "")) not in _ALLOWED_TYPES:
        raise IngestError("unsupported_content", "source is not HTML or plain text", 415)
    encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if encoding not in {"", "identity"}:
        raise IngestError(
            "unsupported_encoding",
            "compressed source responses are not accepted",
            415,
        )
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise IngestError("source_too_large", "source body is too large", 413)


async def read_limited(response: httpx.Response, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > limit:
            raise IngestError("source_too_large", "source body is too large", 413)
        chunks.append(chunk)
    return b"".join(chunks)


def _decode(raw: bytes, encoding: str) -> str:
    try:
        return raw.decode(encoding, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _media_type(content_type: str) -> str:
    return content_type.partition(";")[0].strip().lower()


def _charset(content_type: str) -> str:
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            return value.strip("\"'")
    return "utf-8"


def _normalize(value: str) -> str:
    return " ".join(value.split()).strip()
