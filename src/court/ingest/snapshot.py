"""Text snapshots of evidence pages, kept so a verdict's sources can be re-checked later.

Each page is fetched through the same guarded path as article ingestion (public
http/https only, size and content-type limits, bounded redirects), only when the
site's robots.txt allows it. The extracted text is stored content-addressed as
``<sha256>.txt`` under the cache directory, and a journal entry records when it
was fetched and where the copy lives.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from court.ingest.content import read_limited
from court.ingest.errors import IngestError
from court.ingest.fetch import fetch_article
from court.ingest.url_security import Resolver, resolve_host, validate_public_url

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from court.config import Settings

USER_AGENT = "court-of-truth/0.2"
_ROBOTS_MAX_BYTES = 500_000
_SERVER_ERROR = 500
_CLIENT_ERROR = 400
_SUCCESS = 200

SnapshotStatus = Literal["saved", "skipped", "failed"]


@dataclass(frozen=True)
class PageSnapshot:
    url: str
    fetched_at: str
    status: SnapshotStatus
    path: str | None = None
    sha256: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PageArchive:
    """Fetches evidence pages once per URL and stores their extracted text."""

    def __init__(
        self,
        cache_dir: Path,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._settings = settings
        self._resolver = resolver or resolve_host
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.fetch_timeout_s),
            transport=transport,
            headers={"accept-encoding": "identity", "user-agent": USER_AGENT},
            trust_env=False,
        )
        self._robots: dict[str, RobotFileParser | None] = {}

    async def snapshot_all(self, urls: Iterable[str]) -> list[dict[str, object]]:
        """Snapshot each distinct URL in order and return the journal entries."""
        entries: list[dict[str, object]] = []
        for url in dict.fromkeys(urls):
            entries.append((await self.snapshot(url)).as_dict())
        return entries

    async def snapshot(self, url: str) -> PageSnapshot:
        fetched_at = _utc_now()
        try:
            if not await self._allowed(url):
                return PageSnapshot(url, fetched_at, "skipped", reason="robots_disallowed")
            article = await fetch_article(url, self._client, self._settings, self._resolver)
            text = f"{article.title}\n\n{article.text}\n" if article.title else f"{article.text}\n"
            payload = text.encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_dir / f"{digest}.txt"
            if not path.exists():
                path.write_bytes(payload)
        except IngestError as exc:
            return PageSnapshot(url, fetched_at, "failed", reason=exc.code)
        except httpx.HTTPError:
            return PageSnapshot(url, fetched_at, "failed", reason="fetch_error")
        except OSError:
            return PageSnapshot(url, fetched_at, "failed", reason="write_error")
        return PageSnapshot(url, fetched_at, "saved", path=str(path), sha256=digest)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        if origin not in self._robots:
            self._robots[origin] = await self._load_robots(f"{origin}/robots.txt")
        rules = self._robots[origin]
        return rules is None or rules.can_fetch(USER_AGENT, url)

    async def _load_robots(self, robots_url: str) -> RobotFileParser | None:
        """Parse robots.txt; ``None`` means no rules apply (missing file, RFC 9309)."""
        await validate_public_url(robots_url, self._resolver)
        async with self._client.stream("GET", robots_url, follow_redirects=False) as response:
            if _CLIENT_ERROR <= response.status_code < _SERVER_ERROR:
                return None
            if response.status_code != _SUCCESS:
                raise IngestError("robots_unavailable", "robots.txt is unavailable", 502)
            raw = await read_limited(response, _ROBOTS_MAX_BYTES)
        parser = RobotFileParser()
        parser.parse(raw.decode("utf-8", errors="replace").splitlines())
        return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
