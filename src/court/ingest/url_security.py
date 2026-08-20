import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Sequence
from urllib.parse import urlsplit

from court.ingest.errors import IngestError

Resolver = Callable[[str, int], Awaitable[Sequence[str]]]
_BLOCKED_HOSTS = {"localhost", "localhost.localdomain"}


async def resolve_host(host: str, port: int) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    try:
        records = await loop.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise IngestError("dns_failed", "source host could not be resolved", 502) from exc
    return sorted({str(record[4][0]) for record in records})


async def validate_public_url(url: str, resolver: Resolver) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise IngestError("unsafe_url", "only http and https URLs are accepted", 422)
    if parsed.username or parsed.password or parsed.fragment:
        raise IngestError("unsafe_url", "URL credentials and fragments are not accepted", 422)
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host or host in _BLOCKED_HOSTS:
        raise IngestError("unsafe_url", "URL host is not public", 422)
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise IngestError("unsafe_url", "URL port is invalid", 422) from exc
    if port not in {80, 443}:
        raise IngestError("unsafe_url", "URL port is not allowed", 422)

    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        addresses = await resolver(host, port)
    else:
        addresses = [str(literal)]
    if not addresses:
        raise IngestError("dns_failed", "source host returned no addresses", 502)
    if any(not _is_global(value) for value in addresses):
        raise IngestError("unsafe_url", "URL resolves to a non-public address", 422)


def _is_global(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError as exc:
        raise IngestError("dns_failed", "source host returned an invalid address", 502) from exc
