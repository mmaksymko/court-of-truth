import asyncio
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

_WINDOW_SECONDS = 60.0


class LimitExceededError(RuntimeError):
    pass


class OperationLimits:
    def __init__(self, rate_per_minute: int, concurrency: int) -> None:
        self._rate = rate_per_minute
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._active_total = 0
        self._concurrency_limit = concurrency
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

    async def check_rate(self, subject: str) -> None:
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS
        async with self._lock:
            if now - self._last_cleanup >= _WINDOW_SECONDS:
                self._calls = defaultdict(
                    deque,
                    {
                        key: calls
                        for key, calls in self._calls.items()
                        if calls and calls[-1] > cutoff
                    },
                )
                self._last_cleanup = now
            calls = self._calls[subject]
            while calls and calls[0] <= cutoff:
                calls.popleft()
            if len(calls) >= self._rate:
                raise LimitExceededError("operation rate limit exceeded")
            calls.append(now)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._active_total >= self._concurrency_limit:
                raise LimitExceededError("operation concurrency limit reached")
            self._active_total += 1
        try:
            yield
        finally:
            async with self._lock:
                self._active_total -= 1
