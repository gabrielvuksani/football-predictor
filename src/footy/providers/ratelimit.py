from __future__ import annotations
import logging
import time
from collections import deque

import httpx

log = logging.getLogger(__name__)

# Network errors worth retrying (DNS, connect, reset, timeout)
TRANSIENT_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    ConnectionResetError,
    OSError,
)

_DEFAULT_BACKOFF = (2.0, 5.0, 10.0, 20.0)


class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: deque[float] = deque()

    def wait(self):
        now = time.monotonic()
        while self.calls and now - self.calls[0] > self.period:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            sleep_for = self.period - (now - self.calls[0]) + 0.01
            time.sleep(max(0.0, sleep_for))
        self.calls.append(time.monotonic())


def retry_request(
    fn,
    *,
    max_retries: int = 4,
    backoff: tuple[float, ...] = _DEFAULT_BACKOFF,
    label: str = "HTTP",
):
    """Call *fn* (which should return an httpx.Response or value), retrying on transient network errors.

    Args:
        fn: callable that performs the HTTP request
        max_retries: total attempts (including first)
        backoff: sleep durations between retries
        label: provider name for log messages

    Returns:
        Whatever *fn* returns on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except TRANSIENT_ERRORS as exc:
            last_exc = exc
            wait = backoff[min(attempt, len(backoff) - 1)]
            log.warning("%s transient error (attempt %d/%d): %s — retrying in %.1fs",
                        label, attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
    # All retries exhausted
    raise last_exc  # type: ignore[misc]
