"""One polite HTTP client, shared by every fetcher in the project.

The rate limit is a *global* token bucket, not a per-thread sleep. Eight threads each
sleeping one second is eight requests per second at the server, which is how a project
gets its IP blocked from a volunteer-run API. One bucket, refilled at a fixed rate,
handed out to whichever thread asks first, is the only version that actually holds.

Retry policy, in order of how OSM signals distress:

  509  Bandwidth Limit Exceeded - this is the explicit "you are going too fast" signal.
       Back off exponentially and keep the backoff, do not reset it on the next success.
  429  Too Many Requests - same treatment, honouring Retry-After when present.
  5xx  Server trouble, usually transient. Retry with jitter.
  4xx  Our fault. Do not retry; return the response and let the caller record it.

A failed fetch returns its status rather than raising, because the coverage number in
the paper has to be measured rather than assumed.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import requests

from mapguard.config import HEADERS, TIMEOUT


class TokenBucket:
    """Global rate limiter. `take()` blocks until a token is available."""

    def __init__(self, rate: float, burst: float | None = None):
        self.rate = float(rate)
        self.capacity = float(burst if burst is not None else max(1.0, rate))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def take(self, n: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait = deficit / self.rate
            time.sleep(min(wait, 1.0))


@dataclass
class Response:
    """What a fetch produced, whether or not it worked."""

    url: str
    status: int          # HTTP status, or 0 when the request never completed
    text: str
    seconds: float
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200


class PoliteSession:
    """A requests.Session behind a shared token bucket and a backoff policy."""

    def __init__(
        self,
        rate: float,
        *,
        max_retries: int = 5,
        timeout: int = TIMEOUT,
        headers: dict[str, str] | None = None,
    ):
        self.bucket = TokenBucket(rate)
        self.max_retries = max_retries
        self.timeout = timeout
        self.headers = dict(headers or HEADERS)
        self._local = threading.local()
        # Set when the server tells us to slow down; every thread waits it out.
        self._penalty_until = 0.0
        self._penalty_lock = threading.Lock()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(self.headers)
            self._local.session = s
        return s

    def _serve_penalty(self) -> None:
        while True:
            with self._penalty_lock:
                remaining = self._penalty_until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 2.0))

    def _penalise(self, seconds: float) -> None:
        with self._penalty_lock:
            self._penalty_until = max(self._penalty_until, time.monotonic() + seconds)

    def get(self, url: str, **kwargs) -> Response:
        started = time.monotonic()
        last_error: str | None = None
        status = 0
        text = ""

        for attempt in range(1, self.max_retries + 1):
            self._serve_penalty()
            self.bucket.take()
            try:
                r = self._session().get(url, timeout=self.timeout, **kwargs)
                status, text = r.status_code, r.text

                if status == 200:
                    return Response(url, status, text, time.monotonic() - started, attempt)

                if status in (429, 509):
                    # The explicit slow-down signals. Penalise everyone, not just us.
                    retry_after = r.headers.get("Retry-After")
                    wait = float(retry_after) if (retry_after or "").isdigit() else 2**attempt
                    self._penalise(min(wait, 120.0))
                    last_error = f"HTTP {status}, backing off {wait:.0f}s"
                    continue

                if 500 <= status < 600:
                    time.sleep(min(2**attempt, 30) * (0.5 + random.random()))
                    last_error = f"HTTP {status}"
                    continue

                # 4xx that is not a rate limit is our problem, and it is final.
                return Response(url, status, text, time.monotonic() - started, attempt)

            except requests.RequestException as exc:
                last_error = repr(exc)[:200]
                status = 0
                time.sleep(min(2**attempt, 30) * (0.5 + random.random()))

        return Response(url, status, text, time.monotonic() - started, self.max_retries, last_error)
