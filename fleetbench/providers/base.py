"""Provider base: a rate-limited worker abstraction shared by every provider.

A provider wraps one API or CLI that can serve one or more models. The runner
gives each provider its own worker so providers run concurrently, while each
provider throttles itself to its own requests-per-minute (RPM) budget.

Design notes for testability:
- The retry loop takes injectable ``clock`` and ``sleep`` callables so unit
  tests exercise backoff and the n/a-after-3 rule without real time passing and
  without any network.
- ``call`` is the only network/subprocess seam a concrete provider implements.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..scrub import scrub


class ProviderError(Exception):
    """Raised by ``call`` when a request fails.

    ``retryable`` marks transient failures (HTTP 429/5xx, timeouts) that the
    retry loop should back off and retry. Non-retryable errors (a model not on
    the plan, a 4xx other than 429) stop immediately and surface as n/a.
    """

    def __init__(self, message: str, *, retryable: bool, status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class Completion:
    """The result of one model call, graded downstream."""

    text: str
    latency_s: float
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    attempts: int = 1
    # When set, this item did not produce a usable answer. It is reported, never
    # silently dropped (REQ-FB-02).
    na_reason: Optional[str] = None
    raw_meta: dict = field(default_factory=dict)

    @property
    def is_na(self) -> bool:
        return self.na_reason is not None


class RateLimiter:
    """A simple thread-safe RPM throttle.

    Enforces a minimum spacing of ``60 / rpm`` seconds between acquisitions.
    ``rpm <= 0`` means unlimited. Uses injectable ``clock``/``sleep`` so tests
    can drive it deterministically.
    """

    def __init__(
        self,
        rpm: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm = rpm
        self._min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = self._clock()
            wait = self._next_allowed - now
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
            self._next_allowed = max(now, self._next_allowed) + self._min_interval


class Provider:
    """Base class for all providers.

    Subclasses set ``name`` and implement ``call(model, prompt) -> Completion``
    (the raw request) and ``available_models() -> list[str]``. The base owns the
    rate limiter and the retry/backoff/n-a loop via ``run_item``.
    """

    name: str = "base"

    # Retry policy (REQ-FB-02): back off on 429/5xx, n/a after 3 failed attempts.
    MAX_ATTEMPTS = 3
    BASE_BACKOFF_S = 1.0
    MAX_BACKOFF_S = 30.0

    def __init__(
        self,
        *,
        rpm: float = 0.0,
        concurrency: int = 1,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm = rpm
        self.concurrency = concurrency
        self._clock = clock
        self._sleep = sleep
        self.limiter = RateLimiter(rpm, clock=clock, sleep=sleep)

    # --- seams a concrete provider implements ---------------------------------
    def call(self, model: str, prompt: str) -> Completion:  # pragma: no cover
        raise NotImplementedError

    def available_models(self) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    # --- the shared retry loop ------------------------------------------------
    def _backoff_seconds(self, attempt: int) -> float:
        # attempt is 1-based; exponential with a cap. Deterministic (no jitter)
        # so tests can assert the exact schedule.
        delay = self.BASE_BACKOFF_S * (2 ** (attempt - 1))
        return min(delay, self.MAX_BACKOFF_S)

    def run_item(self, model: str, prompt: str) -> Completion:
        """Run one prompt with rate limiting, backoff, and the n/a-after-3 rule.

        Returns a ``Completion``. On exhausted retries or a non-retryable error
        it returns a Completion with ``na_reason`` set (never raises for an
        expected provider failure).
        """
        last_error: Optional[str] = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            self.limiter.acquire()
            try:
                completion = self.call(model, prompt)
                completion.attempts = attempt
                return completion
            except ProviderError as exc:
                last_error = str(exc)
                if not exc.retryable or attempt == self.MAX_ATTEMPTS:
                    break
                self._sleep(self._backoff_seconds(attempt))
            except Exception as exc:  # unexpected: treat as non-retryable n/a
                last_error = f"{type(exc).__name__}: {exc}"
                break
        # Scrub identifiers and collapse whitespace: this reason may be
        # persisted to a public artifact and rendered into a markdown table.
        reason = scrub((last_error or "unknown").replace("\n", " ").replace("\r", " ")).strip()
        return Completion(
            text="",
            latency_s=0.0,
            attempts=self.MAX_ATTEMPTS,
            na_reason=f"error after retries: {reason}",
        )
