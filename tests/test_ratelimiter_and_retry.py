"""Rate limiter spacing and the provider retry/backoff/n-a loop.

Uses a fake clock and fake sleep so no real time passes and no network is hit.
"""

from __future__ import annotations

from fleetbench.providers.base import Completion, Provider, ProviderError, RateLimiter


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_ratelimiter_spaces_calls_by_min_interval():
    ft = FakeTime()
    rl = RateLimiter(rpm=60, clock=ft.clock, sleep=ft.sleep)  # 1s spacing
    rl.acquire()  # first is free
    rl.acquire()  # must wait ~1s
    rl.acquire()  # another ~1s
    assert ft.sleeps == [1.0, 1.0]


def test_ratelimiter_unlimited_never_sleeps():
    ft = FakeTime()
    rl = RateLimiter(rpm=0, clock=ft.clock, sleep=ft.sleep)
    for _ in range(5):
        rl.acquire()
    assert ft.sleeps == []


class _FlakyProvider(Provider):
    name = "flaky"

    def __init__(self, fail_times, retryable=True, **kw):
        super().__init__(**kw)
        self.fail_times = fail_times
        self.retryable = retryable
        self.calls = 0

    def available_models(self):
        return ["m"]

    def call(self, model, prompt):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("boom", retryable=self.retryable)
        return Completion(text="ok", latency_s=0.1)


def test_retry_succeeds_after_transient_failures():
    ft = FakeTime()
    p = _FlakyProvider(fail_times=2, clock=ft.clock, sleep=ft.sleep)
    result = p.run_item("m", "hi")
    assert result.na_reason is None
    assert result.text == "ok"
    assert result.attempts == 3
    assert ft.sleeps == [1.0, 2.0]  # exponential backoff schedule


def test_na_after_three_failures():
    ft = FakeTime()
    p = _FlakyProvider(fail_times=5, clock=ft.clock, sleep=ft.sleep)
    result = p.run_item("m", "hi")
    assert result.is_na
    assert "after retries" in result.na_reason
    assert p.calls == 3  # MAX_ATTEMPTS


def test_non_retryable_error_stops_immediately():
    ft = FakeTime()
    p = _FlakyProvider(fail_times=5, retryable=False, clock=ft.clock, sleep=ft.sleep)
    result = p.run_item("m", "hi")
    assert result.is_na
    assert p.calls == 1  # no retry on non-retryable
    assert ft.sleeps == []
