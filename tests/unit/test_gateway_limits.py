"""Token buckets, per-tenant quotas and the Retry-After arithmetic behind every 429.

The clock is injected everywhere, so these tests assert exact refill values rather than
sleeping and hoping.
"""

from __future__ import annotations

import math

import pytest

from turboserve.gateway.limits import (
    LimiterRegistry,
    RateLimitExceeded,
    TenantLimiter,
    TokenBucket,
    retry_after_header,
)
from turboserve.gateway.tenants import Tenant, TenantRegistry


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


# -- TokenBucket ------------------------------------------------------------------------


def test_bucket_starts_full_and_drains() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=1, now=0.0)
    assert bucket.level(now=0.0) == 10
    assert bucket.try_acquire(4, now=0.0) is True
    assert bucket.level(now=0.0) == 6
    assert bucket.try_acquire(7, now=0.0) is False
    assert bucket.level(now=0.0) == 6  # a refused acquire spends nothing


def test_bucket_refills_linearly_and_caps_at_capacity() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=2, now=0.0)
    bucket.try_acquire(10, now=0.0)
    assert bucket.level(now=1.0) == pytest.approx(2.0)
    assert bucket.level(now=2.5) == pytest.approx(5.0)
    assert bucket.level(now=1000.0) == 10  # never exceeds the burst


def test_per_minute_builds_the_budget_operators_write() -> None:
    bucket = TokenBucket.per_minute(600, now=0.0)
    assert bucket.capacity == 600
    assert bucket.refill_per_second == pytest.approx(10.0)


def test_charge_may_overdraw_and_retry_after_accounts_for_the_deficit() -> None:
    # This is the tpm case: the completion's length is only known after it is generated.
    bucket = TokenBucket(capacity=100, refill_per_second=10, now=0.0)
    bucket.charge(250, now=0.0)
    assert bucket.level(now=0.0) == pytest.approx(-150.0)
    # 150 to get back to zero plus 50 for the next request, at 10/s.
    assert bucket.retry_after(50, now=0.0) == pytest.approx(20.0)


def test_refund_returns_allowance_without_exceeding_capacity() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=1, now=0.0)
    bucket.try_acquire(5, now=0.0)
    assert bucket.refund(3, now=0.0) == pytest.approx(8.0)
    assert bucket.refund(100, now=0.0) == 10


def test_retry_after_is_zero_when_allowance_exists() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=1, now=0.0)
    assert bucket.retry_after(5, now=0.0) == 0.0


def test_bucket_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        TokenBucket(capacity=0, refill_per_second=1)
    with pytest.raises(ValueError, match="refill_per_second must be positive"):
        TokenBucket(capacity=1, refill_per_second=0)
    with pytest.raises(ValueError, match="must not be negative"):
        TokenBucket(capacity=1, refill_per_second=1).try_acquire(-1)


@pytest.mark.parametrize(
    ("seconds", "expected"), [(0.0, "1"), (0.2, "1"), (1.0, "1"), (30.4, "31")]
)
def test_retry_after_header_rounds_up_to_whole_seconds(seconds: float, expected: str) -> None:
    # RFC 9110 permits only integer seconds; flooring at 1 stops a client busy-looping.
    assert retry_after_header(seconds) == expected
    assert int(retry_after_header(seconds)) >= math.ceil(min(seconds, 1.0))


# -- TenantLimiter ----------------------------------------------------------------------


def test_rpm_refuses_with_a_computable_wait() -> None:
    clock = FakeClock()
    limiter = TenantLimiter(Tenant(tenant_id="acme", rpm=2), clock=clock)
    limiter.check_request()
    limiter.check_request()
    with pytest.raises(RateLimitExceeded) as exc:
        limiter.check_request()
    assert exc.value.limit == "rpm"
    assert exc.value.limit_value == 2
    assert exc.value.retry_after_s == pytest.approx(30.0)  # 1 request at 2/minute

    clock.advance(30.0)
    limiter.check_request()  # the bucket refilled exactly one request


def test_tpm_refuses_on_the_prompt_and_is_charged_again_on_completion() -> None:
    clock = FakeClock()
    limiter = TenantLimiter(Tenant(tenant_id="acme", tpm=600), clock=clock)
    limiter.check_request(prompt_tokens=500)
    limiter.charge_tokens(300)  # completion turned out longer than the remaining budget
    assert limiter.tokens_charged == 800
    with pytest.raises(RateLimitExceeded) as exc:
        limiter.check_request(prompt_tokens=100)
    assert exc.value.limit == "tpm"
    # 200 tokens overdrawn plus 100 wanted, refilling at 10 tokens/s.
    assert exc.value.retry_after_s == pytest.approx(30.0)


def test_a_request_refused_by_tpm_does_not_burn_the_rpm_budget() -> None:
    clock = FakeClock()
    limiter = TenantLimiter(Tenant(tenant_id="acme", rpm=10, tpm=100), clock=clock)
    with pytest.raises(RateLimitExceeded):
        limiter.check_request(prompt_tokens=500)
    assert limiter.snapshot().rpm_level == 10
    assert limiter.admitted == 0
    assert limiter.rejected == 1


def test_concurrency_refuses_immediately_instead_of_queueing() -> None:
    limiter = TenantLimiter(Tenant(tenant_id="acme", max_concurrency=2))
    with limiter.slot(), limiter.slot():
        assert limiter.in_flight == 2
        with pytest.raises(RateLimitExceeded) as exc, limiter.slot():
            pass
        assert exc.value.limit == "concurrency"
        assert exc.value.retry_after_s > 0
    assert limiter.in_flight == 0


def test_the_concurrency_slot_is_released_when_the_body_raises() -> None:
    limiter = TenantLimiter(Tenant(tenant_id="acme", max_concurrency=1))
    with pytest.raises(RuntimeError), limiter.slot():
        raise RuntimeError("stream blew up")
    assert limiter.in_flight == 0


def test_unset_limits_mean_unlimited() -> None:
    limiter = TenantLimiter(Tenant(tenant_id="labs"))
    for _ in range(100):
        limiter.check_request(prompt_tokens=10_000)
    assert limiter.admitted == 100
    snapshot = limiter.snapshot()
    assert snapshot.rpm_level is None
    assert snapshot.tpm_level is None


def test_snapshot_reports_state_for_diagnostics() -> None:
    clock = FakeClock()
    limiter = TenantLimiter(Tenant(tenant_id="acme", rpm=10, tpm=100), clock=clock)
    limiter.check_request(prompt_tokens=20)
    data = limiter.snapshot().to_dict()
    assert data["tenant_id"] == "acme"
    assert data["rpm_level"] == pytest.approx(9.0)
    assert data["tpm_level"] == pytest.approx(80.0)
    assert data["admitted"] == 1


# -- LimiterRegistry --------------------------------------------------------------------


def test_registry_creates_one_limiter_per_tenant_lazily() -> None:
    tenants = TenantRegistry([Tenant(tenant_id="a", rpm=1), Tenant(tenant_id="b")])
    registry = LimiterRegistry(tenants)
    assert len(registry) == 0
    first = registry.for_tenant(tenants["a"])
    assert registry.for_tenant(tenants["a"]) is first  # state must be shared, not rebuilt
    assert len(registry) == 1
    assert registry.for_tenant_id("b") is not None
    assert registry.for_tenant_id("nobody") is None


def test_registry_snapshot_and_reset() -> None:
    tenants = TenantRegistry([Tenant(tenant_id="a", rpm=5)])
    registry = LimiterRegistry(tenants)
    registry.for_tenant(tenants["a"]).check_request()
    assert registry.snapshot()["a"]["admitted"] == 1
    registry.reset()
    assert registry.snapshot() == {}
