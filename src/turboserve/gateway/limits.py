"""Per-tenant quotas: request-rate, token-rate and concurrency.

Three limits, three different failure modes, one response: ``429`` with a ``Retry-After``
header that is computed, not guessed.

* **rpm** -- requests per minute. A classic token bucket: the refill rate is the sustained
  limit and the bucket depth is the burst a tenant may spend at once.
* **tpm** -- tokens per minute. The same bucket, charged twice: once optimistically at
  admission with the prompt's token count, and once on completion with what the request
  actually produced. Completion length is unknowable at admission, so the bucket is allowed
  to go *negative*; the deficit simply pushes the tenant's next ``Retry-After`` out. This
  is both simpler and fairer than reserving ``max_tokens`` up front, which would penalise
  every request for the longest one it might have been.
* **concurrency** -- simultaneous in-flight requests. Not a bucket but a counter, and
  deliberately *not* an :class:`asyncio.Semaphore`: the gateway rejects rather than queues.
  A tenant at its concurrency limit gets an immediate 429 it can back off on, instead of an
  unbounded queue of connections that all time out together. A plain counter also works
  from any event loop, which matters because the limiter is shared with synchronous tests.

The clock is injectable everywhere (``now=`` arguments and a ``clock`` callable) so the
tests below assert exact refill arithmetic instead of sleeping.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator

    from turboserve.gateway.tenants import Tenant, TenantRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "LimitName",
    "LimiterRegistry",
    "RateLimitExceeded",
    "TenantLimiter",
    "TokenBucket",
    "retry_after_header",
]

#: Which quota refused the request; also the value of the ``limit`` metric label.
LimitName = Literal["rpm", "tpm", "concurrency"]

#: Seconds a client is asked to wait after hitting the concurrency limit. Unlike a bucket,
#: a concurrency limit has no refill rate to derive a wait from -- it clears when some other
#: request of the same tenant finishes, which is not predictable from here.
DEFAULT_CONCURRENCY_RETRY_AFTER_S = 1.0

_SECONDS_PER_MINUTE = 60.0


class RateLimitExceeded(Exception):
    """A tenant quota refused the request; the route turns this into a 429.

    ``retry_after_s`` is the wait after which the request would succeed given no other
    traffic, so it is safe to put straight into the ``Retry-After`` header.
    """

    def __init__(
        self,
        tenant_id: str,
        limit: LimitName,
        *,
        retry_after_s: float,
        limit_value: float | None = None,
        message: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.limit = limit
        self.retry_after_s = max(0.0, retry_after_s)
        self.limit_value = limit_value
        self.message = message or (
            f"tenant {tenant_id!r} exceeded its {limit} quota"
            + (f" of {limit_value:g}" if limit_value is not None else "")
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}; retry after {self.retry_after_s:.3f}s"


def retry_after_header(seconds: float) -> str:
    """Render a wait as an HTTP ``Retry-After`` value (whole seconds, at least one).

    RFC 9110 allows only an integer number of seconds or a date. Rounding *up* and flooring
    at one keeps a client from busy-looping on a sub-second wait it cannot express.
    """
    return str(max(1, math.ceil(seconds)))


class TokenBucket:
    """A continuously-refilling bucket of allowance.

    ``capacity`` is the burst (how much may be spent at once after an idle period) and
    ``refill_per_second`` is the sustained rate. The level is recomputed lazily from the
    elapsed time on every access, so an idle bucket costs nothing and there is no timer.

    The level may go negative: :meth:`charge` debits unconditionally, which is how a
    completion whose length was unknown at admission is still paid for. :meth:`retry_after`
    accounts for the deficit.
    """

    __slots__ = ("_level", "_updated", "capacity", "refill_per_second")

    def __init__(
        self,
        capacity: float,
        refill_per_second: float,
        *,
        level: float | None = None,
        now: float | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if refill_per_second <= 0:
            raise ValueError(f"refill_per_second must be positive, got {refill_per_second}")
        self.capacity = float(capacity)
        self.refill_per_second = float(refill_per_second)
        self._level = self.capacity if level is None else float(level)
        self._updated = time.monotonic() if now is None else float(now)

    @classmethod
    def per_minute(
        cls,
        limit: float,
        *,
        burst: float | None = None,
        now: float | None = None,
    ) -> TokenBucket:
        """Build a bucket from a per-minute budget.

        The default burst is the full minute's budget, which is what "600 requests per
        minute" means to the person who wrote it in the config file.
        """
        return cls(burst if burst is not None else limit, limit / _SECONDS_PER_MINUTE, now=now)

    def _refill(self, now: float) -> None:
        elapsed = now - self._updated
        if elapsed > 0:
            self._level = min(self.capacity, self._level + elapsed * self.refill_per_second)
            self._updated = now

    def level(self, *, now: float | None = None) -> float:
        """Current allowance, refilled to ``now``. May be negative after an over-charge."""
        self._refill(time.monotonic() if now is None else now)
        return self._level

    def try_acquire(self, amount: float = 1.0, *, now: float | None = None) -> bool:
        """Spend ``amount`` if it is available; return whether it was spent."""
        if amount < 0:
            raise ValueError(f"amount must not be negative, got {amount}")
        self._refill(time.monotonic() if now is None else now)
        if self._level < amount:
            return False
        self._level -= amount
        return True

    def charge(self, amount: float, *, now: float | None = None) -> float:
        """Spend ``amount`` whether or not it is available; return the resulting level."""
        if amount < 0:
            raise ValueError(f"amount must not be negative, got {amount}")
        self._refill(time.monotonic() if now is None else now)
        self._level -= amount
        return self._level

    def refund(self, amount: float, *, now: float | None = None) -> float:
        """Return ``amount`` to the bucket, capped at capacity; return the new level."""
        if amount < 0:
            raise ValueError(f"amount must not be negative, got {amount}")
        self._refill(time.monotonic() if now is None else now)
        self._level = min(self.capacity, self._level + amount)
        return self._level

    def retry_after(self, amount: float = 1.0, *, now: float | None = None) -> float:
        """Seconds until ``amount`` could be spent, assuming no other traffic."""
        deficit = amount - self.level(now=now)
        if deficit <= 0:
            return 0.0
        return deficit / self.refill_per_second

    def reset(self, *, now: float | None = None) -> None:
        """Refill to capacity, as if the tenant had been idle forever."""
        self._level = self.capacity
        self._updated = time.monotonic() if now is None else now

    def __repr__(self) -> str:
        return (
            f"TokenBucket(capacity={self.capacity:g}, "
            f"refill_per_second={self.refill_per_second:g}, level={self._level:g})"
        )


@dataclass(slots=True)
class LimiterSnapshot:
    """Debug view of a tenant's quota state, exposed by the config-check command."""

    tenant_id: str
    rpm_level: float | None = None
    tpm_level: float | None = None
    in_flight: int = 0
    max_concurrency: int | None = None
    admitted: int = 0
    rejected: int = 0
    tokens_charged: int = 0

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Plain mapping, for JSON output and log lines."""
        return {
            "tenant_id": self.tenant_id,
            "rpm_level": self.rpm_level,
            "tpm_level": self.tpm_level,
            "in_flight": self.in_flight,
            "max_concurrency": self.max_concurrency,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "tokens_charged": self.tokens_charged,
        }


@dataclass(slots=True)
class TenantLimiter:
    """The live quota state of one tenant.

    Created lazily by :class:`LimiterRegistry` on a tenant's first request, so a directory
    with a thousand configured tenants costs nothing until they call.
    """

    tenant: Tenant
    clock: Callable[[], float] = time.monotonic
    concurrency_retry_after_s: float = DEFAULT_CONCURRENCY_RETRY_AFTER_S
    _rpm: TokenBucket | None = field(default=None, init=False, repr=False)
    _tpm: TokenBucket | None = field(default=None, init=False, repr=False)
    _in_flight: int = field(default=0, init=False)
    admitted: int = field(default=0, init=False)
    rejected: int = field(default=0, init=False)
    tokens_charged: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        if self.tenant.rpm is not None:
            self._rpm = TokenBucket.per_minute(self.tenant.rpm, now=now)
        if self.tenant.tpm is not None:
            self._tpm = TokenBucket.per_minute(self.tenant.tpm, now=now)

    @property
    def tenant_id(self) -> str:
        """The tenant this limiter belongs to."""
        return self.tenant.tenant_id

    @property
    def in_flight(self) -> int:
        """Requests currently holding a concurrency slot."""
        return self._in_flight

    def check_request(self, *, prompt_tokens: int = 0, now: float | None = None) -> None:
        """Admit one request, or raise :class:`RateLimitExceeded`.

        Checks rpm first and tpm second, and only debits once *both* pass: charging rpm and
        then failing tpm would burn the tenant's request budget on a request that never ran.
        """
        stamp = self.clock() if now is None else now
        if self._rpm is not None and self._rpm.level(now=stamp) < 1.0:
            self.rejected += 1
            raise RateLimitExceeded(
                self.tenant_id,
                "rpm",
                retry_after_s=self._rpm.retry_after(1.0, now=stamp),
                limit_value=self.tenant.rpm,
            )
        if self._tpm is not None and prompt_tokens > 0:
            level = self._tpm.level(now=stamp)
            if level < prompt_tokens:
                self.rejected += 1
                raise RateLimitExceeded(
                    self.tenant_id,
                    "tpm",
                    retry_after_s=self._tpm.retry_after(prompt_tokens, now=stamp),
                    limit_value=self.tenant.tpm,
                )
        if self._rpm is not None:
            self._rpm.charge(1.0, now=stamp)
        if self._tpm is not None and prompt_tokens > 0:
            self._tpm.charge(prompt_tokens, now=stamp)
            self.tokens_charged += prompt_tokens
        self.admitted += 1

    def charge_tokens(self, tokens: int, *, now: float | None = None) -> None:
        """Debit tokens produced after admission (the completion).

        Unconditional: the tokens have already been generated. Overdrawing pushes the next
        ``Retry-After`` out, which is the intended back-pressure.
        """
        if tokens <= 0 or self._tpm is None:
            return
        self._tpm.charge(tokens, now=self.clock() if now is None else now)
        self.tokens_charged += tokens

    def try_acquire_slot(self) -> bool:
        """Take a concurrency slot if one is free."""
        limit = self.tenant.max_concurrency
        if limit is not None and self._in_flight >= limit:
            return False
        self._in_flight += 1
        return True

    def release_slot(self) -> None:
        """Give a concurrency slot back; never drops below zero."""
        if self._in_flight > 0:
            self._in_flight -= 1

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Hold a concurrency slot for the duration of a request.

        Raises :class:`RateLimitExceeded` immediately when the tenant is at its limit -- the
        caller must not be left waiting, because a 429 now is more useful to a client than a
        connection that eventually succeeds.
        """
        if not self.try_acquire_slot():
            self.rejected += 1
            raise RateLimitExceeded(
                self.tenant_id,
                "concurrency",
                retry_after_s=self.concurrency_retry_after_s,
                limit_value=self.tenant.max_concurrency,
            )
        try:
            yield
        finally:
            self.release_slot()

    def snapshot(self, *, now: float | None = None) -> LimiterSnapshot:
        """Current quota state, for diagnostics."""
        stamp = self.clock() if now is None else now
        return LimiterSnapshot(
            tenant_id=self.tenant_id,
            rpm_level=None if self._rpm is None else self._rpm.level(now=stamp),
            tpm_level=None if self._tpm is None else self._tpm.level(now=stamp),
            in_flight=self._in_flight,
            max_concurrency=self.tenant.max_concurrency,
            admitted=self.admitted,
            rejected=self.rejected,
            tokens_charged=self.tokens_charged,
        )


class LimiterRegistry:
    """Lazily-created :class:`TenantLimiter` per tenant, keyed by id.

    The gateway holds exactly one of these for its lifetime; it is where a tenant's quota
    *state* lives, as opposed to the tenant's quota *configuration* in
    :class:`~turboserve.gateway.tenants.TenantRegistry`.
    """

    __slots__ = ("_clock", "_concurrency_retry_after_s", "_limiters", "_tenants")

    def __init__(
        self,
        tenants: TenantRegistry,
        *,
        clock: Callable[[], float] = time.monotonic,
        concurrency_retry_after_s: float = DEFAULT_CONCURRENCY_RETRY_AFTER_S,
    ) -> None:
        self._tenants = tenants
        self._clock = clock
        self._concurrency_retry_after_s = concurrency_retry_after_s
        self._limiters: dict[str, TenantLimiter] = {}

    def for_tenant(self, tenant: Tenant) -> TenantLimiter:
        """Limiter for ``tenant``, created on first use."""
        limiter = self._limiters.get(tenant.tenant_id)
        if limiter is None:
            limiter = TenantLimiter(
                tenant,
                clock=self._clock,
                concurrency_retry_after_s=self._concurrency_retry_after_s,
            )
            self._limiters[tenant.tenant_id] = limiter
        return limiter

    def for_tenant_id(self, tenant_id: str) -> TenantLimiter | None:
        """Limiter for a tenant id, or ``None`` when the id is not in the directory."""
        limiter = self._limiters.get(tenant_id)
        if limiter is not None:
            return limiter
        tenant = self._tenants.get(tenant_id)
        return None if tenant is None else self.for_tenant(tenant)

    def snapshot(self) -> dict[str, dict[str, float | int | str | None]]:
        """Quota state of every tenant that has sent at least one request."""
        return {
            tenant_id: limiter.snapshot().to_dict()
            for tenant_id, limiter in sorted(self._limiters.items())
        }

    def reset(self) -> None:
        """Drop all quota state. Used by tests and by a configuration reload."""
        self._limiters.clear()

    def __len__(self) -> int:
        return len(self._limiters)
