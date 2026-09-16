"""Choosing a backend for a request, and what to do when that choice fails.

A model name does not identify a process: it identifies a *pool* of replicas, split into a
``stable`` lane and a ``canary`` lane. The router's job is to pick one replica per request
such that four things hold at once.

**Traffic follows the canary weight.** The canary controller owns the weight and moves it
through its steps; the router only reads :attr:`CanaryWeightSource.canary_weight` per
request. Reading it per request rather than caching it is what makes a rollback take effect
on the next request instead of on the next connection.

**Unhealthy replicas are skipped, but not probed on the request path.** Each replica's health
is cached for ``health_ttl_s``; a stale entry is refreshed before a selection, a fresh one is
believed. A failure observed while serving marks the replica unhealthy immediately -- the
strongest health signal available is a request that just failed on it.

**A request may be retried only before its first token.** This is the rule the whole module
exists to enforce. Before the first event nothing has been promised to the client and a
retryable error means another replica may serve it. After the first event the client holds
part of a completion; re-running it elsewhere would duplicate or contradict what they have,
so the failure is converted into a terminating error event and the stream ends. The gateway
never sends a byte to the client before pulling the first event out of this generator, so the
two notions of "first byte" coincide.

**One tenant cannot occupy the whole pool.** The tenant's concurrency slot is taken before
the first event and released when the stream ends, so the refusal is a 429 the client can act
on, delivered before any response body exists.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from turboserve.engine.core.types import FinishReason
from turboserve.gateway.backends import get_backend_cls
from turboserve.gateway.backends.protocol import (
    Backend,
    BackendError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
    NonRetryableBackendError,
    TokenEvent,
)
from turboserve.gateway.usage import ModelPrice, PriceTable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from turboserve.gateway.limits import LimiterRegistry
    from turboserve.gateway.metrics import GatewayMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "BackendConfig",
    "BackendEntry",
    "CanaryWeightSource",
    "Lane",
    "ModelPoolConfig",
    "ModelPool",
    "ModelsFile",
    "RoutedEvent",
    "Router",
    "RouterConfigError",
    "build_router",
]

#: Which side of a progressive rollout a replica is on.
#:
#: Deliberately re-declared rather than imported from :mod:`turboserve.canary.controller`,
#: which declares the identical alias. The dependency only ever runs the other way -- the
#: canary reads the gateway's metrics, the gateway knows the controller only through the
#: structural :class:`CanaryWeightSource` protocol below -- and importing the canary package
#: here to share a two-element ``Literal`` would invert that for no benefit. Both aliases
#: are ``Literal["stable", "canary"]``, so values pass between the modules unchanged and a
#: divergence is a type error at the first call site.
Lane = Literal["stable", "canary"]

_LANES: tuple[Lane, ...] = ("stable", "canary")

#: Percent, because the canary controller's steps are percentages ([1, 5, 25, 50, 100]).
_FULL_WEIGHT = 100.0


@asynccontextmanager
async def _closing(stream: AsyncIterator[TokenEvent]) -> AsyncIterator[AsyncIterator[TokenEvent]]:
    """Close a backend's event stream on the way out, however the block was left.

    :func:`contextlib.aclosing` would do, but it insists on an object with ``aclose`` --
    every backend in this repository is an async generator function and has one, while the
    :class:`~turboserve.gateway.backends.protocol.Backend` protocol only promises an async
    *iterator*. Closing when possible keeps a third-party iterator usable instead of turning
    a served request into an ``AttributeError`` at teardown.
    """
    try:
        yield stream
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


class RouterConfigError(ValueError):
    """``models.yaml`` is missing, unreadable, or does not describe model pools."""


@runtime_checkable
class CanaryWeightSource(Protocol):
    """The slice of the canary controller the router depends on.

    Declared here as a protocol rather than imported from :mod:`turboserve.canary` so that
    the gateway does not depend on the progressive-delivery machinery: a deployment with no
    canary passes ``None`` and the router sends everything to ``stable``.
    :class:`turboserve.canary.controller.CanaryController` satisfies it structurally, and
    ``tests/unit/test_gateway_router.py`` asserts that it still does -- a protocol nothing
    checks is a protocol that drifts.

    Two details exist so that the real controller type-checks against this, not just runs
    against it: ``canary_weight`` is declared as a read-only property (the controller
    computes it from its step index, and a bare annotation would demand a settable
    attribute), and ``observe``'s ``lane`` is typed :data:`Lane` rather than ``str``
    (a parameter may be widened by an implementation, never narrowed, and the controller
    narrows it to the same two-element ``Literal``).
    """

    @property
    def canary_weight(self) -> float:
        """Share of traffic the canary lane should take, in percent (0 = none, 100 = all)."""
        ...

    def observe(
        self,
        lane: Lane,
        ok: bool,
        ttft_ms: float | None,
        e2e_ms: float | None,
        now: float,
    ) -> None:
        """Feed one finished request into the controller's sliding window."""
        ...


# --------------------------------------------------------------------------------------
# Configuration (configs/models.yaml)
# --------------------------------------------------------------------------------------


class BackendConfig(BaseModel):
    """One replica in a model pool, as written in ``models.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    """Identifier used in metric labels and log lines; unique within the pool."""

    backend: str = "openai"
    """Registered backend type: ``openai``, ``mock`` or ``local``."""

    lane: Lane = "stable"
    weight: float = Field(default=1.0, gt=0.0)
    """Relative share *within* its lane; the lane split comes from the canary weight."""

    options: dict[str, Any] = Field(default_factory=dict)
    """Constructor keyword arguments for the backend class (``base_url``, ``models``, ...)."""


class ModelPoolConfig(BaseModel):
    """Every replica that serves one model name, plus its configured price."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    backends: list[BackendConfig] = Field(min_length=1)
    price: ModelPrice | None = None
    """Configured list price used only to attribute spend between tenants."""


class ModelsFile(BaseModel):
    """The parsed ``configs/models.yaml``."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    models: list[ModelPoolConfig] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> ModelsFile:
        """Load and validate a models file."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RouterConfigError(f"cannot read models file {path}: {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RouterConfigError(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise RouterConfigError(f"{path} must contain a YAML mapping")
        try:
            return cls.model_validate(data)
        except ValueError as exc:
            raise RouterConfigError(f"invalid models file {path}: {exc}") from exc

    def price_table(self) -> PriceTable:
        """The price table implied by this file."""
        return PriceTable({pool.name: pool.price for pool in self.models if pool.price is not None})

    def model_names(self) -> list[str]:
        """Pool names, in file order."""
        return [pool.name for pool in self.models]


# --------------------------------------------------------------------------------------
# Runtime pool
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class BackendEntry:
    """A replica plus the router's opinion of it."""

    backend: Backend
    lane: Lane = "stable"
    weight: float = 1.0
    healthy: bool = True
    checked_at: float = float("-inf")
    """Monotonic timestamp of the last health probe; ``-inf`` means never probed."""

    consecutive_failures: int = 0
    served: int = 0

    @property
    def name(self) -> str:
        """The backend's own name, used as the metric label and the retry exclusion key."""
        return self.backend.name

    def is_fresh(self, now: float, ttl: float) -> bool:
        """Whether the cached health verdict is still within its time to live."""
        return (now - self.checked_at) < ttl


class ModelPool:
    """The replicas serving one model name."""

    __slots__ = ("entries", "model")

    def __init__(self, model: str, entries: Iterable[BackendEntry] = ()) -> None:
        self.model = model
        self.entries: list[BackendEntry] = list(entries)

    def add(self, entry: BackendEntry) -> BackendEntry:
        """Append a replica, refusing a duplicate name within the pool."""
        if any(existing.name == entry.name for existing in self.entries):
            raise RouterConfigError(
                f"backend {entry.name!r} is already in the pool for model {self.model!r}"
            )
        self.entries.append(entry)
        return entry

    def lane(self, lane: Lane) -> list[BackendEntry]:
        """Replicas on one lane."""
        return [entry for entry in self.entries if entry.lane == lane]

    def candidates(
        self, lane: Lane, *, exclude: frozenset[str] = frozenset()
    ) -> list[BackendEntry]:
        """Healthy replicas on ``lane`` that have not already been tried."""
        return [
            entry
            for entry in self.entries
            if entry.lane == lane and entry.healthy and entry.name not in exclude
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"ModelPool({self.model!r}, {len(self.entries)} backends)"


@dataclass(slots=True)
class RoutedEvent:
    """A backend event plus where it came from.

    The router is an async generator, so it cannot return the chosen replica before it
    yields; attaching the provenance to every event is what lets the route layer label its
    metrics and its usage record without a second channel.
    """

    event: TokenEvent
    backend: str
    lane: Lane
    attempt: int = 1

    @property
    def finished(self) -> bool:
        """Whether this event terminates the stream."""
        return self.event.finished

    @property
    def is_error(self) -> bool:
        """Whether this event reports a failure."""
        return self.event.is_error


class Router:
    """Model name to replica, with health, weighting, retries and concurrency gating."""

    __slots__ = (
        "_canary",
        "_clock",
        "_health_ttl_s",
        "_limiters",
        "_max_attempts",
        "_metrics",
        "_pools",
        "_rng",
    )

    def __init__(
        self,
        pools: Mapping[str, ModelPool] | None = None,
        *,
        canary: CanaryWeightSource | None = None,
        limiters: LimiterRegistry | None = None,
        metrics: GatewayMetrics | None = None,
        health_ttl_s: float = 5.0,
        max_attempts: int = 3,
        rng: random.Random | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self._pools: dict[str, ModelPool] = dict(pools or {})
        self._canary = canary
        self._limiters = limiters
        self._metrics = metrics
        self._health_ttl_s = health_ttl_s
        self._max_attempts = max_attempts
        self._rng = rng if rng is not None else random.Random()
        self._clock = clock

    # -- composition --------------------------------------------------------------------

    def add_backend(
        self,
        model: str,
        backend: Backend,
        *,
        lane: Lane = "stable",
        weight: float = 1.0,
    ) -> BackendEntry:
        """Register ``backend`` as a replica of ``model``."""
        if weight <= 0.0:
            raise ValueError(f"weight must be positive, got {weight}")
        pool = self._pools.get(model)
        if pool is None:
            pool = ModelPool(model)
            self._pools[model] = pool
        return pool.add(BackendEntry(backend=backend, lane=lane, weight=weight))

    def attach(
        self,
        *,
        limiters: LimiterRegistry | None = None,
        metrics: GatewayMetrics | None = None,
    ) -> None:
        """Give the router the app-level collaborators it was not constructed with.

        ``create_app`` builds the limiter registry and the metrics registry after the router
        (a caller may have supplied a ready-made router), so this closes the loop without
        making the router's constructor depend on the app's construction order. Existing
        collaborators are kept: a router handed in with its own registries keeps them.
        """
        if limiters is not None and self._limiters is None:
            self._limiters = limiters
        if metrics is not None and self._metrics is None:
            self._metrics = metrics

    def add_pool(self, pool: ModelPool) -> None:
        """Install a whole pool, replacing any pool already registered for that model."""
        self._pools[pool.model] = pool

    def pool(self, model: str) -> ModelPool:
        """The pool serving ``model``; raises :class:`ModelNotFoundError` when there is none."""
        pool = self._pools.get(model)
        if pool is None:
            raise ModelNotFoundError(f"no backend serves model {model!r}")
        return pool

    def has_model(self, model: str) -> bool:
        """Whether any replica serves ``model``."""
        return model in self._pools

    def models(self) -> list[str]:
        """Served model names, sorted."""
        return sorted(self._pools)

    @property
    def canary(self) -> CanaryWeightSource | None:
        """The canary controller, if one was attached."""
        return self._canary

    def set_canary(self, canary: CanaryWeightSource | None) -> None:
        """Attach or detach a canary controller at runtime."""
        self._canary = canary

    # -- lane and replica selection -----------------------------------------------------

    def canary_fraction(self) -> float:
        """The canary lane's share as a fraction in ``[0, 1]``.

        The controller publishes a percentage because its configured steps are percentages;
        clamping here rather than trusting it keeps a controller bug from sending every
        request to a lane that may not exist.
        """
        if self._canary is None:
            return 0.0
        weight = float(getattr(self._canary, "canary_weight", 0.0))
        return min(1.0, max(0.0, weight / _FULL_WEIGHT))

    def choose_lane(self, pool: ModelPool, *, exclude: frozenset[str] = frozenset()) -> Lane:
        """Draw a lane for one request.

        A lane with no healthy replica is never chosen, so a canary whose only pod is down
        does not black-hole its share of the traffic -- it simply stops receiving any until
        the pod is healthy again.
        """
        if not pool.candidates("canary", exclude=exclude):
            return "stable"
        if not pool.candidates("stable", exclude=exclude):
            return "canary"
        return "canary" if self._rng.random() < self.canary_fraction() else "stable"

    async def refresh_health(self, entry: BackendEntry, *, now: float | None = None) -> bool:
        """Probe a replica if its cached verdict has expired; return the verdict."""
        stamp = self._clock() if now is None else now
        if entry.is_fresh(stamp, self._health_ttl_s):
            return entry.healthy
        try:
            healthy = bool(await entry.backend.health())
        except Exception as exc:  # noqa: BLE001 - a raising probe is an unhealthy backend
            logger.warning("health probe of %s raised: %s", entry.name, exc)
            healthy = False
        entry.healthy = healthy
        entry.checked_at = stamp
        entry.consecutive_failures = 0 if healthy else entry.consecutive_failures
        return healthy

    async def _refresh_pool(self, pool: ModelPool) -> None:
        """Refresh every stale replica of a pool, then publish the verdicts as metrics."""
        now = self._clock()
        for entry in pool.entries:
            if not entry.is_fresh(now, self._health_ttl_s):
                await self.refresh_health(entry, now=now)
            if self._metrics is not None:
                self._metrics.set_backend_up(
                    backend=entry.name, model=pool.model, lane=entry.lane, up=entry.healthy
                )

    def _weighted_choice(self, entries: Sequence[BackendEntry]) -> BackendEntry:
        """Pick one replica in proportion to its configured weight."""
        if len(entries) == 1:
            return entries[0]
        total = sum(entry.weight for entry in entries)
        draw = self._rng.random() * total
        upto = 0.0
        for entry in entries:
            upto += entry.weight
            if draw <= upto:
                return entry
        return entries[-1]  # pragma: no cover - only reachable through float rounding

    async def select(
        self,
        model: str,
        *,
        exclude: frozenset[str] = frozenset(),
        lane: Lane | None = None,
    ) -> BackendEntry:
        """Choose the replica to serve the next attempt of a request.

        A lane forced by the caller is honoured only if it has a usable replica; otherwise,
        and whenever the drawn lane is empty, the other lane is used. Serving from the wrong
        lane is better than failing, and the lane actually used is reported on every event so
        the canary's statistics stay attributable.
        """
        pool = self.pool(model)
        await self._refresh_pool(pool)
        chosen_lane = lane if lane is not None else self.choose_lane(pool, exclude=exclude)
        candidates = pool.candidates(chosen_lane, exclude=exclude)
        if not candidates:
            for other in _LANES:
                if other == chosen_lane:
                    continue
                candidates = pool.candidates(other, exclude=exclude)
                if candidates:
                    chosen_lane = other
                    break
        if not candidates:
            raise BackendUnavailableError(
                f"no healthy backend for model {model!r}"
                + (f" (already tried: {', '.join(sorted(exclude))})" if exclude else "")
            )
        entry = self._weighted_choice(candidates)
        if self._metrics is not None:
            self._metrics.set_canary_weight(
                model=model, weight=self.canary_fraction() * _FULL_WEIGHT
            )
        return entry

    # -- the request path ---------------------------------------------------------------

    def _mark_failure(self, entry: BackendEntry, exc: BackendError) -> None:
        """Record a failure observed while serving and take the replica out if warranted.

        Only *transport* failures condemn a replica. A rejected request (bad sampling
        parameters, unknown adapter) says nothing about the replica's health, and marking it
        down would let one malformed client empty a pool.
        """
        if isinstance(exc, NonRetryableBackendError):
            return
        entry.consecutive_failures += 1
        entry.healthy = False
        entry.checked_at = self._clock()
        logger.warning(
            "backend %s marked unhealthy after %d consecutive failures: %s",
            entry.name,
            entry.consecutive_failures,
            exc,
        )

    def _observe(
        self,
        entry: BackendEntry,
        *,
        ok: bool,
        ttft_s: float | None,
        e2e_s: float | None,
    ) -> None:
        """Feed one *attempt* to the canary controller, if there is one.

        Every attempt is reported, including one that is retried away before the client sees
        anything, so the canary's error rate describes the replica rather than the router's
        ability to hide it.
        """
        if self._canary is None:
            return
        try:
            self._canary.observe(
                entry.lane,
                ok,
                None if ttft_s is None else ttft_s * 1000.0,
                None if e2e_s is None else e2e_s * 1000.0,
                self._clock(),
            )
        except Exception as exc:  # noqa: BLE001 - the controller must not break serving
            logger.warning("canary controller rejected an observation: %s", exc)

    async def generate(
        self,
        req: GenerateRequest,
        *,
        lane: Lane | None = None,
    ) -> AsyncGenerator[RoutedEvent, None]:
        """Stream ``req`` from a chosen replica, retrying only before the first event.

        Raises before yielding anything (:class:`RateLimitExceeded` from the tenant's
        concurrency gate, or a :class:`BackendError` once the attempts are exhausted), so the
        caller can still answer with an HTTP status. Once it has yielded, it only ever ends
        with a terminating event -- an error becomes
        :meth:`TokenEvent.failure <turboserve.gateway.backends.protocol.TokenEvent.failure>`.
        """
        # `is not None`, not truthiness: LimiterRegistry defines __len__ and an empty
        # one (no tenant has called yet) would otherwise read as absent.
        limiter = (
            self._limiters.for_tenant_id(req.tenant_id) if self._limiters is not None else None
        )
        gate = limiter.slot() if limiter is not None else nullcontext()
        started = False
        # A backend may report a mid-stream failure by *yielding* a terminating error event
        # rather than raising -- that is the documented way to fail after the first byte, and
        # it is what MockBackend's drop injection does. Such a stream ends normally, so
        # without this flag it would be fed to the canary controller as a success and a
        # gate watching in-process observations would never see the drops.
        stream_error = False
        t_start = self._clock()
        t_first: float | None = None
        entry: BackendEntry | None = None
        with gate:
            tried: set[str] = set()
            last_error: BackendError | None = None
            for attempt in range(1, self._max_attempts + 1):
                entry = await self.select(req.model, exclude=frozenset(tried), lane=lane)
                tried.add(entry.name)
                entry.served += 1
                try:
                    async with _closing(entry.backend.generate(req)) as stream:
                        async for event in stream:
                            if not started:
                                started = True
                                t_first = self._clock()
                            stream_error = stream_error or event.is_error
                            yield RoutedEvent(
                                event=event, backend=entry.name, lane=entry.lane, attempt=attempt
                            )
                except BackendError as exc:
                    self._mark_failure(entry, exc)
                    # Observed even when the attempt is about to be retried away. The gate
                    # measures *attempts*, not client-visible outcomes: a canary replica that
                    # fails every request before the first byte would otherwise contribute no
                    # samples at all, and the controller would sit on HOLD until its stall
                    # timeout instead of rolling back on a 100% error rate.
                    self._observe(entry, ok=False, ttft_s=None, e2e_s=None)
                    if started:
                        yield RoutedEvent(
                            event=TokenEvent.failure(
                                req.request_id, str(exc), finish_reason=FinishReason.ABORT
                            ),
                            backend=entry.name,
                            lane=entry.lane,
                            attempt=attempt,
                        )
                        return
                    last_error = exc
                    if not exc.retryable:
                        raise
                    logger.info(
                        "retrying %s after a retryable failure on %s (attempt %d/%d): %s",
                        req.request_id,
                        entry.name,
                        attempt,
                        self._max_attempts,
                        exc,
                    )
                    continue
                except Exception as exc:  # noqa: BLE001 - a backend bug must not 500 the gateway
                    logger.exception("backend %s raised an unexpected error", entry.name)
                    wrapped = NonRetryableBackendError(
                        f"{entry.name} failed: {exc}", backend=entry.name
                    )
                    self._mark_failure(entry, wrapped)
                    # Observed before the re-raise for the same reason as above. A rejected
                    # request does not condemn the replica (see _mark_failure) but it is
                    # still a failed request on that lane, and a build that refuses what the
                    # stable lane accepts is what the gate is for.
                    self._observe(entry, ok=False, ttft_s=None, e2e_s=None)
                    if not started:
                        raise wrapped from exc
                    yield RoutedEvent(
                        event=TokenEvent.failure(
                            req.request_id, str(wrapped), finish_reason=FinishReason.ABORT
                        ),
                        backend=entry.name,
                        lane=entry.lane,
                        attempt=attempt,
                    )
                    return
                else:
                    t_end = self._clock()
                    if stream_error:
                        # The transport was fine, so the replica is not ejected -- but the
                        # request did not succeed, and the gate must be told so.
                        self._observe(entry, ok=False, ttft_s=None, e2e_s=None)
                        return
                    entry.consecutive_failures = 0
                    self._observe(
                        entry,
                        ok=True,
                        ttft_s=None if t_first is None else t_first - t_start,
                        e2e_s=t_end - t_start,
                    )
                    return
            raise last_error or BackendUnavailableError(
                f"no backend served model {req.model!r} after {self._max_attempts} attempts"
            )

    # -- fleet operations ---------------------------------------------------------------

    async def health_report(self) -> dict[str, dict[str, bool]]:
        """Probe every replica and return ``{model: {backend: healthy}}``."""
        report: dict[str, dict[str, bool]] = {}
        for model, pool in sorted(self._pools.items()):
            await self._refresh_pool(pool)
            report[model] = {entry.name: entry.healthy for entry in pool.entries}
        return report

    async def ready(self) -> bool:
        """Whether every served model has at least one healthy replica.

        This is the readiness condition: a gateway that can serve none of its models should
        be taken out of the Service, while one that can serve all of them should not be, even
        if individual replicas are down.
        """
        if not self._pools:
            return False
        report = await self.health_report()
        return all(any(healthy.values()) for healthy in report.values())

    def backends(self) -> list[Backend]:
        """Every distinct backend object in the router, in pool order."""
        seen: dict[int, Backend] = {}
        for pool in self._pools.values():
            for entry in pool.entries:
                seen.setdefault(id(entry.backend), entry.backend)
        return list(seen.values())

    def queue_depth(self) -> int:
        """Requests currently held by tenant concurrency slots, across all tenants.

        Published as ``turboserve_gateway_queue_depth``; the Kubernetes HPA scales on it
        because it rises before latency does.
        """
        if self._limiters is None:
            return 0
        return sum(int(entry.get("in_flight") or 0) for entry in self._limiters.snapshot().values())

    async def close(self) -> None:
        """Close every backend once. Idempotent as far as the backends are."""
        for backend in self.backends():
            try:
                await backend.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must close the rest
                logger.warning("closing backend %s raised: %s", getattr(backend, "name", "?"), exc)

    def __repr__(self) -> str:
        return f"Router(models={self.models()})"


def build_router(
    config: ModelsFile,
    *,
    canary: CanaryWeightSource | None = None,
    limiters: LimiterRegistry | None = None,
    metrics: GatewayMetrics | None = None,
    **kwargs: Any,
) -> Router:
    """Instantiate every backend named in a models file and wire it into a router.

    Backends are constructed from the registry by name, so a deployment adds a backend type
    by installing a module that registers itself -- no change here. A misconfigured entry
    fails at startup with the pool and backend named, rather than on the first request that
    happens to be routed to it.
    """
    router = Router(canary=canary, limiters=limiters, metrics=metrics, **kwargs)
    for pool_config in config.models:
        for backend_config in pool_config.backends:
            try:
                cls = get_backend_cls(backend_config.backend)
            except KeyError as exc:
                raise RouterConfigError(
                    f"model {pool_config.name!r} backend {backend_config.name!r}: {exc}"
                ) from exc
            try:
                backend = cls(name=backend_config.name, **backend_config.options)  # type: ignore[call-arg]
            except (TypeError, ValueError) as exc:
                raise RouterConfigError(
                    f"model {pool_config.name!r} backend {backend_config.name!r} "
                    f"({backend_config.backend}) could not be constructed: {exc}"
                ) from exc
            router.add_backend(
                pool_config.name,
                backend,
                lane=backend_config.lane,
                weight=backend_config.weight,
            )
    logger.info("router serving %d model(s): %s", len(config.models), ", ".join(router.models()))
    return router
