"""Replicas that can be broken on purpose, in process or as a real server.

A chaos experiment is a statement about the *gateway*: when a replica dies, does the router
notice, retry what it is allowed to retry, and stop sending traffic to a corpse? To make
that statement the replicas have to be breakable, and they have to be breakable the way real
ones break. This module provides two flavours, deliberately behind one interface
(:class:`ChaosWorker`) so the harness does not know which it is driving:

:class:`InProcessWorker`
    A :class:`~turboserve.gateway.backends.mock.MockBackend` wrapped in
    :class:`FaultingBackend`. No sockets, no subprocesses, microsecond overhead -- so a test
    can run a five-second experiment with kills in it and still be a unit test. Its "kill"
    is a flag: new requests are refused and in-flight streams abort at their next token
    boundary.

:class:`SubprocessWorker`
    The same mock, but served over HTTP by :func:`serve_worker` in a child process the
    harness can send ``SIGKILL``. The gateway then sees what it would see in production: a
    refused connection, a half-written response, a health probe that times out. This is the
    flavour ``turboserve chaos run`` uses by default, because a fault that never crosses a
    socket cannot prove that the HTTP client's error handling is right.

The server is the real gateway application with authentication off (see
:func:`~turboserve.gateway.backends.mock.build_mock_app` for why that matters): faults
therefore travel the production auth, routing, accounting and SSE code rather than a
simplified copy of it. Faults are injected in two places, matching the two kinds of failure
they model:

* per-request latency and errors live in :class:`FaultingBackend`, below the gateway, where
  the request id is known and the draw can be made per request;
* a partition is an HTTP middleware that answers *every* path with 503, health probes
  included, because a partitioned replica is not selectively unreachable.

The ``/chaos`` control endpoints stay reachable during a partition. They are the operator's
out-of-band channel -- the harness uses them to lift the partition -- and are never part of
the data path being measured.
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from turboserve.chaos.faults import SteadyFaults
from turboserve.gateway.backends.mock import MockBackend, MockConfig
from turboserve.gateway.backends.protocol import (
    Backend,
    BackendUnavailableError,
    GenerateRequest,
    StreamInterruptedError,
    TokenEvent,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator, Mapping, Sequence

    from fastapi import FastAPI

logger = logging.getLogger(__name__)

__all__ = [
    "CHAOS_PREFIX",
    "ChaosWorker",
    "FaultingBackend",
    "InProcessWorker",
    "SubprocessWorker",
    "WorkerFaults",
    "WorkerSpec",
    "WorkerStats",
    "build_worker_app",
    "build_workers",
    "serve_worker",
]

#: Path prefix of the control plane. Excluded from the partition middleware.
CHAOS_PREFIX = "/chaos"

_HTTP_SERVICE_UNAVAILABLE = 503


class WorkerFaults(BaseModel):
    """The per-request faults a single replica is currently applying.

    Mutable with validation on assignment, because that is exactly how it is used: the
    harness assigns to a field of a live worker, or POSTs a new value to ``/chaos/faults``,
    and the next request feels it. A frozen model would force a replica to be rebuilt to
    change a probability, and rebuilding it would reset the very statistics being collected.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    latency_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    partitioned: bool = False

    @classmethod
    def from_steady(cls, steady: SteadyFaults, *, partitioned: bool = False) -> WorkerFaults:
        """Build from a schedule's steady-state faults."""
        return cls(
            latency_probability=steady.latency_probability,
            latency_ms=steady.latency_ms,
            error_probability=steady.error_probability,
            partitioned=partitioned,
        )

    @property
    def is_empty(self) -> bool:
        """Whether this replica is currently behaving perfectly."""
        return (
            not self.partitioned
            and self.error_probability <= 0.0
            and (self.latency_probability <= 0.0 or self.latency_ms <= 0.0)
        )


class WorkerStats(BaseModel):
    """What one replica did, read back at the end of a run (and over ``/chaos/state``)."""

    model_config = ConfigDict(extra="forbid")

    requests: int = 0
    """Requests admitted, i.e. not refused by a kill, a drain or a partition."""

    delayed: int = 0
    injected_errors: int = 0
    aborted: int = 0
    """Streams cut because the replica was killed after it had already sent tokens."""

    refused: int = 0
    """Requests rejected before any token, because the replica was down or draining."""


class FaultingBackend:
    """A :class:`~turboserve.gateway.backends.protocol.Backend` that can be broken.

    Wraps another backend (in practice the mock) and adds three things the gateway must cope
    with, each mapped onto the error class that tells the router what it is allowed to do:

    * **down or draining** -- :class:`BackendUnavailableError`, raised before the first
      token, which is retryable: the router may send the request to another replica and the
      client never learns that anything happened;
    * **killed mid-stream** -- :class:`StreamInterruptedError`, raised after tokens have been
      yielded, which is *not* retryable: the client holds part of a completion and a second
      attempt would duplicate it. This is the one failure a perfect gateway still reports,
      and the reason an ungraceful kill has a floor on its error rate;
    * **slow** -- a sleep before the first token, which is invisible to correctness and
      visible only in the tail latency, which is the point.

    The draws are per request and seeded from the request id, so an in-process run is
    reproducible: the same seed and the same request ids produce the same fates. A request
    arriving over HTTP gets its id from the serving gateway, so a subprocess worker is
    reproducible only in distribution, not request by request.
    """

    def __init__(
        self,
        inner: Backend,
        faults: WorkerFaults | None = None,
        *,
        name: str | None = None,
        seed: int = 0,
    ) -> None:
        self._inner = inner
        self.faults = faults if faults is not None else WorkerFaults()
        self.name: str = name or str(getattr(inner, "name", "worker"))
        self.supports_lora = bool(getattr(inner, "supports_lora", False))
        self.stats = WorkerStats()
        self._seed = seed
        self._alive = True
        self._draining = False
        self._generation = 0

    @property
    def inner(self) -> Backend:
        """The backend being wrapped."""
        return self._inner

    @property
    def alive(self) -> bool:
        """Whether the replica is running (a killed one is not)."""
        return self._alive

    @property
    def draining(self) -> bool:
        """Whether the replica refuses new work while it finishes what it has."""
        return self._draining

    def drain(self) -> None:
        """Stop taking new requests and report unhealthy, but keep serving in-flight ones.

        The graceful half of a rollout: a Kubernetes preStop hook plus a failing readiness
        probe produce exactly this state, and it is what makes a pod deletion invisible to
        clients. Idempotent.
        """
        self._draining = True

    def kill(self) -> None:
        """Take the replica out now. In-flight streams abort at their next token boundary.

        The generation counter is what the in-flight streams watch: bumping it means a
        stream that started before the kill can tell that it did, without the backend having
        to hold a reference to every running generator.
        """
        self._alive = False
        self._draining = True
        self._generation += 1

    def revive(self) -> None:
        """Bring a killed or draining replica back, ready to serve again."""
        self._alive = True
        self._draining = False

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Stream from the wrapped backend, applying whatever faults are configured."""
        generation = self._generation
        if not self._alive:
            self.stats.refused += 1
            raise BackendUnavailableError(f"{self.name} is down", backend=self.name)
        if self._draining:
            self.stats.refused += 1
            raise BackendUnavailableError(f"{self.name} is draining", backend=self.name)
        if self.faults.partitioned:
            self.stats.refused += 1
            raise BackendUnavailableError(f"{self.name} is partitioned", backend=self.name)

        # Both draws happen unconditionally and in a fixed order, so that turning latency
        # injection on does not change which requests the error injection picks: two runs
        # that differ in one fault must stay comparable in every other respect.
        rng = random.Random(f"{self._seed}:{req.request_id}")
        error_draw = rng.random()
        latency_draw = rng.random()
        self.stats.requests += 1

        if self.faults.latency_ms > 0.0 and latency_draw < self.faults.latency_probability:
            self.stats.delayed += 1
            await asyncio.sleep(self.faults.latency_ms / 1000.0)
        if error_draw < self.faults.error_probability:
            self.stats.injected_errors += 1
            raise BackendUnavailableError(
                f"{self.name}: injected pre-first-token fault", backend=self.name
            )

        stream = self._inner.generate(req)
        try:
            async for event in stream:
                if self._generation != generation or not self._alive:
                    self.stats.aborted += 1
                    raise StreamInterruptedError(
                        f"{self.name} was killed with a stream in flight", backend=self.name
                    )
                yield event
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

    async def health(self) -> bool:
        """False while down, draining or partitioned; otherwise the inner backend's verdict."""
        if not self._alive or self._draining or self.faults.partitioned:
            return False
        return bool(await self._inner.health())

    async def models(self) -> list[str]:
        """Model names the wrapped backend serves."""
        return await self._inner.models()

    async def close(self) -> None:
        """Close the wrapped backend and stop serving. Idempotent."""
        self._alive = False
        await self._inner.close()

    def snapshot(self) -> dict[str, Any]:
        """Everything a result file records about this replica."""
        return {
            "name": self.name,
            "alive": self._alive,
            "draining": self._draining,
            "faults": self.faults.model_dump(),
            "stats": self.stats.model_dump(),
        }

    def __repr__(self) -> str:
        state = "up"
        if not self._alive:
            state = "down"
        elif self._draining:
            state = "draining"
        return f"FaultingBackend(name={self.name!r}, state={state})"


class WorkerSpec(BaseModel):
    """How one replica is configured, whether it runs in this process or another.

    One model for both flavours so that an in-process run and a subprocess run of the same
    experiment differ in exactly one field (``port``), and a result file records the replica
    configuration the same way either way.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(default="worker", min_length=1)
    host: str = "127.0.0.1"
    port: int = Field(default=0, ge=0, le=65535)
    """TCP port for a subprocess worker; zero means the replica runs in this process."""

    model: str = Field(default="mock-model", min_length=1)
    seed: int = 1234
    ttft_ms: float = Field(default=0.0, ge=0.0)
    itl_ms: float = Field(default=0.0, ge=0.0)
    jitter: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=32, ge=1)
    tokens_per_event: int = Field(default=1, ge=1)
    log_level: str = "WARNING"

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible base URL a gateway would point at this replica."""
        return f"http://{self.host}:{self.port}/v1"

    def mock_config(self) -> MockConfig:
        """The mock configuration this replica serves behind its fault injector."""
        return MockConfig(
            name=self.name,
            models=[self.model],
            max_tokens=self.max_tokens,
            ttft_ms=self.ttft_ms,
            itl_ms=self.itl_ms,
            jitter=self.jitter,
            tokens_per_event=self.tokens_per_event,
            seed=self.seed,
        )

    def argv(self, python: str | None = None) -> list[str]:
        """The command line that starts this replica as a child process."""
        return [
            python or sys.executable,
            "-m",
            "turboserve.chaos.worker",
            "--name",
            self.name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model",
            self.model,
            "--seed",
            str(self.seed),
            "--ttft-ms",
            f"{self.ttft_ms:g}",
            "--itl-ms",
            f"{self.itl_ms:g}",
            "--jitter",
            f"{self.jitter:g}",
            "--max-tokens",
            str(self.max_tokens),
            "--tokens-per-event",
            str(self.tokens_per_event),
            "--log-level",
            self.log_level,
        ]


# --------------------------------------------------------------------------------------
# The HTTP worker
# --------------------------------------------------------------------------------------


class LatencyRequest(BaseModel):
    """Body of ``POST /chaos/latency``."""

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(ge=0.0, le=1.0)
    ms: float = Field(default=0.0, ge=0.0)


class ErrorRequest(BaseModel):
    """Body of ``POST /chaos/error``."""

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(ge=0.0, le=1.0)


class PartitionRequest(BaseModel):
    """Body of ``POST /chaos/partition``."""

    model_config = ConfigDict(extra="forbid")

    partitioned: bool = True


class DrainRequest(BaseModel):
    """Body of ``POST /chaos/drain``."""

    model_config = ConfigDict(extra="forbid")

    draining: bool = True


def build_worker_app(spec: WorkerSpec, faults: WorkerFaults | None = None) -> FastAPI:
    """Build the OpenAI-compatible server for one breakable replica.

    FastAPI and the gateway app are imported inside the function: the harness's in-process
    mode never needs them, and a chaos run that only kills in-process workers should not pay
    for the web stack.
    """
    from fastapi import APIRouter, Request, Response
    from fastapi.responses import JSONResponse

    from turboserve.config import Settings
    from turboserve.gateway.app import GatewayOptions, create_app
    from turboserve.gateway.router import Router
    from turboserve.gateway.tenants import TenantRegistry

    backend = FaultingBackend(
        MockBackend(spec.mock_config()), faults, name=spec.name, seed=spec.seed
    )
    # A model server does not retry itself and does not cache an opinion about its own
    # health: one attempt, and every probe asks the backend. Without this the replica would
    # hide the very faults it was asked to inject -- an injected error would be retried
    # internally and reported as "no healthy backend", and one injected error would take the
    # replica out of its own rotation for the length of the health TTL.
    router = Router(max_attempts=1, health_ttl_s=0.0)
    router.add_backend(spec.model, backend)
    app = create_app(
        Settings(model=spec.model),
        router=router,
        tenants=TenantRegistry.default(),
        options=GatewayOptions(require_auth=False, served_model_names=[spec.model]),
    )
    app.state.chaos_backend = backend

    @app.middleware("http")
    async def _partition_middleware(request: Request, call_next: Any) -> Response:
        """Answer every data-plane path with 503 while the replica is partitioned."""
        if backend.faults.partitioned and not request.url.path.startswith(CHAOS_PREFIX):
            return JSONResponse(
                {"error": {"message": f"{spec.name} is partitioned", "type": "chaos"}},
                status_code=_HTTP_SERVICE_UNAVAILABLE,
            )
        response: Response = await call_next(request)
        return response

    control = APIRouter(prefix=CHAOS_PREFIX, tags=["chaos"])

    def state() -> dict[str, Any]:
        return backend.snapshot() | {"mock": _mock_stats(backend)}

    @control.get("/state")
    async def chaos_state() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Current fault configuration and counters for this replica."""
        return state()

    @control.post("/faults")
    async def set_faults(faults_in: WorkerFaults) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Replace the whole fault configuration."""
        backend.faults = faults_in
        logger.info("%s: faults set to %s", spec.name, faults_in.model_dump())
        return state()

    @control.post("/latency")
    async def set_latency(body: LatencyRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Delay a fraction of requests before their first token."""
        backend.faults.latency_probability = body.probability
        backend.faults.latency_ms = body.ms
        return state()

    @control.post("/error")
    async def set_error(body: ErrorRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Fail a fraction of requests before their first token."""
        backend.faults.error_probability = body.probability
        return state()

    @control.post("/partition")
    async def set_partition(body: PartitionRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Make the replica unreachable, or reachable again."""
        backend.faults.partitioned = body.partitioned
        logger.info("%s: partitioned=%s", spec.name, body.partitioned)
        return state()

    @control.post("/drain")
    async def set_drain(body: DrainRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Stop accepting new requests while finishing the ones in flight, or resume."""
        if body.draining:
            backend.drain()
        else:
            backend.revive()
        return state()

    @control.post("/reset")
    async def reset() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Clear every fault and bring the replica back up."""
        backend.faults = WorkerFaults()
        backend.revive()
        return state()

    app.include_router(control)
    return app


def _mock_stats(backend: FaultingBackend) -> dict[str, Any]:
    """The wrapped mock's counters, when the wrapped backend is a mock."""
    inner = backend.inner
    stats = getattr(inner, "stats", None)
    return stats.model_dump() if isinstance(stats, BaseModel) else {}


def serve_worker(spec: WorkerSpec, faults: WorkerFaults | None = None) -> None:
    """Run one worker under uvicorn until the process is stopped or killed."""
    import uvicorn

    app = build_worker_app(spec, faults)
    logger.info("chaos worker %s serving on %s", spec.name, spec.base_url)
    uvicorn.run(app, host=spec.host, port=spec.port, log_level=spec.log_level.lower())


def _main(argv: Sequence[str] | None = None) -> None:
    """``python -m turboserve.chaos.worker`` -- argparse, to keep start-up cheap."""
    import argparse

    from turboserve.logging_utils import configure_logging

    parser = argparse.ArgumentParser(description="Run one breakable OpenAI-compatible replica.")
    parser.add_argument("--name", default="worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ttft-ms", type=float, default=0.0)
    parser.add_argument("--itl-ms", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--tokens-per-event", type=int, default=1)
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--latency-probability", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--error-probability", type=float, default=0.0)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    spec = WorkerSpec(
        name=args.name,
        host=args.host,
        port=args.port,
        model=args.model,
        seed=args.seed,
        ttft_ms=args.ttft_ms,
        itl_ms=args.itl_ms,
        jitter=args.jitter,
        max_tokens=args.max_tokens,
        tokens_per_event=args.tokens_per_event,
        log_level=args.log_level,
    )
    faults = WorkerFaults(
        latency_probability=args.latency_probability,
        latency_ms=args.latency_ms,
        error_probability=args.error_probability,
    )
    serve_worker(spec, faults)


# --------------------------------------------------------------------------------------
# The two worker flavours
# --------------------------------------------------------------------------------------


@runtime_checkable
class ChaosWorker(Protocol):
    """One replica the harness can start, break and restart.

    Every method is a coroutine even where one flavour could do the work synchronously: the
    harness applies a timeline with ``await`` and must not have to know which flavour it
    holds, and a subprocess worker genuinely has to wait for a process and an HTTP call.
    """

    name: str

    @property
    def backend(self) -> Backend:
        """What the router routes to. Stable across restarts."""
        ...

    @property
    def alive(self) -> bool:
        """Whether the replica is currently running."""
        ...

    async def start(self) -> None:
        """Bring the replica up and wait until it can serve."""
        ...

    async def stop(self) -> None:
        """Shut the replica down for good and release its resources."""
        ...

    async def drain(self) -> None:
        """Refuse new requests, keep serving in-flight ones, report unhealthy."""
        ...

    async def kill(self) -> None:
        """Take the replica out abruptly, losing whatever was in flight."""
        ...

    async def restart(self) -> None:
        """Bring a killed replica back and re-apply its steady-state faults."""
        ...

    async def apply_faults(self, faults: WorkerFaults) -> None:
        """Replace the per-request fault configuration."""
        ...

    async def set_partitioned(self, partitioned: bool) -> None:
        """Make the replica unreachable, or reachable again."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Configuration and counters, for the result file."""
        ...


@dataclass(slots=True)
class _Lifecycle:
    """Counters every worker flavour keeps about its own disruptions."""

    kills: int = 0
    restarts: int = 0
    drains: int = 0
    partitions: int = 0

    def to_dict(self) -> dict[str, int]:
        """JSON-ready counters for the result file."""
        return {
            "kills": self.kills,
            "restarts": self.restarts,
            "drains": self.drains,
            "partitions": self.partitions,
        }


class InProcessWorker:
    """A replica that lives in this event loop, for tests and for a laptop smoke run."""

    def __init__(self, spec: WorkerSpec, faults: WorkerFaults | None = None) -> None:
        self.spec = spec
        self.name = spec.name
        self._backend = FaultingBackend(
            MockBackend(spec.mock_config()),
            faults,
            name=spec.name,
            seed=spec.seed,
        )
        self._lifecycle = _Lifecycle()

    @property
    def backend(self) -> Backend:
        """The faulting backend the router routes to."""
        return self._backend

    @property
    def faults(self) -> WorkerFaults:
        """The live fault configuration; assigning to its fields takes effect immediately."""
        return self._backend.faults

    @property
    def alive(self) -> bool:
        """Whether the replica is serving."""
        return self._backend.alive

    async def start(self) -> None:
        """Bring the replica up. Cheap and idempotent -- there is no process to spawn."""
        self._backend.revive()

    async def stop(self) -> None:
        """Close the backend; further requests fail as unavailable."""
        await self._backend.close()

    async def drain(self) -> None:
        """Mark unready and refuse new requests."""
        self._backend.drain()
        self._lifecycle.drains += 1

    async def kill(self) -> None:
        """Take the replica out; in-flight streams abort at their next token boundary."""
        self._backend.kill()
        self._lifecycle.kills += 1

    async def restart(self) -> None:
        """Bring the replica back. Its counters are kept, its faults are unchanged."""
        self._backend.revive()
        self._lifecycle.restarts += 1

    async def apply_faults(self, faults: WorkerFaults) -> None:
        """Replace the fault configuration wholesale."""
        self._backend.faults = faults.model_copy(deep=True)

    async def set_partitioned(self, partitioned: bool) -> None:
        """Refuse or resume every request, health probes included."""
        self._backend.faults.partitioned = partitioned
        if partitioned:
            self._lifecycle.partitions += 1

    def snapshot(self) -> dict[str, Any]:
        """Configuration and counters for the result file."""
        return self._backend.snapshot() | {
            "mode": "inprocess",
            "lifecycle": self._lifecycle.to_dict(),
            "mock": _mock_stats(self._backend),
        }

    def __repr__(self) -> str:
        return f"InProcessWorker(name={self.name!r}, alive={self.alive})"


class SubprocessWorker:
    """A replica served over HTTP by a child process the harness can really kill.

    The backend object is created once and reused across restarts: its base URL does not
    change, and re-creating it would drop the connection pool and reset the metric labels
    that identify this replica. A restarted process comes back with no faults configured,
    so :meth:`restart` re-applies the last configuration -- otherwise the second half of a
    run would silently be the easy half.
    """

    def __init__(
        self,
        spec: WorkerSpec,
        *,
        python: str | None = None,
        startup_timeout_s: float = 30.0,
        shutdown_timeout_s: float = 5.0,
        control_timeout_s: float = 5.0,
        request_timeout_s: float = 120.0,
        connect_timeout_s: float = 1.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if spec.port <= 0:
            raise ValueError("a subprocess worker needs a real port; got 0")
        from turboserve.gateway.backends.openai_compat import OpenAICompatBackend

        self.spec = spec
        self.name = spec.name
        self._python = python
        self._startup_timeout_s = startup_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._env = dict(env) if env is not None else None
        self._process: asyncio.subprocess.Process | None = None
        self._faults = WorkerFaults()
        self._lifecycle = _Lifecycle()
        self._origin = f"http://{spec.host}:{spec.port}"
        self._control = httpx.AsyncClient(base_url=self._origin, timeout=control_timeout_s)
        # The health probe deliberately reads /readyz rather than /healthz: /healthz says
        # only that the process answers, while /readyz reflects the replica's own backend,
        # which is what a drain flips.
        self._backend = OpenAICompatBackend(
            spec.base_url,
            name=spec.name,
            timeout_s=request_timeout_s,
            connect_timeout_s=connect_timeout_s,
            health_path="/readyz",
        )

    @property
    def backend(self) -> Backend:
        """The HTTP backend the router routes to."""
        return self._backend

    @property
    def alive(self) -> bool:
        """Whether the child process is running."""
        return self._process is not None and self._process.returncode is None

    def argv(self) -> list[str]:
        """The command line used to spawn the replica."""
        return self.spec.argv(self._python)

    async def start(self) -> None:
        """Spawn the child process and wait until it answers its liveness probe."""
        if self.alive:
            return
        argv = self.argv()
        logger.info("starting chaos worker %s: %s", self.name, " ".join(argv))
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=self._env,
        )
        await self._wait_ready()

    async def _wait_ready(self) -> None:
        """Poll ``/healthz`` until the server answers, or give up with a clear error."""
        deadline = time.monotonic() + self._startup_timeout_s
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.returncode is not None:
                raise RuntimeError(
                    f"chaos worker {self.name} exited with code {process.returncode} "
                    f"before it became ready"
                )
            try:
                response = await self._control.get("/healthz")
            except httpx.HTTPError:
                await asyncio.sleep(0.05)
                continue
            if response.status_code == httpx.codes.OK:
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"chaos worker {self.name} did not become ready within {self._startup_timeout_s:g}s"
        )

    async def stop(self) -> None:
        """Ask the child to exit, kill it if it will not, and close the HTTP clients."""
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=self._shutdown_timeout_s)
            except TimeoutError:
                process.kill()
                await process.wait()
        self._process = None
        await self._control.aclose()
        await self._backend.close()

    async def drain(self) -> None:
        """Tell the replica to refuse new requests and report unready."""
        await self._post("/chaos/drain", {"draining": True})
        self._lifecycle.drains += 1

    async def kill(self) -> None:
        """``SIGKILL`` the child process: no cleanup, no goodbye, in-flight streams lost."""
        process = self._process
        if process is None or process.returncode is not None:
            return
        process.kill()
        await process.wait()
        self._process = None
        self._lifecycle.kills += 1
        logger.info("chaos worker %s killed", self.name)

    async def restart(self) -> None:
        """Start a fresh process and put its faults back the way they were."""
        await self.start()
        self._lifecycle.restarts += 1
        if not self._faults.is_empty:
            await self._post("/chaos/faults", self._faults.model_dump())

    async def apply_faults(self, faults: WorkerFaults) -> None:
        """Replace the fault configuration, remembering it for the next restart."""
        self._faults = faults.model_copy(deep=True)
        await self._post("/chaos/faults", self._faults.model_dump())

    async def set_partitioned(self, partitioned: bool) -> None:
        """Make the replica answer 503 to everything, or stop doing so."""
        self._faults.partitioned = partitioned
        await self._post("/chaos/partition", {"partitioned": partitioned})
        if partitioned:
            self._lifecycle.partitions += 1

    async def state(self) -> dict[str, Any]:
        """Read the replica's own view of its faults and counters."""
        response = await self._control.get("/chaos/state")
        response.raise_for_status()
        data = response.json()
        return dict(data) if isinstance(data, dict) else {}

    async def _post(self, path: str, body: Mapping[str, Any]) -> None:
        """Send one control-plane request, tolerating a replica that is not up.

        A control call to a killed replica is expected -- the timeline may drain something
        that has already died -- and must not abort the run, so a transport failure is
        logged and swallowed. A non-2xx answer is a bug in this module and is logged loudly.
        """
        if not self.alive:
            logger.debug("skipping %s on %s: not running", path, self.name)
            return
        try:
            response = await self._control.post(path, json=dict(body))
        except httpx.HTTPError as exc:
            logger.warning("control call %s to %s failed: %s", path, self.name, exc)
            return
        if response.status_code >= httpx.codes.BAD_REQUEST:
            logger.error(
                "control call %s to %s returned HTTP %d: %s",
                path,
                self.name,
                response.status_code,
                response.text[:200],
            )

    def snapshot(self) -> dict[str, Any]:
        """Configuration and counters for the result file."""
        return {
            "name": self.name,
            "mode": "subprocess",
            "alive": self.alive,
            "base_url": self.spec.base_url,
            "faults": self._faults.model_dump(),
            "lifecycle": self._lifecycle.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"SubprocessWorker(name={self.name!r}, url={self.spec.base_url!r}, alive={self.alive})"
        )


def build_workers(
    specs: Sequence[WorkerSpec],
    *,
    mode: str = "subprocess",
    faults: WorkerFaults | None = None,
    python: str | None = None,
) -> list[ChaosWorker]:
    """Build one worker per spec in the requested mode."""
    if mode == "inprocess":
        return [
            InProcessWorker(spec, faults.model_copy(deep=True) if faults is not None else None)
            for spec in specs
        ]
    if mode == "subprocess":
        return [SubprocessWorker(spec, python=python) for spec in specs]
    raise ValueError(f"mode must be 'inprocess' or 'subprocess', got {mode!r}")


if __name__ == "__main__":  # pragma: no cover - module entry point
    _main()
