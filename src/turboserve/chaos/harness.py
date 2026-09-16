"""The chaos harness: a fleet that keeps breaking, a gateway in front of it, and a verdict.

The experiment this module runs is one sentence long: *send steady open-loop traffic through
the gateway's router at a fleet of replicas that are being killed, delayed, failed and
partitioned on a fixed schedule, and record what the client saw*. Everything below exists to
make that sentence measurable.

What is being measured is the router, not the replicas. The replicas are mock servers (see
:mod:`turboserve.chaos.worker`) precisely so that no model, GPU or checkpoint is involved
and the only thing that can vary between two runs is the gateway's behaviour. Four numbers
come out of it, and each one answers a question that "did it survive?" does not:

``error_rate``
    The fraction of requests the client saw fail. A killed replica should be invisible; a
    killed replica *with a stream in flight* cannot be, because those bytes are already gone
    (see :class:`~turboserve.gateway.backends.protocol.StreamInterruptedError`). The harness
    therefore also reports the failures separately by cause, so a run that fails its
    threshold says *why* rather than just how much.
``retries``
    Attempts minus requests, read off the router's own per-replica counters. This is the
    evidence that the error rate is low *because* the retry path worked, not because nothing
    was ever routed at a dead replica.
``recovery``
    Per disruption, the time from a replica going out to the first request it successfully
    serves again. It covers the restart, the process start-up and the router's health-cache
    TTL, which is the part an operator actually controls.
``latency during faults vs in steady state``
    The same records split by whether they were sent inside a disruption window. A gateway
    that hides every failure by retrying can still ruin the tail, and one aggregate p95 over
    the whole run would hide that entirely.

The load is open-loop (Poisson arrivals at a fixed rate) for the reason given in
:mod:`turboserve.bench.loadgen`: a closed loop slows down when the system does, which is
exactly the wrong instrument for a failure experiment, because it hides queueing behind
client backpressure.

Results are written in the ordinary benchmark schema
(:class:`~turboserve.bench.records.RunResult`) with the chaos-specific figures under
``summary["chaos"]``, so the report renderer, the index and
``deploy/kind/assert_error_rate.py`` all read a chaos run with no special case.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer

from turboserve.bench.loadgen import LoadSpec, build_requests, load_config, run_open_loop
from turboserve.bench.metrics import summarize_records
from turboserve.bench.prompts import BenchPrompt
from turboserve.bench.records import SLO, Percentiles, RunResult
from turboserve.chaos.faults import (
    FaultAction,
    FaultError,
    FaultEvent,
    FaultSchedule,
    outage_windows,
    parse_duration,
)
from turboserve.chaos.worker import ChaosWorker, WorkerFaults, WorkerSpec, build_workers
from turboserve.gateway.router import Router

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator, Iterable, Mapping, Sequence

    from turboserve.bench.records import RequestRecord
    from turboserve.gateway.backends.protocol import Backend, GenerateRequest, TokenEvent

logger = logging.getLogger(__name__)

__all__ = [
    "ChaosHarness",
    "ChaosReport",
    "ChaosSpec",
    "Disruption",
    "chaos_app",
    "default_result_path",
]

NS_PER_S = 1_000_000_000

WorkerMode = Literal["inprocess", "subprocess"]


def default_result_path(
    results_dir: Path | str = "results", *, now: datetime | None = None
) -> Path:
    """``results/chaos/<utc timestamp>.json`` -- where a run writes itself by default."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return Path(results_dir) / "chaos" / f"{stamp}.json"


@dataclass(frozen=True, slots=True)
class ChaosSpec:
    """Everything that defines one chaos experiment.

    Frozen, and embedded verbatim in the result file: a run that cannot be described by its
    own output cannot be reproduced, and "we killed some pods for a while" is not a
    description.
    """

    faults: FaultSchedule = field(default_factory=FaultSchedule)
    replicas: int = 3
    duration_s: float = 60.0
    rate_rps: float = 20.0
    mode: WorkerMode = "subprocess"
    model: str = "mock-model"
    tenant: str = "chaos"
    seed: int = 20260916
    profile: str = "custom"

    input_tokens: tuple[int, int] = (128, 512)
    output_tokens: tuple[int, int] = (32, 64)
    ttft_ms: float = 20.0
    itl_ms: float = 5.0
    jitter: float = 0.0
    tokens_per_event: int = 1

    max_attempts: int = 3
    """Router attempts per request. Retries happen only before the first token."""

    health_ttl_s: float = 1.0
    """How long a replica's health verdict is believed. It bounds the recovery time."""

    request_timeout_s: float | None = 60.0
    pool_size: int = 0
    """Distinct prompts to cycle through; zero derives one from the rate."""

    host: str = "127.0.0.1"
    base_port: int = 9100
    slo: SLO | None = None
    gpu_price_per_hour: float | None = None
    price_source: str | None = None

    def __post_init__(self) -> None:
        if self.replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {self.replicas}")
        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.rate_rps <= 0.0:
            raise ValueError(f"rate_rps must be positive, got {self.rate_rps}")
        if self.mode not in ("inprocess", "subprocess"):
            raise ValueError(f"mode must be 'inprocess' or 'subprocess', got {self.mode!r}")
        for name, (low, high) in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if low < 1 or high < low:
                raise ValueError(f"{name}={low, high} must satisfy 1 <= min <= max")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.pool_size < 0:
            raise ValueError(f"pool_size must be >= 0, got {self.pool_size}")

    @property
    def prompt_pool_size(self) -> int:
        """How many distinct prompts to build.

        Enough that the workload is not one prompt repeated -- which the mock would not care
        about but a real replica behind the same harness would -- and few enough that a long
        run does not spend its memory on prompts it sends once.
        """
        if self.pool_size:
            return self.pool_size
        return max(8, min(512, int(self.rate_rps * 5)))

    def worker_specs(self) -> list[WorkerSpec]:
        """One :class:`WorkerSpec` per replica, ports assigned from ``base_port``.

        Each replica gets its own mock seed (``seed + index``) so that two replicas serving
        the same request id do not produce identical latencies -- a fleet whose members are
        bit-identical would hide any effect that depends on which replica served what.
        """
        return [
            WorkerSpec(
                name=f"replica-{index}",
                host=self.host,
                port=0 if self.mode == "inprocess" else self.base_port + index,
                model=self.model,
                seed=self.seed + index,
                ttft_ms=self.ttft_ms,
                itl_ms=self.itl_ms,
                jitter=self.jitter,
                max_tokens=self.output_tokens[1],
                tokens_per_event=self.tokens_per_event,
            )
            for index in range(self.replicas)
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description for the result file's ``config`` block."""
        return {
            "replicas": self.replicas,
            "duration_s": self.duration_s,
            "rate_rps": self.rate_rps,
            "mode": self.mode,
            "model": self.model,
            "tenant": self.tenant,
            "seed": self.seed,
            "input_tokens": list(self.input_tokens),
            "output_tokens": list(self.output_tokens),
            "ttft_ms": self.ttft_ms,
            "itl_ms": self.itl_ms,
            "jitter": self.jitter,
            "tokens_per_event": self.tokens_per_event,
            "max_attempts": self.max_attempts,
            "health_ttl_s": self.health_ttl_s,
            "request_timeout_s": self.request_timeout_s,
            "prompt_pool_size": self.prompt_pool_size,
            "replica_engine": "mock",
            "faults": self.faults.to_dict(),
        }


@dataclass(slots=True)
class Disruption:
    """One replica's time out of service, and how long the fleet took to use it again.

    ``recovered_s`` is deliberately not the moment the replica came back: it is the moment
    the *router* successfully served a request from it again. The gap between the two is the
    health-cache TTL plus whatever the replica needed to start, and it is the part an
    operator can tune, so it belongs in the measurement.
    """

    target: str
    kind: str
    spec: str
    started_s: float
    killed_s: float | None = None
    restored_s: float | None = None
    recovered_s: float | None = None
    error: str | None = None

    @property
    def recovery_s(self) -> float | None:
        """Seconds from going out of service to serving successfully again."""
        if self.recovered_s is None:
            return None
        return max(0.0, self.recovered_s - self.started_s)

    @property
    def downtime_s(self) -> float | None:
        """Seconds between the replica going out and being brought back."""
        if self.restored_s is None:
            return None
        return max(0.0, self.restored_s - self.started_s)

    def window(self, *, fallback_end_s: float) -> tuple[float, float]:
        """The interval this disruption impaired the fleet for.

        It ends when traffic was served by the replica again, or -- if that never happened
        within the run -- at the end of the run, because there is no evidence it recovered.
        """
        end = self.recovered_s if self.recovered_s is not None else fallback_end_s
        return (self.started_s, max(self.started_s, end))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for the result file."""
        return {
            "target": self.target,
            "kind": self.kind,
            "spec": self.spec,
            "started_s": self.started_s,
            "killed_s": self.killed_s,
            "restored_s": self.restored_s,
            "recovered_s": self.recovered_s,
            "recovery_s": self.recovery_s,
            "downtime_s": self.downtime_s,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class _Completion:
    """When a request finished, which replica served it, and whether it succeeded."""

    request_id: str
    t_s: float
    backend: str
    ok: bool


@dataclass(slots=True)
class ChaosReport:
    """The outcome of one run: the standard result object plus typed access to the extras."""

    run: RunResult
    spec: ChaosSpec
    events: tuple[FaultEvent, ...] = ()
    disruptions: tuple[Disruption, ...] = ()
    path: Path | None = None

    @property
    def num_requests(self) -> int:
        """Requests the client sent and observed."""
        return len(self.run.requests)

    @property
    def num_failed(self) -> int:
        """Requests that did not complete successfully."""
        return sum(1 for record in self.run.requests if not record.ok)

    @property
    def error_rate(self) -> float:
        """Failed requests as a fraction of all of them; zero for an empty run."""
        total = self.num_requests
        return (self.num_failed / total) if total else 0.0

    @property
    def chaos(self) -> dict[str, Any]:
        """The ``summary["chaos"]`` block."""
        block = self.run.summary.get("chaos")
        return dict(block) if isinstance(block, dict) else {}

    @property
    def retries(self) -> int:
        """Router attempts beyond the first, across the whole run."""
        return int(self.chaos.get("retries", 0))

    @property
    def recoveries_s(self) -> list[float]:
        """Observed recovery times, in seconds."""
        return [
            disruption.recovery_s
            for disruption in self.disruptions
            if disruption.recovery_s is not None
        ]

    def save(self, path: Path | str, *, update_index: bool = True) -> Path:
        """Write the run file (and register it in ``results/index.json``)."""
        self.path = self.run.save(path, update_index=update_index)
        return self.path


class _CountingBackend:
    """Wraps a replica's backend to record which requests were attempted where.

    The router already counts attempts per replica, but not *per request*, and the
    difference is the whole point of the retry metric: three attempts spread over three
    requests says the retry path never ran, while three attempts on one request says it ran
    twice. Counting here -- one layer below the router, one above the replica -- is the only
    place where the request id and the chosen replica are both in hand, and it costs one
    dictionary append per attempt.
    """

    __slots__ = ("_attempts", "_inner", "name", "supports_lora")

    def __init__(self, inner: Backend, attempts: dict[str, list[str]]) -> None:
        self._inner = inner
        self._attempts = attempts
        self.name: str = str(getattr(inner, "name", "replica"))
        self.supports_lora = bool(getattr(inner, "supports_lora", False))

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Record this attempt, then stream from the replica."""
        self._attempts.setdefault(req.request_id, []).append(self.name)
        stream = self._inner.generate(req)
        try:
            async for event in stream:
                yield event
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

    async def health(self) -> bool:
        """The replica's own verdict."""
        return bool(await self._inner.health())

    async def models(self) -> list[str]:
        """The replica's model list."""
        return await self._inner.models()

    async def close(self) -> None:
        """Close the replica's backend."""
        await self._inner.close()


class _RoutedLoad:
    """Adapts :meth:`Router.generate` to the load generator, recording what the router did.

    The load generator wants ``generate(req) -> AsyncIterator[TokenEvent]`` and knows nothing
    about lanes or replicas; the harness needs to know which replica served each request and
    when it finished, to attribute recoveries. Doing that here rather than by inspecting the
    records afterwards is the only place both facts exist at once.
    """

    __slots__ = ("_router", "_t0_ns", "backend_of", "completions")

    def __init__(self, router: Router, *, t0_ns: int) -> None:
        self._router = router
        self._t0_ns = t0_ns
        self.backend_of: dict[str, str] = {}
        self.completions: list[_Completion] = []

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Stream one request through the router, remembering how it went."""
        backend = ""
        ok = True
        completed = False
        try:
            async for routed in self._router.generate(req):
                if not backend:
                    backend = routed.backend
                    self.backend_of[req.request_id] = backend
                if routed.event.is_error:
                    ok = False
                yield routed.event
            completed = True
        finally:
            self.completions.append(
                _Completion(
                    request_id=req.request_id,
                    t_s=(time.monotonic_ns() - self._t0_ns) / NS_PER_S,
                    backend=backend,
                    ok=ok and completed,
                )
            )


class ChaosHarness:
    """Runs one :class:`ChaosSpec`: start the fleet, drive load, apply faults, report.

    The fault timeline and the load generator run as two concurrent tasks over one monotonic
    clock, so a fault's timestamp and a request's timestamp are directly comparable -- which
    is what makes "p95 during faults" a definition rather than an impression.
    """

    def __init__(
        self,
        spec: ChaosSpec,
        *,
        workers: Sequence[ChaosWorker] | None = None,
        hardware: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self._hardware = dict(hardware) if hardware is not None else None
        self._workers: list[ChaosWorker] = (
            list(workers)
            if workers is not None
            else build_workers(
                spec.worker_specs(),
                mode=spec.mode,
                faults=WorkerFaults.from_steady(spec.faults.steady()),
            )
        )
        if not self._workers:
            raise ValueError("a chaos run needs at least one worker")

    @property
    def workers(self) -> list[ChaosWorker]:
        """The replicas this harness drives."""
        return list(self._workers)

    def prompts(self) -> list[BenchPrompt]:
        """Seeded synthetic prompts, as token ids.

        Ids rather than text, and no tokenizer: the replicas are mock servers that count
        ids, and pulling a real tokenizer into a fault-injection run would make it depend on
        a model download for no measurement gain. The lengths come from the spec so the
        request *shape* still matches the profile the run claims to represent.
        """
        rng = random.Random(self.spec.seed)
        prompts: list[BenchPrompt] = []
        for index in range(self.spec.prompt_pool_size):
            num_input = rng.randint(*self.spec.input_tokens)
            num_output = rng.randint(*self.spec.output_tokens)
            prompts.append(
                BenchPrompt(
                    prompt_id=f"chaos-{index:05d}",
                    token_ids=[rng.randrange(1000, 60000) for _ in range(num_input)],
                    text="",
                    max_tokens=num_output,
                    tenant=self.spec.tenant,
                    source="synthetic",
                )
            )
        return prompts

    def build_router(self, attempts: dict[str, list[str]] | None = None) -> Router:
        """A router over the fleet, with the retry and health behaviour under test.

        ``attempts`` is the per-request attempt log :class:`_CountingBackend` fills in; pass
        one to measure retries, omit it to route straight at the replicas.
        """
        router = Router(
            health_ttl_s=self.spec.health_ttl_s,
            max_attempts=self.spec.max_attempts,
            rng=random.Random(self.spec.seed),
        )
        for worker in self._workers:
            backend = (
                worker.backend if attempts is None else _CountingBackend(worker.backend, attempts)
            )
            router.add_backend(self.spec.model, backend)
        return router

    async def run(self) -> ChaosReport:
        """Execute the experiment and return its report. Does not write any file."""
        spec = self.spec
        names = [worker.name for worker in self._workers]
        events = spec.faults.events(duration_s=spec.duration_s, workers=names, seed=spec.seed)
        outages = outage_windows(events, names, duration_s=spec.duration_s)
        if outages:
            logger.warning(
                "this schedule leaves no replica serving during %s; the error rate it "
                "produces describes the schedule, not the gateway",
                ", ".join(f"[{start:g}s, {end:g}s)" for start, end in outages),
            )

        run = RunResult.start(
            "chaos",
            spec.profile,
            config=self._config(),
            hardware=self._hardware,
            gpu_price_per_hour=spec.gpu_price_per_hour,
            price_source=spec.price_source,
        )
        attempts_by_request: dict[str, list[str]] = {}
        router = self.build_router(attempts_by_request)
        disruptions: list[Disruption] = []
        try:
            for worker in self._workers:
                await worker.start()
            steady = spec.faults.steady()
            if not steady.is_empty:
                for worker in self._workers:
                    await worker.apply_faults(WorkerFaults.from_steady(steady))
            # Probe once before the clock starts, so the first requests are not charged for
            # a cold health cache that has nothing to do with the faults being measured.
            await router.health_report()

            requests = build_requests(self.prompts(), model=spec.model, id_prefix="")
            t0_ns = time.monotonic_ns()
            tracker = _RoutedLoad(router, t0_ns=t0_ns)
            fault_task = asyncio.create_task(
                self._drive_faults(events, t0_ns=t0_ns, disruptions=disruptions)
            )
            load_spec = LoadSpec(
                mode="open",
                rate_rps=spec.rate_rps,
                duration_s=spec.duration_s,
                repeat_requests=True,
                request_timeout_s=spec.request_timeout_s,
                seed=spec.seed,
            )
            try:
                load = await run_open_loop(requests, tracker.generate, spec=load_spec)
            finally:
                fault_task.cancel()
                await asyncio.gather(fault_task, return_exceptions=True)

            for record in load.records:
                record.backend = tracker.backend_of.get(record.request_id, "")
                run.add(record)
            run.finish(slo=spec.slo)
            run.summary["chaos"] = self._chaos_summary(
                attempts_by_request=attempts_by_request,
                tracker=tracker,
                events=events,
                disruptions=disruptions,
                outages=outages,
                records=load.records,
                t0_ns=t0_ns,
                wall_s=load.wall_seconds,
                load_config=load_config(load_spec, prompt_pool=spec.prompt_pool_size),
            )
        finally:
            for worker in self._workers:
                try:
                    await worker.stop()
                except Exception:  # noqa: BLE001 - one replica must not block the teardown
                    logger.exception("stopping worker %s failed", worker.name)
            await router.close()

        return ChaosReport(
            run=run,
            spec=spec,
            events=events,
            disruptions=tuple(disruptions),
        )

    # -- the fault timeline -------------------------------------------------------------

    async def _drive_faults(
        self,
        events: Sequence[FaultEvent],
        *,
        t0_ns: int,
        disruptions: list[Disruption],
    ) -> None:
        """Apply each event at its scheduled moment, relative to ``t0_ns``.

        Sleeps are computed against the absolute target rather than as a gap after the
        previous action, so the time an action itself takes (a process kill, an HTTP call)
        does not accumulate into the schedule and shift every later fault.
        """
        by_name = {worker.name: worker for worker in self._workers}
        open_by_target: dict[str, Disruption] = {}
        for event in events:
            delay = (t0_ns + event.t_s * NS_PER_S - time.monotonic_ns()) / NS_PER_S
            if delay > 0:
                await asyncio.sleep(delay)
            worker = by_name.get(event.target)
            if worker is None:  # pragma: no cover - events() validates targets
                logger.error("fault event targets unknown worker %s", event.target)
                continue
            try:
                await self._apply(
                    event,
                    worker,
                    t0_ns=t0_ns,
                    open_by_target=open_by_target,
                    disruptions=disruptions,
                )
            except Exception as exc:  # noqa: BLE001 - a failed injection is data, not a crash
                logger.exception("applying %s to %s failed", event.action, event.target)
                disruption = open_by_target.get(event.target)
                if disruption is not None:
                    disruption.error = f"{type(exc).__name__}: {exc}"

    async def _apply(
        self,
        event: FaultEvent,
        worker: ChaosWorker,
        *,
        t0_ns: int,
        open_by_target: dict[str, Disruption],
        disruptions: list[Disruption],
    ) -> None:
        """Perform one timeline action and keep the disruption bookkeeping straight.

        A disruption's start is stamped *before* the action and its restoration *after* it,
        because the two are not symmetrical: taking a replica out is immediate, while
        bringing one back means waiting for a process to start and answer, which for a real
        subprocess is the larger part of the outage. Stamping the restoration beforehand
        would credit the fleet with a recovery it had not had yet.
        """
        now_s = (time.monotonic_ns() - t0_ns) / NS_PER_S

        def begin() -> Disruption:
            existing = open_by_target.get(event.target)
            if existing is not None:
                return existing
            disruption = Disruption(
                target=event.target, kind=event.kind, spec=event.spec, started_s=now_s
            )
            open_by_target[event.target] = disruption
            disruptions.append(disruption)
            return disruption

        match event.action:
            case FaultAction.DRAIN:
                begin()
                await worker.drain()
            case FaultAction.KILL:
                begin().killed_s = now_s
                await worker.kill()
            case FaultAction.RESTART:
                await worker.restart()
                disruption = open_by_target.pop(event.target, None)
                if disruption is not None:
                    disruption.restored_s = (time.monotonic_ns() - t0_ns) / NS_PER_S
            case FaultAction.PARTITION_START:
                begin()
                await worker.set_partitioned(True)
            case FaultAction.PARTITION_END:
                await worker.set_partitioned(False)
                disruption = open_by_target.pop(event.target, None)
                if disruption is not None:
                    disruption.restored_s = (time.monotonic_ns() - t0_ns) / NS_PER_S

    # -- reporting ------------------------------------------------------------------------

    def _config(self) -> dict[str, Any]:
        """The result file's ``config`` block."""
        return {
            "scenario": "chaos",
            "profile": self.spec.profile,
            "seed": self.spec.seed,
            "tenants": [self.spec.tenant],
            "workload": self.spec.to_dict(),
        }

    def _chaos_summary(
        self,
        *,
        attempts_by_request: Mapping[str, Sequence[str]],
        tracker: _RoutedLoad,
        events: Sequence[FaultEvent],
        disruptions: Sequence[Disruption],
        outages: Sequence[tuple[float, float]],
        records: Sequence[RequestRecord],
        t0_ns: int,
        wall_s: float,
        load_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Assemble ``summary["chaos"]``: retries, recovery, fault windows and causes."""
        spec = self.spec
        end_s = max(wall_s, spec.duration_s)
        self._attribute_recoveries(disruptions, tracker.completions, fallback_end_s=end_s)

        per_replica: dict[str, int] = {worker.name: 0 for worker in self._workers}
        attempts = 0
        retries = 0
        for tried in attempts_by_request.values():
            attempts += len(tried)
            retries += max(0, len(tried) - 1)
            for name in tried:
                per_replica[name] = per_replica.get(name, 0) + 1
        unrouted = sum(1 for record in records if record.request_id not in attempts_by_request)

        windows = _merge_windows(
            disruption.window(fallback_end_s=end_s) for disruption in disruptions
        )
        impaired, steady_state = _split_by_windows(records, windows, t0_ns=t0_ns)
        recoveries = [
            disruption.recovery_s for disruption in disruptions if disruption.recovery_s is not None
        ]

        recorded = {record.request_id for record in records}
        completions_by_backend: dict[str, dict[str, int]] = {}
        for completion in tracker.completions:
            if completion.request_id not in recorded:
                # A request torn down with the run (cancelled at the deadline) has no record,
                # so counting its completion here would disagree with the summary above.
                continue
            bucket = completions_by_backend.setdefault(
                completion.backend or "unrouted", {"ok": 0, "failed": 0}
            )
            bucket["ok" if completion.ok else "failed"] += 1

        return {
            "schedule": spec.faults.to_dict(),
            "replicas": len(self._workers),
            "mode": spec.mode,
            "replica_engine": "mock",
            "load": dict(load_config),
            "attempts": attempts,
            "retries": retries,
            "retry_rate": retries / len(records) if records else 0.0,
            "requests_never_routed": unrouted,
            "attempts_per_replica": per_replica,
            "completions_per_replica": completions_by_backend,
            "fault_events": [event.to_dict() for event in events],
            "disruptions": [disruption.to_dict() for disruption in disruptions],
            "recovery_s": Percentiles.from_values(recoveries).to_dict(),
            "disruptions_observed_recovered": len(recoveries),
            "disruptions_not_observed_recovered": len(disruptions) - len(recoveries),
            "impaired_windows_s": [list(window) for window in windows],
            "impaired_seconds": sum(end - start for start, end in windows),
            "fleet_outage_windows_s": [list(window) for window in outages],
            "requests_during_faults": len(impaired),
            "during_faults": summarize_records(impaired, slo=spec.slo, scenario="chaos"),
            "steady_state": summarize_records(steady_state, slo=spec.slo, scenario="chaos"),
            "failures_by_cause": _failures_by_cause(records),
            "workers": [worker.snapshot() for worker in self._workers],
        }

    @staticmethod
    def _attribute_recoveries(
        disruptions: Sequence[Disruption],
        completions: Sequence[_Completion],
        *,
        fallback_end_s: float,
    ) -> None:
        """Fill in each disruption's ``recovered_s`` from the observed completions.

        A disruption counts as recovered at the first *successful* request served by that
        replica after it was brought back. Requests that succeeded elsewhere prove the fleet
        coped; only one served by the restored replica proves the replica is back in
        rotation, which is the thing that was broken.
        """
        ordered = sorted(completions, key=lambda completion: completion.t_s)
        for disruption in disruptions:
            floor = disruption.restored_s
            if floor is None:
                continue
            for completion in ordered:
                if (
                    completion.ok
                    and completion.backend == disruption.target
                    and completion.t_s >= floor
                    and completion.t_s <= fallback_end_s
                ):
                    disruption.recovered_s = completion.t_s
                    break


def _merge_windows(windows: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals, so an impaired second is counted once."""
    ordered = sorted((start, end) for start, end in windows if end > start)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _split_by_windows(
    records: Sequence[RequestRecord],
    windows: Sequence[tuple[float, float]],
    *,
    t0_ns: int,
) -> tuple[list[RequestRecord], list[RequestRecord]]:
    """Partition records into those sent inside an impaired window and the rest.

    Classified by *send* time rather than completion time: a request that arrived while the
    fleet was whole and finished after a replica died was served by a healthy system for the
    part that matters, and charging its latency to the fault would overstate the damage.

    ``t0_ns`` is the run's origin -- the same one the fault timeline is scheduled against --
    so a window boundary and a send timestamp are measured from the same instant.
    """
    if not windows or not records:
        return [], list(records)
    inside: list[RequestRecord] = []
    outside: list[RequestRecord] = []
    for record in records:
        offset_s = (record.t_send_ns - t0_ns) / NS_PER_S
        if any(start <= offset_s < end for start, end in windows):
            inside.append(record)
        else:
            outside.append(record)
    return inside, outside


def _failures_by_cause(records: Sequence[RequestRecord]) -> dict[str, int]:
    """Count failures by a normalised form of their error message.

    The message carries a replica name and often an address; grouping on the raw string
    would produce one bucket per failure and say nothing. What matters is the *kind* of
    failure, because "aborted mid-stream" and "no healthy replica" are two different
    verdicts on the gateway.
    """
    causes: dict[str, int] = {}
    for record in records:
        if record.ok:
            continue
        cause = _cause_of(record.error)
        causes[cause] = causes.get(cause, 0) + 1
    return dict(sorted(causes.items(), key=lambda item: (-item[1], item[0])))


def _cause_of(error: str | None) -> str:
    """Bucket one error message into a cause."""
    if not error:
        return "unknown"
    text = error.lower()
    for needle, cause in (
        ("killed with a stream in flight", "killed_mid_stream"),
        ("no healthy backend", "no_healthy_replica"),
        ("is draining", "draining"),
        ("is partitioned", "partitioned"),
        ("is down", "replica_down"),
        ("injected pre-first-token fault", "injected_error"),
        ("timed out", "timeout"),
        ("unreachable", "unreachable"),
        ("interrupted", "stream_interrupted"),
    ):
        if needle in text:
            return cause
    return "other"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

chaos_app = typer.Typer(
    name="chaos",
    help="Break replicas on purpose and measure what the gateway manages to hide.",
    no_args_is_help=True,
)

_FAULT_HELP = (
    "Fault specification, repeatable: kill:every=10s[,grace=5s,restart=2s,target=replica-0], "
    "latency:p=0.05,ms=500, error:p=0.01, partition:at=20s,for=5s."
)


def _schedule(specs: Sequence[str]) -> FaultSchedule:
    """Parse ``--faults`` values, turning a parse error into a CLI error."""
    try:
        return FaultSchedule.parse(specs)
    except FaultError as exc:
        raise typer.BadParameter(str(exc), param_hint="--faults") from exc


def _duration(text: str, *, param: str) -> float:
    """Parse a duration option, turning a parse error into a CLI error."""
    try:
        return parse_duration(text)
    except FaultError as exc:
        raise typer.BadParameter(str(exc), param_hint=param) from exc


@chaos_app.command("run")
def run_command(
    replicas: Annotated[
        int | None, typer.Option("--replicas", min=1, help="Replicas to run.")
    ] = None,
    duration: Annotated[
        str | None, typer.Option("--duration", help="How long to drive load, e.g. 60s.")
    ] = None,
    rps: Annotated[
        float | None, typer.Option("--rps", min=0.0, help="Open-loop arrival rate.")
    ] = None,
    faults: Annotated[list[str] | None, typer.Option("--faults", help=_FAULT_HELP)] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help=(
                "'subprocess' runs each replica as a real HTTP server that can be SIGKILLed; "
                "'inprocess' keeps them in this process, which is faster but cannot exercise "
                "the HTTP client's error handling."
            ),
        ),
    ] = "subprocess",
    model: Annotated[str, typer.Option("--model", help="Model name the replicas serve.")] = (
        "mock-model"
    ),
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed for arrivals, prompts and victims.")
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help=(
                "Bench profile (configs/bench/profiles.yaml) supplying the fleet size, rate, "
                "duration, request shape and kill cadence. Explicit flags override it."
            ),
        ),
    ] = None,
    ttft_ms: Annotated[float, typer.Option("--ttft-ms", min=0.0)] = 20.0,
    itl_ms: Annotated[float, typer.Option("--itl-ms", min=0.0)] = 5.0,
    max_attempts: Annotated[int, typer.Option("--max-attempts", min=1)] = 3,
    health_ttl_s: Annotated[float, typer.Option("--health-ttl", min=0.0)] = 1.0,
    base_port: Annotated[int, typer.Option("--base-port", min=1, max=65535)] = 9100,
    out: Annotated[
        Path | None, typer.Option("--out", help="Result file; defaults to results/chaos/<ts>.json.")
    ] = None,
    slo_ttft_ms: Annotated[float | None, typer.Option("--slo-ttft-ms", min=0.0)] = None,
    slo_e2e_ms: Annotated[float | None, typer.Option("--slo-e2e-ms", min=0.0)] = None,
) -> None:
    """Run a chaos experiment and write its result file.

    A profile supplies the fleet size, rate, duration and request shape; any flag given
    explicitly wins over it, which is why the four options it can fill in default to None
    rather than to a number.
    """
    from rich.console import Console
    from rich.table import Table

    from turboserve.config import get_settings

    spec_kwargs: dict[str, Any] = {}
    if profile:
        spec_kwargs = _profile_defaults(profile)
        if not faults:
            faults = [f"kill:every={spec_kwargs.pop('_fault_interval_s'):g}s"]
        else:
            spec_kwargs.pop("_fault_interval_s", None)
    if replicas is not None:
        spec_kwargs["replicas"] = replicas
    if duration is not None:
        spec_kwargs["duration_s"] = _duration(duration, param="--duration")
    if rps is not None:
        spec_kwargs["rate_rps"] = rps
    if seed is not None:
        spec_kwargs["seed"] = seed
    if slo_ttft_ms is not None or slo_e2e_ms is not None:
        spec_kwargs["slo"] = SLO(ttft_ms=slo_ttft_ms, e2e_ms=slo_e2e_ms)

    price = os.environ.get("TURBOSERVE_GPU_PRICE_PER_HOUR")
    try:
        spec = ChaosSpec(
            faults=_schedule(faults or []),
            mode=_worker_mode(mode),
            model=model,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            max_attempts=max_attempts,
            health_ttl_s=health_ttl_s,
            base_port=base_port,
            gpu_price_per_hour=float(price) if price else None,
            price_source=os.environ.get("TURBOSERVE_GPU_PRICE_SOURCE"),
            **spec_kwargs,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    report = asyncio.run(ChaosHarness(spec).run())
    target = out if out is not None else default_result_path(get_settings().results_dir)
    report.save(target)

    console = Console()
    table = Table(title=f"chaos run ({spec.faults.describe() or 'no faults'})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("requests", str(report.num_requests))
    table.add_row("failed", str(report.num_failed))
    table.add_row("error rate", f"{report.error_rate:.4%}")
    table.add_row("retries", str(report.retries))
    table.add_row("disruptions", str(len(report.disruptions)))
    table.add_row("recovered", str(len(report.recoveries_s)))
    table.add_row("requests during faults", str(report.chaos.get("requests_during_faults", 0)))
    table.add_row("wall seconds", f"{report.run.summary.get('wall_s', 0.0):.1f}")
    console.print(table)
    console.print(f"wrote {report.path}")


def _worker_mode(text: str) -> WorkerMode:
    """Validate ``--mode`` and narrow it to the literal the spec expects."""
    if text == "inprocess":
        return "inprocess"
    if text == "subprocess":
        return "subprocess"
    raise typer.BadParameter(
        f"--mode must be 'inprocess' or 'subprocess', got {text!r}", param_hint="--mode"
    )


def _profile_defaults(name: str) -> dict[str, Any]:
    """Spec arguments taken from a bench profile's ``chaos`` sizing.

    The profile's *model* is deliberately not used: the replicas here are mock servers, and
    naming a checkpoint in a run that never loaded one would misdescribe the result file.
    Only the shape of the load comes from the profile.
    """
    from turboserve.bench.profiles import load_profile

    try:
        loaded = load_profile(name)
    except KeyError as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from exc
    chaos_profile = loaded.scenarios.chaos
    return {
        "replicas": chaos_profile.num_workers,
        "duration_s": chaos_profile.duration_s,
        "rate_rps": chaos_profile.rate_rps,
        "input_tokens": chaos_profile.input_tokens.as_tuple(),
        "output_tokens": chaos_profile.output_tokens.as_tuple(),
        "seed": loaded.seed,
        "profile": name,
        "slo": chaos_profile.slo.to_slo() if chaos_profile.slo is not None else None,
        "_fault_interval_s": chaos_profile.fault_interval_s,
    }


@chaos_app.command("plan")
def plan_command(
    replicas: Annotated[int, typer.Option("--replicas", min=1)] = 3,
    duration: Annotated[str, typer.Option("--duration")] = "60s",
    faults: Annotated[list[str] | None, typer.Option("--faults", help=_FAULT_HELP)] = None,
    seed: Annotated[int, typer.Option("--seed")] = 20260916,
) -> None:
    """Print the fault timeline a schedule expands to, without running anything."""
    from rich.console import Console
    from rich.table import Table

    schedule = _schedule(faults or [])
    duration_s = _duration(duration, param="--duration")
    names = [f"replica-{index}" for index in range(replicas)]
    try:
        events = schedule.events(duration_s=duration_s, workers=names, seed=seed)
    except FaultError as exc:
        raise typer.BadParameter(str(exc), param_hint="--faults") from exc

    console = Console()
    table = Table(title=f"fault timeline over {duration_s:g}s on {replicas} replica(s)")
    for column in ("t (s)", "action", "target", "from"):
        table.add_column(column)
    for event in events:
        table.add_row(f"{event.t_s:.3f}", str(event.action), event.target, event.spec)
    console.print(table)

    steady = schedule.steady()
    if not steady.is_empty:
        console.print(
            f"steady faults: error p={steady.error_probability:g}, "
            f"latency p={steady.latency_probability:g} of {steady.latency_ms:g} ms"
        )
    outages = outage_windows(events, names, duration_s=duration_s)
    if outages:
        console.print(
            "[bold red]warning[/]: every replica is out of service during "
            + ", ".join(f"[{start:g}s, {end:g}s)" for start, end in outages)
        )
