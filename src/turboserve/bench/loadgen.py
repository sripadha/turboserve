"""The load generator: when to send requests, and what to record about each one.

Two drivers, because they answer different questions and a benchmark that only has one of
them is easy to mislead yourself with:

* **Closed loop** (:func:`run_closed_loop`) keeps ``concurrency`` requests in flight and
  sends the next one only when a previous one finishes. It measures a system's capacity:
  throughput at a fixed load level, with latency that cannot run away because the client
  slows down with the server. This is the driver for the batching, prefix-cache, speculative
  and adapter scenarios, where the comparison is "same load, which configuration serves it
  better".
* **Open loop** (:func:`run_open_loop`) sends at a fixed average rate with Poisson
  inter-arrival times, whatever the server is doing. It measures a system's behaviour under
  offered load: if the server cannot keep up, queueing latency grows without bound and the
  tail explodes, which is precisely the failure a closed-loop test cannot see. This is the
  driver for the chaos scenario and for any SLO/goodput question.

Poisson arrivals rather than a fixed period because request arrivals in a real service are
memoryless: the bursts a Poisson process produces are what fill a queue, and a metronome
would systematically under-report tail latency.

The drivers are deliberately backend-agnostic. Anything matching
``generate(req) -> AsyncIterator[TokenEvent]`` works -- the reference engine's backend, an
HTTP backend pointed at vLLM, or the mock the chaos harness injects faults into -- because
that is the whole :class:`~turboserve.gateway.backends.protocol.Backend` surface a client
needs. Observation is delegated to :class:`~turboserve.bench.metrics.RecordBuilder`, so the
question "what is TTFT" is answered in one file and not in two drivers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from turboserve.bench.metrics import RecordBuilder
from turboserve.bench.records import RequestRecord
from turboserve.engine.core.types import SamplingParams
from turboserve.gateway.backends.protocol import BackendError, GenerateRequest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence

    from turboserve.bench.prompts import BenchPrompt
    from turboserve.bench.records import RunResult
    from turboserve.gateway.backends.protocol import TokenEvent

logger = logging.getLogger(__name__)

__all__ = [
    "GenerateCallable",
    "LoadResult",
    "LoadSpec",
    "build_requests",
    "load_config",
    "poisson_offsets",
    "run_closed_loop",
    "run_load",
    "run_open_loop",
]

type GenerateCallable = Callable[[GenerateRequest], AsyncIterator[TokenEvent]]
"""What the drivers need from a backend: a call returning an async iterator of events.

A :class:`~turboserve.gateway.backends.protocol.Backend`'s bound ``generate`` matches it,
and so does any plain async generator function, which is what the tests drive the drivers
with.
"""

NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class LoadSpec:
    """How the load is shaped: the driver, its intensity, and when to stop.

    ``mode`` selects the driver; ``concurrency`` is read by the closed loop and
    ``rate_rps`` by the open loop. Both honour ``duration_s`` as a stop condition, which is
    how a run can be bounded by time instead of by request count.
    """

    mode: Literal["closed", "open"] = "closed"
    concurrency: int = 1
    rate_rps: float | None = None
    duration_s: float | None = None
    max_in_flight: int | None = None
    """Optional ceiling on concurrent open-loop requests.

    Leave it ``None`` for a genuine open loop. Setting it turns the generator closed-loop
    once the ceiling is reached (arrivals block), which hides queueing latency behind
    client-side backpressure -- useful only as a safety valve against a hung backend
    accumulating unbounded tasks, and recorded in the run config when it is used.
    """

    request_timeout_s: float | None = None
    """Per-request deadline; on expiry the stream is aborted and the request counts failed."""

    repeat_requests: bool = False
    """Cycle through the request pool when it runs out; only meaningful with ``duration_s``."""

    seed: int = 0
    lane: str = "stable"
    backend_name: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("closed", "open"):
            raise ValueError(f"mode must be 'closed' or 'open', got {self.mode!r}")
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")
        if self.mode == "open" and (self.rate_rps is None or self.rate_rps <= 0):
            raise ValueError("open-loop load needs a positive rate_rps")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.max_in_flight is not None and self.max_in_flight < 1:
            raise ValueError(f"max_in_flight must be >= 1, got {self.max_in_flight}")
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError(f"request_timeout_s must be positive, got {self.request_timeout_s}")
        if self.repeat_requests and self.duration_s is None:
            raise ValueError("repeat_requests needs duration_s, or the run would never end")


@dataclass(slots=True)
class LoadResult:
    """What one load phase produced: the records, the window, and the load actually applied.

    ``max_in_flight_observed`` is reported because the *intended* concurrency and the
    achieved one differ whenever the client is the bottleneck, and a run whose peak
    in-flight count never reached the configured concurrency measured something other than
    what its config says.
    """

    records: list[RequestRecord] = field(default_factory=list)
    warmup_records: list[RequestRecord] = field(default_factory=list)
    started_ns: int = 0
    finished_ns: int = 0
    num_sent: int = 0
    max_in_flight_observed: int = 0

    @property
    def wall_seconds(self) -> float:
        """Seconds from the first send to the last completion of the measured phase."""
        return max(self.finished_ns - self.started_ns, 0) / NS_PER_S

    @property
    def num_failed(self) -> int:
        """Measured requests that did not complete successfully."""
        return sum(1 for record in self.records if not record.ok)

    def into(self, run: RunResult) -> RunResult:
        """Append the measured records to a run (warmup records are not published)."""
        for record in self.records:
            run.add(record)
        return run


def poisson_offsets(rate_rps: float, count: int, *, seed: int = 0) -> list[float]:
    """Arrival times, in seconds from the start, of a Poisson process at ``rate_rps``.

    The first arrival is at zero and each subsequent gap is drawn from
    ``Exponential(rate)``; the process is therefore conditioned on an arrival at the origin
    rather than idling for one mean gap before the run begins. The gaps -- which are what
    the burstiness comes from -- are exactly those of a Poisson process.

    The same ``random.Random(seed)`` stream is consumed by :func:`run_open_loop`, so this
    function predicts the schedule that driver will follow.
    """
    if rate_rps <= 0:
        raise ValueError(f"rate_rps must be positive, got {rate_rps}")
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    rng = random.Random(seed)
    offsets: list[float] = []
    elapsed = 0.0
    for _ in range(count):
        offsets.append(elapsed)
        elapsed += rng.expovariate(rate_rps)
    return offsets


def build_requests(
    prompts: Sequence[BenchPrompt],
    *,
    model: str,
    sampling: SamplingParams | None = None,
    send_text: bool = False,
    lora: str | Callable[[BenchPrompt], str | None] | None = None,
    stream: bool = True,
    priority: int = 0,
    id_prefix: str = "",
) -> list[GenerateRequest]:
    """Turn prompts into gateway requests.

    Defaults chosen for measurement rather than for pleasant output: greedy sampling, so
    two engines given the same prompt do the same amount of work, and ``ignore_eos`` so
    every request produces exactly the number of output tokens the profile asked for. A run
    that stopped early on an end-of-sequence token would report a throughput computed over
    a workload nobody specified. Pass an explicit ``sampling`` to override, and note that
    ``max_tokens`` is then the same for every request instead of coming from the prompt.

    Token ids are sent by default. Text is sent only when ``send_text`` is set, for backends
    that cannot accept ids -- and never for the prefix-cache scenario, where re-tokenising
    the text can merge tokens across the prefix boundary and change what is shared.

    ``lora`` is either a constant adapter name or a function of the prompt, which is how the
    multi-adapter scenario spreads N adapters over the request stream.
    """
    requests: list[GenerateRequest] = []
    for prompt in prompts:
        if prompt.num_prompt_tokens == 0 and not send_text:
            raise ValueError(f"prompt {prompt.prompt_id!r} has no token ids to send")
        payload: str | list[int] = prompt.text if send_text else list(prompt.token_ids)
        adapter = lora(prompt) if callable(lora) else lora
        requests.append(
            GenerateRequest(
                request_id=f"{id_prefix}{prompt.prompt_id}",
                tenant_id=prompt.tenant,
                model=model,
                prompt=payload,
                sampling=sampling
                or SamplingParams(max_tokens=prompt.max_tokens, temperature=0.0, ignore_eos=True),
                lora=adapter,
                priority=priority,
                stream=stream,
            )
        )
    return requests


class _InFlight:
    """Counts concurrent requests and remembers the peak.

    No lock: asyncio runs one task at a time in a thread, and every mutation below happens
    between awaits.
    """

    __slots__ = ("current", "peak")

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        """Register one more request in flight."""
        self.current += 1
        self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        """Register one fewer request in flight."""
        self.current -= 1


async def _consume(
    request: GenerateRequest,
    generate: GenerateCallable,
    builder: RecordBuilder,
) -> bool:
    """Drain one stream into ``builder``; returns whether a terminating event arrived.

    The iterator is always closed, including on timeout or cancellation: the
    :class:`Backend` contract says that closing the iterator aborts the work behind it, and
    a load generator that abandoned streams instead of closing them would leave the engine
    decoding for a client that stopped listening, corrupting every subsequent measurement.
    """
    stream = generate(request)
    saw_final = False
    try:
        async for event in stream:
            builder.observe(event)
            saw_final = event.finished
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()
    return saw_final


async def _drive_one(
    request: GenerateRequest,
    generate: GenerateCallable,
    spec: LoadSpec,
    in_flight: _InFlight,
    prompt_tokens: int,
) -> RequestRecord:
    """Send one request, observe its stream, and return the record for it.

    Every failure mode becomes a failed record rather than an exception: one backend error
    in a thousand-request run must not lose the other nine hundred and ninety-nine, and the
    error rate is itself a headline metric. Cancellation is the exception -- it means the
    whole run is being torn down, so it is recorded and re-raised.
    """
    builder = RecordBuilder(
        request.request_id,
        tenant=request.tenant_id,
        prompt_tokens=prompt_tokens,
        backend=spec.backend_name,
        lane=spec.lane,
    )
    in_flight.enter()
    builder.sent()
    try:
        if spec.request_timeout_s is None:
            saw_final = await _consume(request, generate, builder)
        else:
            async with asyncio.timeout(spec.request_timeout_s):
                saw_final = await _consume(request, generate, builder)
        if not saw_final:
            builder.failed("stream ended without a terminating event")
    except TimeoutError:
        builder.failed(f"timed out after {spec.request_timeout_s}s")
    except asyncio.CancelledError:
        builder.failed("cancelled")
        raise
    except BackendError as exc:
        builder.failed(str(exc))
    except Exception as exc:  # noqa: BLE001 - one request's failure is data, not a crash
        logger.debug("request %s failed", request.request_id, exc_info=True)
        builder.failed(f"{type(exc).__name__}: {exc}")
    finally:
        in_flight.leave()
    return builder.build()


def _prompt_tokens_for(request: GenerateRequest, overrides: Mapping[str, int] | None) -> int:
    """Prompt length to seed the record with, before the backend's usage block overrides it."""
    if overrides is not None and request.request_id in overrides:
        return overrides[request.request_id]
    ids = request.prompt_token_ids
    return len(ids) if ids is not None else 0


async def run_closed_loop(
    requests: Sequence[GenerateRequest],
    generate: GenerateCallable,
    *,
    spec: LoadSpec,
    prompt_tokens: Mapping[str, int] | None = None,
    on_record: Callable[[RequestRecord], None] | None = None,
) -> LoadResult:
    """Keep ``spec.concurrency`` requests in flight until the pool or the clock runs out.

    Workers pull from a shared cursor rather than being handed a fixed slice each, so one
    slow request delays only its own worker instead of leaving a whole slice unsent.
    """
    result = LoadResult()
    if not requests:
        return result
    in_flight = _InFlight()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + spec.duration_s if spec.duration_s is not None else None
    cursor = 0
    collected: list[RequestRecord] = []

    async def worker() -> None:
        nonlocal cursor
        while True:
            if cursor >= len(requests):
                return
            if deadline is not None and loop.time() >= deadline:
                return
            index = cursor
            cursor += 1
            request = requests[index]
            record = await _drive_one(
                request,
                generate,
                spec,
                in_flight,
                _prompt_tokens_for(request, prompt_tokens),
            )
            collected.append(record)
            if on_record is not None:
                on_record(record)

    result.started_ns = time.monotonic_ns()
    workers = [asyncio.create_task(worker()) for _ in range(spec.concurrency)]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    result.finished_ns = time.monotonic_ns()
    result.records = collected
    result.num_sent = len(collected)
    result.max_in_flight_observed = in_flight.peak
    return result


async def run_open_loop(
    requests: Sequence[GenerateRequest],
    generate: GenerateCallable,
    *,
    spec: LoadSpec,
    prompt_tokens: Mapping[str, int] | None = None,
    on_record: Callable[[RequestRecord], None] | None = None,
) -> LoadResult:
    """Send at ``spec.rate_rps`` with Poisson inter-arrival times, regardless of progress.

    Arrival times are computed as absolute offsets from the start of the run and slept to,
    rather than as a sleep of one gap after each launch: the latter accumulates the launch
    overhead into the schedule, so a long run would drift to a lower effective rate than
    the one its result file claims.
    """
    result = LoadResult()
    rate = spec.rate_rps
    if rate is None or rate <= 0:  # pragma: no cover - LoadSpec validates this
        raise ValueError("open-loop load needs a positive rate_rps")
    if not requests:
        return result
    in_flight = _InFlight()
    loop = asyncio.get_running_loop()
    rng = random.Random(spec.seed)
    semaphore = asyncio.Semaphore(spec.max_in_flight) if spec.max_in_flight else None
    collected: list[RequestRecord] = []
    tasks: list[asyncio.Task[RequestRecord]] = []

    async def launch(request: GenerateRequest) -> RequestRecord:
        try:
            record = await _drive_one(
                request,
                generate,
                spec,
                in_flight,
                _prompt_tokens_for(request, prompt_tokens),
            )
        finally:
            if semaphore is not None:
                semaphore.release()
        collected.append(record)
        if on_record is not None:
            on_record(record)
        return record

    start = loop.time()
    result.started_ns = time.monotonic_ns()
    deadline = start + spec.duration_s if spec.duration_s is not None else None
    offset = 0.0
    index = 0
    try:
        while spec.repeat_requests or index < len(requests):
            target = start + offset
            if deadline is not None and target >= deadline:
                break
            delay = target - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            if semaphore is not None:
                await semaphore.acquire()
            tasks.append(asyncio.create_task(launch(_nth_request(requests, index))))
            index += 1
            offset += rng.expovariate(rate)
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    result.finished_ns = time.monotonic_ns()
    result.records = collected
    result.num_sent = index
    result.max_in_flight_observed = in_flight.peak
    return result


def _nth_request(requests: Sequence[GenerateRequest], index: int) -> GenerateRequest:
    """The ``index``-th request, cycling the pool with a distinct id on each repetition.

    Ids must stay unique across a run: they are the join key between the records, the
    engine's logs and the gateway's metrics, and a duplicated id would make two different
    requests indistinguishable in the result file.
    """
    if index < len(requests):
        return requests[index]
    base = requests[index % len(requests)]
    return base.model_copy(
        update={
            "request_id": f"{base.request_id}#{index // len(requests)}",
            "arrival_ts": time.perf_counter(),
        }
    )


async def run_load(
    requests: Sequence[GenerateRequest],
    generate: GenerateCallable,
    *,
    spec: LoadSpec,
    warmup: Sequence[GenerateRequest] = (),
    prompt_tokens: Mapping[str, int] | None = None,
    on_record: Callable[[RequestRecord], None] | None = None,
) -> LoadResult:
    """Run the warmup phase, then the measured phase with the driver ``spec`` names.

    The warmup is always closed-loop and its records are kept separately: they exist to pay
    for lazy CUDA context creation, kernel autotuning and the first tokenizer call, and
    including them would put a one-off multi-second outlier into the run's tail latency.
    """
    warmup_records: list[RequestRecord] = []
    if warmup:
        warmup_spec = LoadSpec(
            mode="closed",
            concurrency=min(spec.concurrency, len(warmup)),
            request_timeout_s=spec.request_timeout_s,
            seed=spec.seed,
            lane=spec.lane,
            backend_name=spec.backend_name,
        )
        warmup_result = await run_closed_loop(
            warmup, generate, spec=warmup_spec, prompt_tokens=prompt_tokens
        )
        warmup_records = warmup_result.records
        logger.debug("warmup finished: %d requests", len(warmup_records))
    if spec.mode == "open":
        result = await run_open_loop(
            requests, generate, spec=spec, prompt_tokens=prompt_tokens, on_record=on_record
        )
    else:
        result = await run_closed_loop(
            requests, generate, spec=spec, prompt_tokens=prompt_tokens, on_record=on_record
        )
    result.warmup_records = warmup_records
    return result


def load_config(spec: LoadSpec, **extra: Any) -> dict[str, Any]:
    """The load-generator settings a scenario embeds in its result file's ``config``."""
    block: dict[str, Any] = {
        "mode": spec.mode,
        "concurrency": spec.concurrency,
        "rate_rps": spec.rate_rps,
        "duration_s": spec.duration_s,
        "max_in_flight": spec.max_in_flight,
        "request_timeout_s": spec.request_timeout_s,
        "repeat_requests": spec.repeat_requests,
        "seed": spec.seed,
        "lane": spec.lane,
        "backend": spec.backend_name,
    }
    block.update(extra)
    return block
