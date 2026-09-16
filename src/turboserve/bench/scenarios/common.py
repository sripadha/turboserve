"""Shared machinery for the benchmark scenarios.

Every scenario in this package is the same five steps -- pick the workload from a profile,
build one or more *arms* (a backend plus the configuration that distinguishes it), send the
same seeded prompts to each, record what came back, write one result file per arm -- and
only the middle step differs between them. That shape lives here so that a scenario module
is a description of an experiment rather than a re-implementation of the harness, and so
that two scenarios cannot drift apart in how they name an arm, seed a prompt set or size a
KV pool.

Three things in particular are centralised because getting them wrong silently changes what
a result means:

``BaselineBackend``
    The ``transformers`` baselines have no streaming interface and no async one; this
    adapter gives them the gateway's :class:`~turboserve.gateway.backends.protocol.Backend`
    surface so the *same* load generator drives them and the reference engine. It is also
    the place where the baselines' one honest limitation is documented: a blocking
    ``generate`` call produces nothing until it is finished, so a client observes the whole
    completion at once and its time-to-first-token is its end-to-end time.

``build_prompt_pool``
    One seeded prompt set, reused by every arm of a scenario, sent as token ids. Two arms
    that drew different prompts are not comparable, and a prompt re-tokenised from text can
    lose the prefix sharing a prefix-cache experiment is trying to measure.

``open_run`` / ``write_run``
    The result-file conventions the report renderer reads: ``config["label"]``,
    ``config["backend"]``, ``config["baseline_label"]``, ``config["load"]`` and
    ``summary["derived"]``. A scenario that invented its own keys would render as an
    unlabelled row with no comparison column.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from turboserve.bench.loadgen import LoadSpec, load_config
from turboserve.bench.prompts import BenchPrompt
from turboserve.bench.records import SLO, RunResult
from turboserve.engine.core.types import EngineConfig, FinishReason, SchedulerConfig
from turboserve.gateway.backends.protocol import (
    BackendRequestError,
    BackendUnavailableError,
    GenerateRequest,
    TokenEvent,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator, Mapping, Sequence

    from turboserve.bench.profiles import SLOSpec
    from turboserve.engine.core.types import RequestOutput
    from turboserve.engine.runtime.naive import BaselineEngine
    from turboserve.gateway.backends.local_engine import LocalEngineBackend
    from turboserve.gateway.backends.openai_compat import OpenAICompatBackend

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TENANT",
    "AnyBackend",
    "DEFAULT_VOCAB_SIZE",
    "GPU_PRICE_ENV",
    "GPU_PRICE_SOURCE_ENV",
    "REMOTE_ENGINE_LABELS",
    "ArmOutcome",
    "BaselineBackend",
    "BaselineKind",
    "EngineOptions",
    "ScenarioError",
    "arm_table",
    "baseline_backend",
    "build_engine_config",
    "build_load_spec",
    "build_prompt_pool",
    "close_backend",
    "default_results_dir",
    "engine_derived_stats",
    "gpu_price",
    "load_tokenizer",
    "normalise_base_url",
    "openai_backend",
    "open_run",
    "reference_backend",
    "remote_server_info",
    "resolve_slo",
    "result_path",
    "run_timestamp",
    "write_run",
]

#: Tenant used when a profile names none (profiles always do; the default keeps the
#: helpers usable from a test that builds a prompt pool directly).
DEFAULT_TENANT = "bench"

#: Fallback vocabulary ceiling for tokenizer-free prompts. Small enough to be inside every
#: checkpoint this repository serves, so a pool built without a tokenizer is still a valid
#: input to a real model rather than a stream of out-of-range ids.
DEFAULT_VOCAB_SIZE = 32_000

#: Environment variables ``scripts/vastai/run_remote.sh`` exports on the measurement host.
GPU_PRICE_ENV = "TURBOSERVE_GPU_PRICE_PER_HOUR"
GPU_PRICE_SOURCE_ENV = "TURBOSERVE_GPU_PRICE_SOURCE"

#: The production engines a scenario can measure over an OpenAI-compatible URL, and how
#: each is named in a rendered table. They are *arms*, not backend types: all of them are
#: served by :class:`~turboserve.gateway.backends.openai_compat.OpenAICompatBackend` over
#: the same protocol, and the name exists so that a result file says which server produced
#: it and two engines writing into one results directory do not render as one arm measured
#: twice. Scenarios that take a ``--url`` build their arm names from this mapping, so an
#: engine is added here once rather than in each scenario.
REMOTE_ENGINE_LABELS: dict[str, str] = {
    "vllm": "vLLM",
    "sglang": "SGLang",
}

BaselineKind = Literal["naive_hf", "static_batch"]
"""Which ``transformers`` baseline to build: one request per call, or fixed-size batches."""

type AnyBackend = LocalEngineBackend | OpenAICompatBackend | BaselineBackend
"""The three things a scenario can be served by.

A union rather than the :class:`~turboserve.gateway.backends.protocol.Backend` protocol:
that protocol declares ``supports_lora`` as a settable attribute, which a backend exposing
it as a read-only property (:class:`LocalEngineBackend` does, because it is derived from
the configured adapter map) does not structurally satisfy. The union keeps the scenarios
precisely typed without weakening the protocol every backend in the gateway is checked
against.
"""


class ScenarioError(RuntimeError):
    """A scenario cannot run as configured (bad arm, unreachable server, missing model)."""


# ---------------------------------------------------------------------------------------
# result-file plumbing
# ---------------------------------------------------------------------------------------


def default_results_dir() -> Path:
    """Where result files go unless the caller says otherwise.

    Read from :class:`~turboserve.config.Settings` so that ``TURBOSERVE_RESULTS_DIR``
    relocates every scenario at once -- the measurement host writes to a volume that is
    rsynced back, not into the checkout.
    """
    from turboserve.config import get_settings

    return get_settings().results_dir


def run_timestamp(now: datetime | None = None) -> str:
    """Compact UTC stamp used as a result file's name."""
    moment = now if now is not None else datetime.now(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def result_path(
    scenario: str,
    results_dir: Path | str | None = None,
    *,
    label: str = "",
    now: datetime | None = None,
) -> Path:
    """``<results_dir>/<scenario>/<timestamp>[-<label>].json``.

    The label is in the file name as well as in the payload because a scenario writes one
    file per arm inside the same second, and a name collision would have one arm silently
    overwrite another.
    """
    base = Path(results_dir) if results_dir is not None else default_results_dir()
    stem = run_timestamp(now)
    if label:
        stem = f"{stem}-{_slug(label)}"
    return base / scenario / f"{stem}.json"


def _slug(text: str) -> str:
    """Filesystem-safe form of an arm label."""
    kept = [char if char.isalnum() else "-" for char in text.strip().lower()]
    slug = "".join(kept)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "arm"


def gpu_price() -> tuple[float | None, str | None]:
    """The GPU price and its source, as exported by the provisioning scripts.

    A result without a price simply has no cost column; a result with a *wrong* price would
    put a fabricated dollar figure in a table, so an unparsable value is dropped with a
    warning rather than guessed at.
    """
    raw = os.environ.get(GPU_PRICE_ENV)
    if not raw:
        return None, None
    try:
        price = float(raw)
    except ValueError:
        logger.warning("ignoring unparsable %s=%r", GPU_PRICE_ENV, raw)
        return None, None
    return price, os.environ.get(GPU_PRICE_SOURCE_ENV)


def resolve_slo(
    profile_slo: SLOSpec | None,
    *,
    ttft_ms: float | None = None,
    tpot_ms: float | None = None,
    e2e_ms: float | None = None,
) -> SLO | None:
    """Merge the profile's objective with command-line overrides.

    Returns ``None`` when no objective is asserted at all, which is what tells
    :meth:`RunResult.summarize` to leave the goodput block empty rather than score against
    an objective nobody set.
    """
    base = profile_slo.to_slo() if profile_slo is not None else SLO()
    merged = SLO(
        ttft_ms=ttft_ms if ttft_ms is not None else base.ttft_ms,
        tpot_ms=tpot_ms if tpot_ms is not None else base.tpot_ms,
        e2e_ms=e2e_ms if e2e_ms is not None else base.e2e_ms,
    )
    if merged.ttft_ms is None and merged.tpot_ms is None and merged.e2e_ms is None:
        return None
    return merged


def open_run(
    scenario: str,
    profile: str,
    *,
    label: str,
    backend: str,
    baseline_label: str | None,
    spec: LoadSpec,
    config: Mapping[str, Any] | None = None,
    load_extra: Mapping[str, Any] | None = None,
) -> RunResult:
    """Start a :class:`RunResult` carrying the conventions the report renderer reads.

    ``label`` names the arm in every table; ``baseline_label`` is repeated into *every* arm
    of a scenario so the relative column survives one file being deleted; ``config["load"]``
    is where the plots find the concurrency.
    """
    block: dict[str, Any] = dict(config or {})
    block.update(
        {
            "label": label,
            "backend": backend,
            "load": load_config(spec, **dict(load_extra or {})),
        }
    )
    if baseline_label is not None:
        block["baseline_label"] = baseline_label
    price, source = gpu_price()
    return RunResult.start(
        scenario,
        profile,
        config=block,
        gpu_price_per_hour=price,
        price_source=source,
    )


def write_run(
    run: RunResult,
    path: Path,
    *,
    slo: SLO | None = None,
    derived: Mapping[str, Any] | None = None,
) -> Path:
    """Finish, decorate and save a run; returns the path actually written.

    ``derived`` is attached *after* :meth:`RunResult.finish`, because ``finish`` replaces
    ``summary`` wholesale and a scenario figure written before it would be discarded.
    """
    run.finish(slo=slo)
    if derived:
        run.summary["derived"] = dict(derived)
    return run.save(path)


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """One arm's result: what it was, where it landed, and what it measured."""

    label: str
    backend: str
    path: Path
    summary: dict[str, Any] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)

    @property
    def num_requests(self) -> int:
        """Requests the arm recorded."""
        value = self.summary.get("num_requests", 0)
        return int(value) if isinstance(value, int | float) else 0

    @property
    def error_rate(self) -> float:
        """Fraction of the arm's requests that failed."""
        value = self.summary.get("error_rate", 0.0)
        return float(value) if isinstance(value, int | float) else 0.0


def arm_table(title: str, outcomes: Sequence[ArmOutcome]) -> None:
    """Print a short per-arm table to the terminal.

    Deliberately not a results table: it shows how many requests each arm completed and
    where its file went, so an operator can see a run finished. Every published figure comes
    out of the JSON through ``turboserve bench render``.
    """
    from rich.console import Console
    from rich.table import Table

    table = Table(title=title)
    table.add_column("arm")
    table.add_column("backend")
    table.add_column("requests", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("file")
    for outcome in outcomes:
        failed = outcome.summary.get("num_failed", 0)
        table.add_row(
            outcome.label,
            outcome.backend,
            str(outcome.num_requests),
            str(failed),
            str(outcome.path),
        )
    Console().print(table)


# ---------------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------------


def load_tokenizer(model: str, *, local_files_only: bool = False) -> Any | None:
    """The model's tokenizer, or ``None`` when it cannot be loaded.

    ``None`` is a supported outcome, not a failure: a run against a mock server or a remote
    engine whose tokenizer is not on this machine still needs prompts, and
    :func:`build_prompt_pool` falls back to random token ids of the requested length. The
    run then measures the same request *shape* without claiming to have tokenised anything.
    """
    from turboserve.engine.runtime.streaming import get_tokenizer

    try:
        return get_tokenizer(model, local_files_only=local_files_only)
    except Exception as exc:  # noqa: BLE001 - any loader failure is a fallback, not a crash
        logger.warning("no tokenizer for %s (%s); using tokenizer-free prompts", model, exc)
        return None


def build_prompt_pool(
    *,
    count: int,
    input_tokens: tuple[int, int],
    output_tokens: tuple[int, int],
    seed: int,
    tenants: Sequence[str] = (DEFAULT_TENANT,),
    shared_prefix_tokens: int = 0,
    tokenizer: Any | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    id_prefix: str = "synthetic",
) -> list[BenchPrompt]:
    """One seeded prompt pool, shared by every arm of a scenario.

    With a tokenizer this delegates to
    :class:`~turboserve.bench.prompts.SyntheticPromptBuilder`, which guarantees the exact
    token length and carries the decoded text alongside the ids. Without one it draws ids
    directly: the lengths, the shared prefix and the seed are still honoured, so the shape
    of the workload is unchanged, but there is no text view.
    """
    if count < 1:
        raise ScenarioError(f"a prompt pool needs at least one prompt, got {count}")
    if not tenants:
        raise ScenarioError("at least one tenant is required")
    if tokenizer is not None:
        from turboserve.bench.prompts import SyntheticPromptBuilder

        builder = SyntheticPromptBuilder(tokenizer, seed=seed)
        return builder.build_batch(
            count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            shared_prefix_tokens=shared_prefix_tokens,
            tenants=tuple(tenants),
            prefix=id_prefix,
        )
    return _tokenizer_free_pool(
        count=count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        seed=seed,
        tenants=tuple(tenants),
        shared_prefix_tokens=shared_prefix_tokens,
        vocab_size=vocab_size,
        id_prefix=id_prefix,
    )


def _tokenizer_free_pool(
    *,
    count: int,
    input_tokens: tuple[int, int],
    output_tokens: tuple[int, int],
    seed: int,
    tenants: tuple[str, ...],
    shared_prefix_tokens: int,
    vocab_size: int,
    id_prefix: str,
) -> list[BenchPrompt]:
    """Random-id prompts of exact length, with an optional literally shared prefix."""
    low_in, high_in = input_tokens
    low_out, high_out = output_tokens
    if low_in < 1 or high_in < low_in:
        raise ScenarioError(f"input_tokens={input_tokens} must satisfy 1 <= min <= max")
    if low_out < 1 or high_out < low_out:
        raise ScenarioError(f"output_tokens={output_tokens} must satisfy 1 <= min <= max")
    if shared_prefix_tokens >= low_in:
        raise ScenarioError(
            f"shared_prefix_tokens ({shared_prefix_tokens}) must be smaller than the "
            f"shortest prompt ({low_in})"
        )
    if vocab_size < 2:
        raise ScenarioError(f"vocab_size must be at least 2, got {vocab_size}")
    rng = random.Random(seed)
    prefix = [rng.randrange(vocab_size) for _ in range(shared_prefix_tokens)]
    prompts: list[BenchPrompt] = []
    for index in range(count):
        num_input = rng.randint(low_in, high_in)
        num_output = rng.randint(low_out, high_out)
        suffix = [rng.randrange(vocab_size) for _ in range(num_input - shared_prefix_tokens)]
        prompts.append(
            BenchPrompt(
                prompt_id=f"{id_prefix}-{index:05d}",
                token_ids=[*prefix, *suffix],
                text="",
                max_tokens=num_output,
                tenant=tenants[index % len(tenants)],
                shared_prefix_tokens=shared_prefix_tokens,
                source="synthetic",
                round_trip_exact=False,
            )
        )
    return prompts


# ---------------------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineOptions:
    """Engine knobs a scenario exposes on the command line.

    Separated from :class:`~turboserve.engine.core.types.EngineConfig` because a scenario
    needs to build several configurations that differ in exactly one field (prefix caching
    on and off, say) and a frozen options object makes that a ``dataclasses.replace``
    rather than a dictionary of magic keys.
    """

    dtype: str = "auto"
    device: str = "auto"
    seed: int | None = None
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    block_size: int = 16
    num_blocks: int | None = None
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None


def build_engine_config(model: str, options: EngineOptions) -> EngineConfig:
    """An :class:`EngineConfig` for one arm.

    ``num_blocks`` is passed through rather than always profiled: a CPU smoke run would
    otherwise size its KV pool against host RAM and allocate far more than the experiment
    needs, and a test would take seconds it does not have to.
    """
    scheduler = SchedulerConfig(
        max_num_seqs=options.max_num_seqs,
        max_num_batched_tokens=options.max_num_batched_tokens,
        block_size=options.block_size,
        num_blocks=options.num_blocks,
        enable_chunked_prefill=options.enable_chunked_prefill,
        enable_prefix_caching=options.enable_prefix_caching,
    )
    return EngineConfig(
        model=model,
        dtype=options.dtype,  # type: ignore[arg-type]
        device=options.device,  # type: ignore[arg-type]
        seed=options.seed,
        gpu_memory_utilization=options.gpu_memory_utilization,
        max_model_len=options.max_model_len,
        scheduler=scheduler,
    )


def reference_backend(
    config: EngineConfig,
    *,
    name: str = "reference",
    local_files_only: bool = False,
) -> LocalEngineBackend:
    """The from-scratch engine behind the gateway's in-process backend.

    Nothing is loaded here: :class:`LocalEngineBackend` builds the engine on its first
    request, which keeps a scenario's argument validation fast and means a mis-typed arm
    fails before a checkpoint is read.
    """
    from turboserve.gateway.backends.local_engine import LocalEngineBackend

    return LocalEngineBackend(
        name=name,
        config=config,
        local_files_only=local_files_only,
    )


def openai_backend(
    url: str,
    *,
    name: str = "vllm",
    api_key: str | None = None,
    timeout_s: float = 600.0,
) -> OpenAICompatBackend:
    """A backend pointed at an OpenAI-compatible server (vLLM, SGLang, this repo's gateway).

    ``name`` is the arm's name in the result file, not a switch: the same class drives every
    such server, because the request body, the SSE framing and the usage block are the same
    protocol. What the server is gets *recorded* by :func:`remote_server_info`, not assumed
    here.
    """
    from turboserve.gateway.backends.openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(
        normalise_base_url(url),
        name=name,
        api_key=api_key,
        timeout_s=timeout_s,
    )


async def remote_server_info(backend: AnyBackend) -> dict[str, Any]:
    """What a remote server says about itself, for the arm's ``engine`` block.

    A client cannot see the flags a server was started with, and those flags are most of
    what a benchmark number means -- whether the prefix cache was on, what the context
    window was, which speculative algorithm was running. SGLang answers ``/version`` and
    ``/get_server_info``, vLLM answers ``/version``, and anything else answers neither; all
    three cases are fine, because the block is recorded when it exists and omitted when it
    does not.

    Never fails a run: a server that refuses the probe contributes nothing, exactly as a
    backend with no engine counters contributes no counters.
    """
    probe = getattr(backend, "server_info", None)
    if not callable(probe):
        return {}
    try:
        info = await probe()
    except Exception:  # noqa: BLE001 - provenance must never fail a measurement
        logger.debug("backend %r could not report server info", backend, exc_info=True)
        return {}
    return dict(info) if isinstance(info, dict) else {}


def normalise_base_url(url: str) -> str:
    """Accept ``http://host:port`` as well as ``http://host:port/v1``.

    The OpenAI-compatible backend posts to ``<base>/completions``, so the ``/v1`` prefix is
    part of the base. Every other tool in this repository -- the kind end-to-end job, the
    deployment manifests, a human typing a vLLM address -- names a server by its origin, so
    the prefix is appended when it is missing instead of producing a 404 much later.
    """
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        raise ScenarioError("an empty URL cannot be used as a benchmark target")
    if not cleaned.startswith(("http://", "https://")):
        raise ScenarioError(f"URL must start with http:// or https://, got {url!r}")
    tail = cleaned.rsplit("/", 1)[-1]
    if tail.startswith("v") and tail[1:].isdigit():
        return cleaned
    return cleaned + "/v1"


def baseline_backend(
    kind: BaselineKind,
    config: EngineConfig,
    *,
    batch_size: int | None = None,
    local_files_only: bool = False,
    name: str | None = None,
) -> BaselineBackend:
    """Build one of the ``transformers`` baselines wrapped in the backend adapter."""
    from turboserve.engine.runtime.naive import NaiveHFEngine, StaticBatchHFEngine

    if kind == "naive_hf":
        engine: BaselineEngine = NaiveHFEngine(config, local_files_only=local_files_only)
    elif kind == "static_batch":
        engine = StaticBatchHFEngine(
            config,
            batch_size=batch_size,
            flush_partial=True,
            local_files_only=local_files_only,
        )
    else:  # pragma: no cover - the Literal keeps callers honest
        raise ScenarioError(f"unknown baseline {kind!r}")
    return BaselineBackend(engine, name=name or kind)


async def close_backend(backend: AnyBackend | None) -> None:
    """Close a backend, tolerating one that never opened anything."""
    if backend is None:
        return
    with contextlib.suppress(Exception):
        await backend.close()


def engine_derived_stats(backend: AnyBackend) -> dict[str, Any]:
    """The engine counters worth putting in a result's ``derived`` block.

    Only keys the engine actually reported are copied, so an arm served by a remote server
    contributes an empty block rather than a row of zeroes that reads like a measurement.
    """
    stats_fn = getattr(backend, "stats", None)
    if not callable(stats_fn):
        return {}
    try:
        stats = stats_fn()
    except Exception:  # noqa: BLE001 - observability must never fail a run
        logger.debug("backend %r could not report stats", backend, exc_info=True)
        return {}
    if not isinstance(stats, dict):
        return {}
    wanted = (
        "backend",
        "block_size",
        "kv_bytes",
        "kv_utilization",
        "num_batches",
        "num_cached_prompt_tokens",
        "num_generated_tokens",
        "num_kv_blocks",
        "num_preemptions",
        "num_prompt_tokens",
        "num_steps",
        "prefix_hit_rate",
    )
    return {key: stats[key] for key in wanted if key in stats}


class BaselineBackend:
    """Give a blocking ``transformers`` baseline the gateway's streaming backend surface.

    The adapter exists so that one load generator drives every arm of a comparison. It owns
    a single driver task, and the wrapped engine is touched from that task alone: arrivals
    are buffered in an inbox and flushed into the engine between steps, because
    :meth:`BaselineEngine.step` runs in a worker thread (``generate`` is a long blocking
    call and must not hold the event loop) and the engine's queue is not designed for
    concurrent mutation.

    **What this costs the baselines, and why it is honest.** ``transformers.generate``
    returns nothing until the whole completion exists, so exactly one event is emitted per
    request and a client's time-to-first-token equals its end-to-end time. That is what a
    caller of these baselines genuinely experiences; fabricating a per-token trickle from a
    completion that arrived in one piece would invent a TTFT that nothing measured. The
    comparison these arms exist for is throughput, and throughput is unaffected.

    Buffering also gives the static-batch baseline its defining behaviour for free: a
    request that arrives while a batch is generating waits for the next one, which is the
    head-of-line delay continuous batching removes.
    """

    #: Baselines have no adapter support; a request naming one must not be routed here.
    supports_lora = False

    def __init__(
        self,
        engine: BaselineEngine,
        *,
        name: str | None = None,
        batch_window_s: float = 0.01,
    ) -> None:
        """Wrap ``engine``.

        Args:
            engine: the baseline to drive. The adapter owns it and closes it.
            name: the name recorded in metrics and result files; defaults to the engine's
                own ``backend_name`` (``naive_hf`` or ``static_batch``).
            batch_window_s: how long the driver waits after the first arrival before
                starting a batch, so requests issued together land in the same one. It is
                small relative to any generation call, and it is the only place the adapter
                influences what is measured, which is why it is a named argument rather
                than a constant buried in the loop.
        """
        if batch_window_s < 0:
            raise ValueError(f"batch_window_s must be >= 0, got {batch_window_s}")
        self._engine = engine
        self.name = name or engine.backend_name
        self._batch_window_s = batch_window_s
        self._inbox: list[_Arrival] = []
        self._pending: dict[str, asyncio.Future[RequestOutput]] = {}
        self._wake = asyncio.Event()
        self._driver: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def engine(self) -> BaselineEngine:
        """The wrapped baseline, for stats and for tests."""
        return self._engine

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._closed

    def stats(self) -> dict[str, int | float | str]:
        """The baseline's counters, in the shape every engine in this repo reports."""
        return self._engine.stats()

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Queue one request and yield its single terminating event.

        Cancellation (the load generator closing the iterator, a timeout) aborts the
        request if it is still queued. A request already inside a ``generate`` call cannot
        be cancelled -- that is the baseline's limitation, not the adapter's -- so it runs
        to completion and its result is discarded.
        """
        if self._closed:
            raise BackendUnavailableError("baseline backend is closed", backend=self.name)
        self._ensure_driver()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RequestOutput] = loop.create_future()
        if req.request_id in self._pending:
            raise BackendRequestError(
                f"request {req.request_id!r} is already in flight", backend=self.name
            )
        prompt = req.prompt_token_ids
        payload: str | list[int] = prompt if prompt is not None else (req.prompt_text or "")
        self._pending[req.request_id] = future
        self._inbox.append(_Arrival(request_id=req.request_id, prompt=payload, request=req))
        self._wake.set()
        try:
            output = await future
        except asyncio.CancelledError:
            self._engine.abort(req.request_id)
            raise
        finally:
            self._pending.pop(req.request_id, None)
        yield TokenEvent.final(
            req.request_id,
            output.finish_reason or FinishReason.STOP,
            token_ids=list(output.new_token_ids),
            text=output.text_delta,
            usage=output.usage(),
        )

    async def health(self) -> bool:
        """Healthy until closed; the baseline has no upstream to probe."""
        return not self._closed

    async def models(self) -> list[str]:
        """The single checkpoint this baseline was built around."""
        return [self._engine.config.model]

    async def close(self) -> None:
        """Stop the driver, fail anything still waiting, and release the model. Idempotent."""
        if self._closed:
            return
        self._closed = True
        driver, self._driver = self._driver, None
        if driver is not None:
            driver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await driver
        self._fail_pending(BackendUnavailableError("baseline backend closed", backend=self.name))
        self._engine.close()

    # -- driver -------------------------------------------------------------------------

    def _ensure_driver(self) -> None:
        """Start the single driver task on first use."""
        if self._driver is None or self._driver.done():
            self._driver = asyncio.create_task(self._drive(), name=f"baseline-{self.name}")

    async def _drive(self) -> None:
        """Flush arrivals into the engine and step it until the queue drains, forever.

        The engine is stepped in a worker thread so the event loop keeps accepting
        arrivals while a generation call runs; those arrivals are picked up by the next
        flush, which is exactly the admission behaviour a static batch has.
        """
        try:
            while not self._closed:
                await self._wake.wait()
                self._wake.clear()
                if self._batch_window_s:
                    await asyncio.sleep(self._batch_window_s)
                self._flush_inbox()
                while self._engine.has_unfinished():
                    outputs = await asyncio.to_thread(self._engine.step)
                    self._resolve(outputs)
                    self._flush_inbox()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead driver must fail its waiters loudly
            logger.exception("baseline driver for %s died", self.name)
            self._fail_pending(
                BackendUnavailableError(f"baseline engine failed: {exc}", backend=self.name)
            )

    def _flush_inbox(self) -> None:
        """Hand buffered arrivals to the engine, failing the ones it rejects."""
        if not self._inbox:
            return
        arrivals, self._inbox = self._inbox, []
        for arrival in arrivals:
            future = self._pending.get(arrival.request_id)
            if future is None or future.done():
                continue
            try:
                self._engine.add_request(
                    arrival.request_id,
                    arrival.prompt,
                    arrival.request.sampling,
                    tenant_id=arrival.request.tenant_id,
                    priority=arrival.request.priority,
                    arrival=arrival.arrival,
                )
            except (ValueError, RuntimeError) as exc:
                future.set_exception(BackendRequestError(str(exc), backend=self.name))

    def _resolve(self, outputs: Sequence[RequestOutput]) -> None:
        """Deliver finished completions to the tasks waiting on them."""
        for output in outputs:
            future = self._pending.get(output.request_id)
            if future is not None and not future.done():
                future.set_result(output)

    def _fail_pending(self, error: BaseException) -> None:
        """Fail every waiting request with the same error."""
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(error)

    def __repr__(self) -> str:
        return f"BaselineBackend(name={self.name!r}, engine={type(self._engine).__name__})"


@dataclass(slots=True)
class _Arrival:
    """A request buffered between the event loop and the engine's driver task."""

    request_id: str
    prompt: str | list[int]
    request: GenerateRequest
    arrival: float = field(default_factory=time.perf_counter)


def build_load_spec(
    *,
    mode: Literal["closed", "open"] = "closed",
    concurrency: int = 1,
    rate_rps: float | None = None,
    duration_s: float | None = None,
    seed: int = 0,
    backend_name: str = "",
    request_timeout_s: float | None = None,
    repeat_requests: bool = False,
) -> LoadSpec:
    """A :class:`LoadSpec`, turning its validation errors into scenario errors."""
    try:
        return LoadSpec(
            mode=mode,
            concurrency=concurrency,
            rate_rps=rate_rps,
            duration_s=duration_s,
            seed=seed,
            backend_name=backend_name,
            request_timeout_s=request_timeout_s,
            repeat_requests=repeat_requests,
        )
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc
