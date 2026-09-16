"""Scenario 4: many adapters on one base model, against the same model with none.

Two questions, and they are different kinds of question, which is why this scenario
produces two blocks of figures rather than one table.

**What does serving N adapters cost in VRAM?** This is arithmetic over real tensor sizes,
not a measurement: the adapter pool is preallocated, so the bytes are known exactly once
the model and the slot budget are known. The comparison is against the only alternative
that serves the same N tenants -- N separately merged copies of the base model, i.e.
``N * base_bytes``. :meth:`~turboserve.engine.lora.registry.LoRARegistry.vram_report`
computes both sides from the tensors themselves and this module copies the result into the
run's ``summary["derived"]["vram"]``.

**What does serving N adapters cost in latency?** This is a measurement, and it needs a
control: the same engine, the same prompts, the same concurrency, the same output lengths,
with no adapter at all. The mixed-adapter arms round-robin over the N adapters, so every
step contains several adapters at once and the grouped (SGMV) path is exercised rather than
the degenerate single-adapter case. The difference in p95 TTFT/TPOT between an arm and the
base-only arm is the cost of the adapter machinery, and the base-only arm is recorded as
``baseline_label`` in every arm's config so a reader of one file can find the control.

Both engine paths are supported. ``--backend reference`` drives the in-process engine with
:mod:`turboserve.engine.lora` installed; ``--backend vllm --url`` drives a vLLM server
started with ``--enable-lora --max-loras N --max-lora-rank R``, where an adapter is
addressed by putting its name in the OpenAI ``model`` field (which is what
:class:`~turboserve.gateway.backends.openai_compat.OpenAICompatBackend` does with
``GenerateRequest.lora``). The VRAM block is only produced for the reference backend: a
remote server's tensor sizes are not something this process can measure, and the repository
does not print numbers it did not obtain.

Nothing in this module chooses workload sizes. They come from
``configs/bench/profiles.yaml`` and are embedded verbatim in every result file.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer

from turboserve.bench.metrics import pct_delta
from turboserve.bench.profiles import MultiLoRAProfile, ProfileError, load_profile
from turboserve.bench.records import RunResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence

    from turboserve.bench.loadgen import GenerateCallable
    from turboserve.bench.profiles import BenchProfile
    from turboserve.bench.prompts import BenchPrompt
    from turboserve.bench.records import SLO
    from turboserve.engine.core.types import DeviceName
    from turboserve.engine.lora.registry import LoRARegistry

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_LABEL",
    "SCENARIO",
    "ArmResult",
    "adapter_assignment",
    "arm_label",
    "derived_block",
    "multi_lora_app",
    "result_path",
    "run_scenario",
]

#: Scenario name: the results directory, the ``scenario`` field, and the CLI command.
SCENARIO = "multi_lora"

#: Label of the control arm. Every arm records it as ``baseline_label`` so the renderer can
#: build the relative table from any single file plus its siblings.
BASE_LABEL = "base only"

BackendName = Literal["reference", "vllm"]


def arm_label(num_adapters: int) -> str:
    """Human name of an arm: the control, or ``"<n> adapters"``."""
    return BASE_LABEL if num_adapters == 0 else f"{num_adapters} adapters"


def result_path(results_dir: Path | str, label: str, *, now: datetime | None = None) -> Path:
    """``<results_dir>/multi_lora/<utc timestamp>-<slug>.json``.

    The label is in the filename as well as in the JSON because a directory listing is the
    first thing anyone looks at, and ``20260916T101500Z-64-adapters.json`` answers the
    question that ``20260916T101500Z.json`` does not.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(char if char.isalnum() else "-" for char in label).strip("-").lower()
    return Path(results_dir) / SCENARIO / f"{stamp}-{slug}.json"


def adapter_assignment(prompts: Sequence[BenchPrompt], names: Sequence[str]) -> dict[str, str]:
    """Round-robin ``prompt_id -> adapter name`` over ``names``.

    Round-robin rather than random so that every adapter gets the same number of requests
    (a random assignment would leave the tail adapters with few samples and make their p95
    meaningless) and so that two runs of the same profile send the same adapter the same
    prompts.
    """
    if not names:
        return {}
    return {prompt.prompt_id: names[index % len(names)] for index, prompt in enumerate(prompts)}


def _adapter_picker(assignment: Mapping[str, str]) -> Callable[[BenchPrompt], str | None]:
    """A typed ``build_requests(lora=...)`` callable over a prompt-to-adapter table."""

    def pick(prompt: BenchPrompt) -> str | None:
        return assignment.get(prompt.prompt_id)

    return pick


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm of the scenario: its run, where it was written, and what it measured."""

    label: str
    num_adapters: int
    run: RunResult
    path: Path | None = None

    @property
    def summary(self) -> dict[str, Any]:
        """The arm's summary block."""
        return self.run.summary


def _p95(summary: Mapping[str, Any], metric: str) -> float | None:
    """``summary[metric]["p95"]`` when it is there, else ``None``."""
    block = summary.get(metric)
    if isinstance(block, dict):
        value = block.get("p95")
        if isinstance(value, int | float):
            return float(value)
    return None


def derived_block(
    *,
    num_adapters: int,
    baseline: Mapping[str, Any] | None,
    summary: Mapping[str, Any],
    vram: Mapping[str, int | float] | None,
    lora_stats: Mapping[str, int | float] | None,
) -> dict[str, Any]:
    """The scenario-specific figures the report renders under the arm's table.

    ``*_loss_pct`` is the percentage increase of an arm's p95 over the base-only arm's;
    ``None`` for the control itself and whenever the control is missing, because a "loss"
    with nothing to lose against is not a number this repository will print.
    """
    block: dict[str, Any] = {
        "num_adapters": num_adapters,
        "p95_ttft_loss_pct": None,
        "p95_tpot_loss_pct": None,
        "p95_e2e_loss_pct": None,
    }
    if baseline is not None and num_adapters > 0:
        for key, metric in (
            ("p95_ttft_loss_pct", "ttft_ms"),
            ("p95_tpot_loss_pct", "tpot_ms"),
            ("p95_e2e_loss_pct", "e2e_ms"),
        ):
            block[key] = pct_delta(_p95(summary, metric), _p95(baseline, metric))
    if vram is not None:
        block["vram"] = dict(vram)
    if lora_stats is not None:
        block["lora"] = dict(lora_stats)
    return block


# -- backends ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Arm:
    """What one arm needs from whichever backend is driving it."""

    label: str
    num_adapters: int
    adapter_names: tuple[str, ...]
    generate: GenerateCallable
    registry: LoRARegistry | None


class _ReferenceDriver:
    """Drives the in-process reference engine, reinstalling the pool once per arm.

    The engine, the model and the KV cache are built once and shared: rebuilding them per
    arm would make the arms differ by more than the thing under test (a cold allocator, a
    differently fragmented cache), and the comparison would stop meaning what it says.
    Only the adapter pool is rebuilt, because its size *is* the independent variable.
    """

    def __init__(
        self,
        work: MultiLoRAProfile,
        *,
        device: DeviceName,
        adapters_dir: Path,
        max_loras: int | None,
        local_files_only: bool,
    ) -> None:
        from turboserve.engine.core.types import EngineConfig, SchedulerConfig
        from turboserve.engine.runtime.async_engine import AsyncLLMEngine
        from turboserve.engine.runtime.engine import LLMEngine

        # The pool is capped at what the workload can actually use
        # (max_num_seqs sequences of the longest prompt plus completion). Below that cap
        # the device's memory budget still binds, so this only prevents reserving blocks
        # no request in this profile could ever fill.
        config = EngineConfig(
            model=work.model,
            dtype=work.dtype,
            device=device,
            max_model_len=work.input_tokens.max + work.output_tokens.max,
            scheduler=SchedulerConfig(max_num_seqs=max(work.concurrency, 1)),
        )
        self._engine = LLMEngine(config, local_files_only=local_files_only)
        self._async = AsyncLLMEngine(self._engine)
        self._adapters_dir = adapters_dir
        self._max_loras = max_loras
        self._local_files_only = local_files_only
        self._registry: LoRARegistry | None = None
        self._backends: list[Any] = []

    @property
    def tokenizer(self) -> Any:
        """The engine's tokenizer, used to build prompts of exact token lengths."""
        return self._engine.tokenizer

    @property
    def model_name(self) -> str:
        """The name requests must address."""
        return self._engine.config.model

    async def start(self) -> None:
        """Start the engine's step loop."""
        await self._async.start()

    async def close(self) -> None:
        """Shut the engine down; the per-arm backends share it and do not own it."""
        await self._async.close()

    def arm(self, num_adapters: int) -> _Arm:
        """Install a pool sized for ``num_adapters`` and return a backend addressing it."""
        from turboserve.engine.lora.layers import install_lora, uninstall_lora
        from turboserve.engine.lora.registry import LoRARegistry
        from turboserve.gateway.backends.local_engine import LocalEngineBackend

        uninstall_lora(self._engine)
        self._registry = None
        if num_adapters == 0:
            backend = LocalEngineBackend(name="reference", engine=self._async)
            self._backends.append(backend)
            return _Arm(BASE_LABEL, 0, (), backend.generate, None)

        slots = num_adapters if self._max_loras is None else min(self._max_loras, num_adapters)
        registry = LoRARegistry(slots, max_lora_rank=self._widest_rank())
        install_lora(self._engine, registry)
        names = list(registry.register_directory(self._adapters_dir, limit=num_adapters))
        backend = LocalEngineBackend(
            name="reference", engine=self._async, adapters=registry.name_to_id()
        )
        self._backends.append(backend)
        self._registry = registry
        return _Arm(arm_label(num_adapters), num_adapters, tuple(names), backend.generate, registry)

    def _widest_rank(self) -> int:
        """Rank the pool must accommodate, read from the adapters on disk."""
        import json

        from turboserve.engine.lora.adapter import PEFT_CONFIG_FILE, discover_adapters

        ranks = []
        for directory in discover_adapters(self._adapters_dir):
            data = json.loads((directory / PEFT_CONFIG_FILE).read_text(encoding="utf-8"))
            ranks.append(int(data.get("r", 0) or 0))
        if not ranks:
            raise ProfileError(f"no PEFT adapters under {self._adapters_dir}")
        return max(ranks)


class _VLLMDriver:
    """Drives a vLLM server over the OpenAI-compatible API, addressing adapters by name."""

    def __init__(
        self, work: MultiLoRAProfile, *, url: str, adapters_dir: Path, local_files_only: bool
    ) -> None:
        from turboserve.engine.lora.adapter import discover_adapters
        from turboserve.engine.runtime.streaming import get_tokenizer
        from turboserve.gateway.backends.openai_compat import OpenAICompatBackend

        self._names = [path.name for path in discover_adapters(adapters_dir)]
        self._backend = OpenAICompatBackend(url, name="vllm")
        self._tokenizer = get_tokenizer(work.model, local_files_only=local_files_only)
        self._model = work.model

    @property
    def tokenizer(self) -> Any:
        """A local tokenizer for the served model, used only to size prompts."""
        return self._tokenizer

    @property
    def model_name(self) -> str:
        """The name requests must address when they use no adapter."""
        return self._model

    async def start(self) -> None:
        """Nothing to start: the server is someone else's process."""
        return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._backend.close()

    def arm(self, num_adapters: int) -> _Arm:
        """An arm addressing the first ``num_adapters`` adapters the server was given."""
        if num_adapters > len(self._names):
            raise ProfileError(
                f"the adapters directory holds {len(self._names)} adapters but the arm needs "
                f"{num_adapters}; vLLM must also be started with --max-loras >= {num_adapters}"
            )
        names = tuple(self._names[:num_adapters])
        return _Arm(arm_label(num_adapters), num_adapters, names, self._backend.generate, None)


@asynccontextmanager
async def _driver(
    work: MultiLoRAProfile,
    *,
    backend: BackendName,
    device: DeviceName,
    url: str | None,
    adapters_dir: Path,
    max_loras: int | None,
    local_files_only: bool,
) -> AsyncIterator[_ReferenceDriver | _VLLMDriver]:
    """Build, start and always shut down the driver for ``backend``."""
    engine: _ReferenceDriver | _VLLMDriver
    if backend == "vllm":
        if not url:
            raise ProfileError("--backend vllm needs --url, e.g. http://127.0.0.1:8000/v1")
        engine = _VLLMDriver(
            work, url=url, adapters_dir=adapters_dir, local_files_only=local_files_only
        )
    else:
        engine = _ReferenceDriver(
            work,
            device=device,
            adapters_dir=adapters_dir,
            max_loras=max_loras,
            local_files_only=local_files_only,
        )
    await engine.start()
    try:
        yield engine
    finally:
        await engine.close()


# -- the scenario -------------------------------------------------------------------------


async def run_scenario(
    profile: BenchProfile,
    *,
    adapters_dir: Path | str,
    backend: BackendName = "reference",
    url: str | None = None,
    results_dir: Path | str = "results",
    adapter_counts: Sequence[int] | None = None,
    concurrency: int | None = None,
    num_requests: int | None = None,
    max_loras: int | None = None,
    slo: SLO | None = None,
    gpu_price_per_hour: float | None = None,
    price_source: str | None = None,
    local_files_only: bool = False,
    write: bool = True,
    warmup_requests: int = 4,
    hardware: Mapping[str, Any] | None = None,
) -> list[ArmResult]:
    """Run the control arm and one arm per adapter count, writing a result file for each.

    Arms run in one process, back to back, against one engine: the control first (so that
    every later arm has a baseline to record) and then the adapter counts in ascending
    order. The order matters for reproducibility, not for fairness -- each arm sends the
    same prompts with the same seed.
    """
    # Imported here rather than at module scope: ``loadgen`` pulls in the engine's torch
    # types, and wiring this sub-app into the CLI must not make ``turboserve --help`` load
    # torch. Everything else this module needs at import time is pure Python.
    from turboserve.bench.loadgen import LoadSpec, build_requests, load_config, run_load
    from turboserve.bench.prompts import SyntheticPromptBuilder

    work = profile.scenario(SCENARIO)
    if not isinstance(work, MultiLoRAProfile):
        raise ProfileError(f"profile {profile.name!r} has no multi_lora scenario")
    counts = list(adapter_counts if adapter_counts is not None else work.adapter_counts)
    if any(count < 1 for count in counts):
        raise ProfileError(f"adapter counts must be positive, got {counts}")
    load_concurrency = concurrency or work.concurrency
    total_requests = num_requests or work.num_requests
    adapters_path = Path(adapters_dir)

    results: list[ArmResult] = []
    baseline_summary: dict[str, Any] | None = None

    async with _driver(
        work,
        backend=backend,
        device=profile.device,
        url=url,
        adapters_dir=adapters_path,
        max_loras=max_loras,
        local_files_only=local_files_only,
    ) as driver:
        builder = SyntheticPromptBuilder(driver.tokenizer, seed=profile.seed)
        prompts = builder.build_batch(
            total_requests,
            input_tokens=work.input_tokens.as_tuple(),
            output_tokens=work.output_tokens.as_tuple(),
            tenants=tuple(profile.tenants),
            prefix=SCENARIO,
        )
        warmup = prompts[: max(0, min(warmup_requests, len(prompts)))]

        plan = ([0] if work.include_base_only else []) + sorted(counts)
        for num_adapters in plan:
            arm = driver.arm(num_adapters)
            pick = _adapter_picker(adapter_assignment(prompts, arm.adapter_names))
            requests = build_requests(
                prompts,
                model=driver.model_name,
                lora=pick,
                id_prefix=f"{num_adapters}a-",
            )
            warmup_reqs = build_requests(
                warmup,
                model=driver.model_name,
                lora=pick,
                id_prefix=f"{num_adapters}a-warm-",
            )
            spec = LoadSpec(
                mode="closed",
                concurrency=load_concurrency,
                seed=profile.seed,
                backend_name=backend,
            )
            logger.info(
                "arm %s: %d requests at concurrency %d over %d adapters",
                arm.label,
                len(requests),
                load_concurrency,
                num_adapters,
            )
            load = await run_load(
                requests,
                arm.generate,
                spec=spec,
                warmup=warmup_reqs,
                prompt_tokens={p.prompt_id: p.num_prompt_tokens for p in prompts},
            )

            run = RunResult.start(
                SCENARIO,
                profile.name,
                config={
                    **profile.config_for(SCENARIO),
                    "label": arm.label,
                    "backend": backend,
                    "baseline_label": BASE_LABEL,
                    "num_adapters": num_adapters,
                    "adapter_names": list(arm.adapter_names),
                    "adapters_dir": str(adapters_path),
                    "load": load_config(spec, num_adapters=num_adapters),
                },
                gpu_price_per_hour=gpu_price_per_hour,
                price_source=price_source,
                hardware=dict(hardware) if hardware is not None else None,
            )
            load.into(run)
            run.finish(slo=slo)
            run.summary["derived"] = derived_block(
                num_adapters=num_adapters,
                baseline=baseline_summary,
                summary=run.summary,
                vram=(
                    arm.registry.vram_report(num_adapters=num_adapters)
                    if arm.registry is not None
                    else None
                ),
                lora_stats=arm.registry.stats_dict() if arm.registry is not None else None,
            )
            if num_adapters == 0:
                baseline_summary = dict(run.summary)

            path: Path | None = None
            if write:
                path = run.save(result_path(results_dir, arm.label))
                logger.info("wrote %s", path)
            results.append(ArmResult(arm.label, num_adapters, run, path))

    return results


# -- CLI ----------------------------------------------------------------------------------

multi_lora_app = typer.Typer(
    name="multi-lora",
    help="Measure many-adapter serving against the same engine with no adapters.",
    no_args_is_help=True,
)


@multi_lora_app.command("run")
def run_command(  # noqa: PLR0913 - a CLI is a flat list of options
    profile: Annotated[
        str, typer.Option("--profile", help="Profile name in configs/bench/profiles.yaml.")
    ] = "h100",
    adapters_dir: Annotated[
        Path,
        typer.Option(
            "--adapters-dir",
            help="Directory of PEFT adapters, as written by scripts/make_lora_adapters.py.",
        ),
    ] = Path("adapters"),
    out: Annotated[
        Path, typer.Option("--out", help="Results directory; files land in <out>/multi_lora/.")
    ] = Path("results"),
    backend: Annotated[
        str, typer.Option("--backend", help="'reference' (in-process engine) or 'vllm'.")
    ] = "reference",
    url: Annotated[
        str | None, typer.Option("--url", help="vLLM OpenAI base URL, e.g. http://host:8000/v1.")
    ] = None,
    adapters: Annotated[
        str | None,
        typer.Option("--adapters", help="Comma-separated adapter counts, overriding the profile."),
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", min=1, help="Override the profile's concurrency.")
    ] = None,
    requests: Annotated[
        int | None, typer.Option("--requests", min=1, help="Override the profile's request count.")
    ] = None,
    max_loras: Annotated[
        int | None,
        typer.Option(
            "--max-loras",
            min=1,
            help="Cap GPU slots below the arm's adapter count, to exercise LRU eviction.",
        ),
    ] = None,
    slo_ttft_ms: Annotated[float | None, typer.Option("--slo-ttft-ms", min=0.0)] = None,
    slo_tpot_ms: Annotated[float | None, typer.Option("--slo-tpot-ms", min=0.0)] = None,
    slo_e2e_ms: Annotated[float | None, typer.Option("--slo-e2e-ms", min=0.0)] = None,
    local_files_only: Annotated[bool, typer.Option("--local-files-only/--allow-download")] = False,
    profiles_path: Annotated[
        Path | None, typer.Option("--profiles", help="Alternative profiles YAML.")
    ] = None,
) -> None:
    """Run the multi-adapter scenario and write one result file per arm."""
    import asyncio
    import os

    from turboserve.bench.metrics import slo_from_mapping

    if backend not in ("reference", "vllm"):
        raise typer.BadParameter("must be 'reference' or 'vllm'", param_hint="--backend")
    try:
        bench_profile = load_profile(profile, path=profiles_path)
    except (ProfileError, KeyError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from exc

    counts: list[int] | None = None
    if adapters:
        try:
            counts = [int(piece) for piece in adapters.split(",") if piece.strip()]
        except ValueError as exc:
            raise typer.BadParameter("expected a comma-separated list of integers") from exc

    objective = slo_from_mapping(
        {"ttft_ms": slo_ttft_ms, "tpot_ms": slo_tpot_ms, "e2e_ms": slo_e2e_ms}
    )
    price = os.environ.get("TURBOSERVE_GPU_PRICE_PER_HOUR")
    arms = asyncio.run(
        run_scenario(
            bench_profile,
            adapters_dir=adapters_dir,
            backend="vllm" if backend == "vllm" else "reference",
            url=url,
            results_dir=out,
            adapter_counts=counts,
            concurrency=concurrency,
            num_requests=requests,
            max_loras=max_loras,
            slo=objective,
            gpu_price_per_hour=float(price) if price else None,
            price_source=os.environ.get("TURBOSERVE_GPU_PRICE_SOURCE"),
            local_files_only=local_files_only,
        )
    )
    for arm in arms:
        typer.echo(f"{arm.label}: {arm.path if arm.path is not None else '(not written)'}")
