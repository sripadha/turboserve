"""Scenario 2: what an automatic prefix cache is worth on a shared system prompt.

Multi-tenant traffic is full of repetition. Every request to an assistant carries the same
few hundred tokens of instructions; every request in a RAG application repeats the same
retrieved passage across a conversation's turns. Recomputing that prefix's keys and values
on every request is pure waste, and the whole cost of avoiding it is a hash table from
"this block's tokens, and everything before it" to a KV block that already holds the
answer.

The experiment is a controlled one. A single seeded prompt pool is built with a literally
shared prefix of the profile's length followed by a per-request unique suffix, and that one
pool is sent to two arms that differ in exactly one engine flag:

``cache off``
    ``SchedulerConfig.enable_prefix_caching = False``. Every request recomputes the whole
    prompt.

``cache on``
    The same engine with the cache enabled. A request whose prefix blocks are already in
    the pool adopts them instead of recomputing, so its prefill is proportional to its
    unique suffix rather than to its whole prompt.

The same comparison is available against a production engine through ``--url`` (a vLLM
server started with ``--enable-prefix-caching``) and ``--baseline-url`` (one started
without it). Two URLs rather than one, because on vLLM prefix caching is a launch flag: a
client cannot turn it off for one request, and pretending otherwise would compare a server
against itself.

Two details that decide whether the experiment measures anything at all:

* **Prompts are sent as token ids.** Re-tokenising text can merge a token across the
  boundary between the shared prefix and the unique suffix, which changes the block hashes
  and destroys the sharing under measurement.
* **The cache is warmed before the measured phase.** A block is only indexed once its
  tokens are known computed, so requests admitted in the same step that computes a prefix
  cannot hit on it. The warm-up requests carry the same prefix and run to completion first,
  which is what a steady-state service looks like; measuring the cold step instead would
  report the cost of filling the cache as if it were the cost of using it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from turboserve.bench.loadgen import build_requests, run_load
from turboserve.bench.profiles import ProfileError, load_profile
from turboserve.bench.scenarios.common import (
    AnyBackend,
    ArmOutcome,
    EngineOptions,
    ScenarioError,
    arm_table,
    build_engine_config,
    build_load_spec,
    build_prompt_pool,
    close_backend,
    default_results_dir,
    engine_derived_stats,
    load_tokenizer,
    open_run,
    openai_backend,
    reference_backend,
    resolve_slo,
    result_path,
    write_run,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from turboserve.bench.profiles import BenchProfile
    from turboserve.bench.prompts import BenchPrompt

logger = logging.getLogger(__name__)

__all__ = ["SCENARIO", "Arm", "prefix_cache_command", "run_scenario"]

SCENARIO = "prefix_cache"

#: The arm the reference engine's arms are compared against in the rendered relative table.
BASELINE_LABEL = "cache off"

#: The control arm of the vLLM pair. A vLLM server's prefix cache is a launch flag, so the
#: two vLLM arms are two servers, and their comparison only means anything against each
#: other: measuring vLLM-with-cache against *this repository's* engine-without-cache would
#: report the difference between two engines as if it were the cache's doing.
VLLM_BASELINE_LABEL = "vLLM cache off"


class Arm:
    """One configuration of the experiment: a label and how to build its backend."""

    __slots__ = ("backend_name", "caching", "label", "url")

    def __init__(self, label: str, *, backend_name: str, caching: bool, url: str | None = None):
        self.label = label
        self.backend_name = backend_name
        self.caching = caching
        self.url = url

    @property
    def baseline_label(self) -> str:
        """The cache-off arm of this arm's own engine, which is what it is measured against."""
        return VLLM_BASELINE_LABEL if self.url is not None else BASELINE_LABEL

    def build(self, model: str, options: EngineOptions, *, local_files_only: bool) -> AnyBackend:
        """Instantiate the backend this arm is served by."""
        if self.url is not None:
            return openai_backend(self.url, name=self.backend_name)
        config = build_engine_config(model, replace(options, enable_prefix_caching=self.caching))
        return reference_backend(config, name=self.backend_name, local_files_only=local_files_only)

    def __repr__(self) -> str:
        return f"Arm(label={self.label!r}, caching={self.caching}, url={self.url!r})"


def _arms(
    *,
    backends: Sequence[str],
    caching_modes: Sequence[bool],
    url: str | None,
    baseline_url: str | None,
) -> list[Arm]:
    """Expand the profile's backends and caching modes into concrete arms.

    The reference engine contributes one arm per mode because the mode is a configuration
    flag it can be constructed with; a remote server contributes one arm per URL, because
    each URL *is* a server that was launched one way or the other.
    """
    arms: list[Arm] = []
    if "reference" in backends:
        for caching in caching_modes:
            label = "cache on" if caching else BASELINE_LABEL
            arms.append(
                Arm(
                    label,
                    backend_name="reference" if caching else "reference-nocache",
                    caching=caching,
                )
            )
    if "vllm" in backends:
        if baseline_url:
            arms.append(
                Arm(
                    "vLLM cache off",
                    backend_name="vllm-nocache",
                    caching=False,
                    url=baseline_url,
                )
            )
        if url:
            arms.append(Arm("vLLM cache on", backend_name="vllm", caching=True, url=url))
    if not arms:
        raise ScenarioError(
            "no arms to run: the profile lists "
            f"{', '.join(backends) or 'no backends'}; a vllm arm additionally needs --url "
            "and/or --baseline-url"
        )
    return arms


async def run_scenario(
    profile: BenchProfile,
    *,
    model: str | None = None,
    num_requests: int | None = None,
    concurrency: int | None = None,
    shared_prefix_tokens: int | None = None,
    warmup: int = 1,
    url: str | None = None,
    baseline_url: str | None = None,
    options: EngineOptions | None = None,
    results_dir: Path | None = None,
    request_timeout_s: float | None = None,
    local_files_only: bool = False,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    slo_e2e_ms: float | None = None,
) -> list[ArmOutcome]:
    """Send one shared-prefix prompt pool to each arm and write one file per arm."""
    work = profile.scenarios.prefix_cache
    served_model = model or work.model
    count = num_requests if num_requests is not None else work.num_requests
    level = concurrency if concurrency is not None else work.concurrency
    prefix_tokens = (
        shared_prefix_tokens if shared_prefix_tokens is not None else work.shared_prefix_tokens
    )
    engine_options = options or EngineOptions(dtype=work.dtype, device=profile.device)
    slo = resolve_slo(work.slo, ttft_ms=slo_ttft_ms, tpot_ms=slo_tpot_ms, e2e_ms=slo_e2e_ms)
    base_dir = results_dir if results_dir is not None else default_results_dir()
    arms = _arms(
        backends=work.backends,
        caching_modes=work.prefix_caching,
        url=url,
        baseline_url=baseline_url,
    )

    tokenizer = load_tokenizer(served_model, local_files_only=local_files_only)
    pool = build_prompt_pool(
        count=count + warmup,
        input_tokens=work.input_tokens.as_tuple(),
        output_tokens=work.output_tokens.as_tuple(),
        seed=profile.seed,
        tenants=profile.tenants,
        shared_prefix_tokens=prefix_tokens,
        tokenizer=tokenizer,
        id_prefix=SCENARIO,
    )
    warm_prompts: list[BenchPrompt] = pool[:warmup]
    measured: list[BenchPrompt] = pool[warmup:]

    outcomes: list[ArmOutcome] = []
    for arm in arms:
        backend = arm.build(served_model, engine_options, local_files_only=local_files_only)
        try:
            outcomes.append(
                await _run_arm(
                    profile=profile,
                    arm=arm,
                    backend=backend,
                    model=served_model,
                    level=level,
                    prefix_tokens=prefix_tokens,
                    measured=measured,
                    warm_prompts=warm_prompts,
                    base_dir=base_dir,
                    request_timeout_s=request_timeout_s,
                    slo=slo,
                    options=engine_options,
                )
            )
        finally:
            await close_backend(backend)
    return outcomes


async def _run_arm(
    *,
    profile: BenchProfile,
    arm: Arm,
    backend: AnyBackend,
    model: str,
    level: int,
    prefix_tokens: int,
    measured: Sequence[BenchPrompt],
    warm_prompts: Sequence[BenchPrompt],
    base_dir: Path,
    request_timeout_s: float | None,
    slo: Any,
    options: EngineOptions,
) -> ArmOutcome:
    """Warm the arm, drive the measured load, and write its result file."""
    spec = build_load_spec(
        mode="closed",
        concurrency=level,
        seed=profile.seed,
        backend_name=arm.backend_name,
        request_timeout_s=request_timeout_s,
    )
    requests = build_requests(measured, model=model, id_prefix=f"{arm.backend_name}-")
    warm_requests = build_requests(warm_prompts, model=model, id_prefix=f"{arm.backend_name}-warm-")
    logger.info(
        "prefix_cache: arm=%s caching=%s requests=%d prefix=%d",
        arm.label,
        arm.caching,
        len(requests),
        prefix_tokens,
    )
    load = await run_load(requests, backend.generate, spec=spec, warmup=warm_requests)

    run = open_run(
        SCENARIO,
        profile.name,
        label=arm.label,
        backend=arm.backend_name,
        baseline_label=arm.baseline_label,
        spec=spec,
        config={
            **profile.config_for(SCENARIO),
            "model": model,
            "shared_prefix_tokens": prefix_tokens,
            "engine": {
                "kind": "openai" if arm.url else "reference",
                "url": arm.url,
                "enable_prefix_caching": arm.caching,
                "dtype": options.dtype,
                "device": options.device,
                "block_size": options.block_size,
            },
        },
    )
    load.into(run)
    derived = _derived(backend, run, prefix_tokens=prefix_tokens, caching=arm.caching)
    path = write_run(
        run,
        result_path(SCENARIO, base_dir, label=arm.backend_name),
        slo=slo,
        derived=derived,
    )
    return ArmOutcome(
        label=arm.label,
        backend=arm.backend_name,
        path=path,
        summary=dict(run.summary),
        derived=derived,
    )


def _derived(
    backend: AnyBackend,
    run: Any,
    *,
    prefix_tokens: int,
    caching: bool,
) -> dict[str, Any]:
    """The cache-specific figures, alongside the engine's own counters.

    ``cached_prompt_token_fraction`` is computed from what the engine reported against the
    prompt tokens this run actually sent, rather than taken from the engine alone, because
    the engine's counter also covers the warm-up requests.
    """
    derived: dict[str, Any] = {
        "shared_prefix_tokens": prefix_tokens,
        "prefix_caching": caching,
        **engine_derived_stats(backend),
    }
    prompt_tokens = sum(record.prompt_tokens for record in run.ok_requests)
    cached = derived.get("num_cached_prompt_tokens")
    if isinstance(cached, int | float) and prompt_tokens > 0:
        derived["cached_prompt_token_fraction"] = min(float(cached) / prompt_tokens, 1.0)
    return derived


def prefix_cache_command(
    profile: Annotated[
        str, typer.Option("--profile", help="Profile in configs/bench/profiles.yaml.")
    ] = "h100",
    profiles_path: Annotated[
        Path | None, typer.Option("--profiles", help="Alternative profiles file.")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Override the profile's checkpoint.")
    ] = None,
    num_requests: Annotated[
        int | None, typer.Option("--num-requests", min=1, help="Override the request count.")
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", min=1, help="Override the concurrency.")
    ] = None,
    shared_prefix_tokens: Annotated[
        int | None,
        typer.Option("--shared-prefix-tokens", min=1, help="Override the shared prefix length."),
    ] = None,
    warmup: Annotated[
        int,
        typer.Option(
            "--warmup",
            min=0,
            help="Requests sent before the measured phase; they populate the cache.",
        ),
    ] = 1,
    url: Annotated[
        str | None,
        typer.Option("--url", help="OpenAI-compatible server started WITH prefix caching."),
    ] = None,
    baseline_url: Annotated[
        str | None,
        typer.Option("--baseline-url", help="OpenAI-compatible server started WITHOUT it."),
    ] = None,
    dtype: Annotated[
        str | None, typer.Option("--dtype", help="Override the profile's dtype.")
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="Override the profile's device.")
    ] = None,
    max_num_seqs: Annotated[int, typer.Option("--max-num-seqs", min=1)] = 64,
    max_num_batched_tokens: Annotated[int, typer.Option("--max-num-batched-tokens", min=1)] = 2048,
    block_size: Annotated[int, typer.Option("--block-size", min=1)] = 16,
    kv_blocks: Annotated[
        int | None,
        typer.Option("--kv-blocks", min=1, help="Fix the KV pool size instead of profiling it."),
    ] = None,
    request_timeout_s: Annotated[
        float | None, typer.Option("--request-timeout", min=0.0, help="Per-request deadline.")
    ] = None,
    local_files_only: Annotated[
        bool, typer.Option("--local-files-only/--allow-download", help="Never contact the Hub.")
    ] = False,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", help="Where result files are written.")
    ] = None,
    slo_ttft_ms: Annotated[float | None, typer.Option("--slo-ttft-ms", min=0.0)] = None,
    slo_tpot_ms: Annotated[float | None, typer.Option("--slo-tpot-ms", min=0.0)] = None,
    slo_e2e_ms: Annotated[float | None, typer.Option("--slo-e2e-ms", min=0.0)] = None,
) -> None:
    """Measure a shared system prompt with the prefix cache off and on."""
    try:
        loaded = load_profile(profile, path=profiles_path)
    except (ProfileError, KeyError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from exc
    work = loaded.scenarios.prefix_cache
    options = EngineOptions(
        dtype=dtype or work.dtype,
        device=device or loaded.device,
        seed=loaded.seed,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
        num_blocks=kv_blocks,
    )
    try:
        outcomes = asyncio.run(
            run_scenario(
                loaded,
                model=model,
                num_requests=num_requests,
                concurrency=concurrency,
                shared_prefix_tokens=shared_prefix_tokens,
                warmup=warmup,
                url=url,
                baseline_url=baseline_url,
                options=options,
                results_dir=results_dir,
                request_timeout_s=request_timeout_s,
                local_files_only=local_files_only,
                slo_ttft_ms=slo_ttft_ms,
                slo_tpot_ms=slo_tpot_ms,
                slo_e2e_ms=slo_e2e_ms,
            )
        )
    except ScenarioError as exc:
        raise typer.BadParameter(str(exc)) from exc
    arm_table(f"{SCENARIO} ({loaded.name})", outcomes)
