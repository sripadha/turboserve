"""Scenario 1: sequential decoding vs static batching vs continuous batching.

The question is what *batching policy* is worth, so every arm is given the identical
seeded prompt pool, the identical sampling parameters and the identical client-side load,
and only the thing that serves them changes:

``naive``
    :class:`~turboserve.engine.runtime.naive.NaiveHFEngine` -- one request per
    ``transformers.generate`` call, run to completion before the next starts. The lower
    bound: a decode step reads every weight of the model to produce a single token, so the
    arithmetic units idle on memory while the rest of the queue is not served at all.

``static batch``
    :class:`~turboserve.engine.runtime.naive.StaticBatchHFEngine` -- collect a batch, pad
    it to the longest prompt, generate for the longest requested completion, then start the
    next batch. Fixes the idle-hardware problem and introduces two of its own: padding
    waste, and no admission until the whole batch finishes.

``continuous batching``
    This repository's :class:`~turboserve.engine.runtime.engine.LLMEngine` behind the
    gateway's in-process backend -- a token-level scheduler that admits, preempts and
    retires sequences inside the decode loop.

``vLLM`` / ``SGLang``
    The same prompts sent to a production engine over its OpenAI-compatible API
    (``--url``), so the reference engine's row sits next to one a reader already trusts.
    Both are the same arm mechanically -- one ``OpenAICompatBackend`` against one URL -- and
    they are separate arms only because a row has to say which server produced it. One
    invocation measures one server, so a machine running both is swept once per engine.

Three deliberate choices about what is *not* varied.

**Prefix caching is off** for every engine arm by default. The prompt pool is reused across
the concurrency sweep, so a warm cache would make the later runs look faster for a reason
this scenario is not about; it has its own scenario.

**``ignore_eos`` is on** (the default of
:func:`~turboserve.bench.loadgen.build_requests`), so each arm generates exactly the token
count asked for instead of stopping at a different point per engine.

**Every request asks for the same number of output tokens**, the mean of the profile's
output range unless ``--output-tokens`` says otherwise, so the arms are given identical
work. This is forced by the baselines: ``transformers.generate`` applies one sampling
configuration to a whole batch and
:class:`~turboserve.engine.runtime.naive.StaticBatchHFEngine` rejects a heterogeneous one
rather than silently serving it with the first request's settings. The cost is that this
scenario does not show the part of continuous batching's advantage that comes from
*retiring short sequences early*, which understates it; the input lengths stay
heterogeneous, so the padding waste of a static batch is still measured. The prefix-cache,
speculative and adapter scenarios, which have no baseline arm, keep the profile's full
output distribution.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from turboserve.bench.loadgen import build_requests, run_load
from turboserve.bench.profiles import ProfileError, load_profile
from turboserve.bench.scenarios.common import (
    REMOTE_ENGINE_LABELS,
    AnyBackend,
    ArmOutcome,
    EngineOptions,
    ScenarioError,
    arm_table,
    baseline_backend,
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
    remote_server_info,
    resolve_slo,
    result_path,
    write_run,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from turboserve.bench.profiles import BenchProfile, NaiveVsCBProfile
    from turboserve.bench.prompts import BenchPrompt

logger = logging.getLogger(__name__)

__all__ = ["ARM_LABELS", "SCENARIO", "naive_vs_cb_command", "run_scenario"]

SCENARIO = "naive_vs_cb"

#: How each profile backend name is shown in the rendered tables. The keys are the values
#: a profile's ``backends`` list may hold; the values are what a reader sees. The remote
#: engines come from ``REMOTE_ENGINE_LABELS`` rather than being spelled again here, so a
#: production engine is named in exactly one place in this package.
ARM_LABELS: dict[str, str] = {
    "naive_hf": "naive",
    "static_batch": "static batch",
    "reference": "continuous batching",
    **REMOTE_ENGINE_LABELS,
}

#: The arm every other arm is compared against in the rendered relative table.
BASELINE_ARM = "naive_hf"

#: A second reference the renderer draws a relative table against when it was measured.
#: Sequential decoding is the baseline because it is the floor, but the comparison a reader
#: of this scenario came for is continuous batching against a *padded static batch* -- the
#: thing a serving stack does when it batches naively -- and that ratio is only a rendered
#: number if some arm asks for it (``config["compare_to"]``).
SECONDARY_ARM = "static_batch"


def _select_arms(work: NaiveVsCBProfile, requested: Sequence[str] | None) -> list[str]:
    """The arms to run, in the profile's order, validated against what it declares."""
    available = list(work.backends)
    if not requested:
        return available
    unknown = [arm for arm in requested if arm not in ARM_LABELS]
    if unknown:
        raise ScenarioError(
            f"unknown arm(s) {', '.join(unknown)}; known: {', '.join(sorted(ARM_LABELS))}"
        )
    missing = [arm for arm in requested if arm not in available]
    if missing:
        raise ScenarioError(
            f"arm(s) {', '.join(missing)} are not in profile's backends ({', '.join(available)})"
        )
    return [arm for arm in available if arm in requested]


def _make_backend(
    arm: str,
    model: str,
    options: EngineOptions,
    *,
    batch_size: int,
    url: str | None,
    api_key: str | None,
    local_files_only: bool,
) -> AnyBackend:
    """Build the backend one arm is served by."""
    if arm in REMOTE_ENGINE_LABELS:
        if not url:
            raise ScenarioError(
                f"the {arm} arm needs --url pointing at an OpenAI-compatible server"
            )
        return openai_backend(url, name=arm, api_key=api_key)
    config = build_engine_config(model, options)
    if arm == "reference":
        return reference_backend(config, local_files_only=local_files_only)
    if arm == "naive_hf":
        return baseline_backend("naive_hf", config, local_files_only=local_files_only)
    if arm == "static_batch":
        return baseline_backend(
            "static_batch", config, batch_size=batch_size, local_files_only=local_files_only
        )
    raise ScenarioError(f"unknown arm {arm!r}")  # pragma: no cover - guarded by _select_arms


async def run_scenario(
    profile: BenchProfile,
    *,
    arms: Sequence[str] | None = None,
    model: str | None = None,
    num_requests: int | None = None,
    concurrencies: Sequence[int] | None = None,
    output_tokens: int | None = None,
    warmup: int = 2,
    url: str | None = None,
    api_key: str | None = None,
    options: EngineOptions | None = None,
    results_dir: Path | None = None,
    request_timeout_s: float | None = None,
    local_files_only: bool = False,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    slo_e2e_ms: float | None = None,
) -> list[ArmOutcome]:
    """Run every selected arm at every selected concurrency and write one file per run.

    One backend is built per arm and reused across the concurrency sweep: loading a
    checkpoint once instead of once per load level saves most of the run's wall time, and
    it is safe because nothing in the sweep depends on engine state (prefix caching is off
    and the scheduler is drained between runs when the previous load finishes).
    """
    work = profile.scenarios.naive_vs_cb
    selected = _select_arms(work, arms)
    if not selected:
        raise ScenarioError("no arms selected")
    levels = list(concurrencies) if concurrencies else list(work.concurrencies)
    if any(level < 1 for level in levels):
        raise ScenarioError(f"concurrencies must be >= 1, got {levels}")
    served_model = model or work.model
    count = num_requests if num_requests is not None else work.num_requests
    uniform_output = output_tokens if output_tokens is not None else _mean_output(work)
    if uniform_output < 1:
        raise ScenarioError(f"output tokens must be >= 1, got {uniform_output}")
    engine_options = options or EngineOptions(dtype=work.dtype, device=profile.device)
    slo = resolve_slo(work.slo, ttft_ms=slo_ttft_ms, tpot_ms=slo_tpot_ms, e2e_ms=slo_e2e_ms)
    base_dir = results_dir if results_dir is not None else default_results_dir()

    tokenizer = load_tokenizer(served_model, local_files_only=local_files_only)
    pool = build_prompt_pool(
        count=count + warmup,
        input_tokens=work.input_tokens.as_tuple(),
        output_tokens=(uniform_output, uniform_output),
        seed=profile.seed,
        tenants=profile.tenants,
        tokenizer=tokenizer,
        id_prefix=SCENARIO,
    )
    warm_prompts: list[BenchPrompt] = pool[:warmup]
    measured: list[BenchPrompt] = pool[warmup:]

    baseline_label = ARM_LABELS[BASELINE_ARM if BASELINE_ARM in selected else selected[0]]
    outcomes: list[ArmOutcome] = []
    for arm in selected:
        backend = _make_backend(
            arm,
            served_model,
            engine_options,
            batch_size=max(levels),
            url=url,
            api_key=api_key,
            local_files_only=local_files_only,
        )
        # Asked once per arm rather than once per run: it is a property of the server, not
        # of the load, and a server that answers nothing must not be re-probed per level.
        server = await remote_server_info(backend) if arm in REMOTE_ENGINE_LABELS else {}
        try:
            for level in levels:
                outcomes.append(
                    await _run_one(
                        profile=profile,
                        work_model=served_model,
                        arm=arm,
                        backend=backend,
                        level=level,
                        measured=measured,
                        warm_prompts=warm_prompts,
                        baseline_label=baseline_label,
                        compare_to_static=SECONDARY_ARM in selected and arm != SECONDARY_ARM,
                        base_dir=base_dir,
                        request_timeout_s=request_timeout_s,
                        slo=slo,
                        options=engine_options,
                        url=url,
                        server=server,
                        uniform_output=uniform_output,
                    )
                )
        finally:
            await close_backend(backend)
    return outcomes


async def _run_one(
    *,
    profile: BenchProfile,
    work_model: str,
    arm: str,
    backend: AnyBackend,
    level: int,
    measured: Sequence[BenchPrompt],
    warm_prompts: Sequence[BenchPrompt],
    baseline_label: str,
    compare_to_static: bool,
    base_dir: Path,
    request_timeout_s: float | None,
    slo: Any,
    options: EngineOptions,
    url: str | None,
    server: Mapping[str, Any],
    uniform_output: int,
) -> ArmOutcome:
    """One arm at one concurrency: drive the load, summarise it, write the file."""
    label = ARM_LABELS[arm]
    spec = build_load_spec(
        mode="closed",
        concurrency=level,
        seed=profile.seed,
        backend_name=arm,
        request_timeout_s=request_timeout_s,
    )
    requests = build_requests(measured, model=work_model, id_prefix=f"{arm}-c{level}-")
    warm_requests = build_requests(warm_prompts, model=work_model, id_prefix=f"{arm}-warm-")
    logger.info("naive_vs_cb: arm=%s concurrency=%d requests=%d", arm, level, len(requests))
    load = await run_load(requests, backend.generate, spec=spec, warmup=warm_requests)

    run = open_run(
        SCENARIO,
        profile.name,
        label=label,
        backend=arm,
        baseline_label=baseline_label,
        spec=spec,
        config={
            **profile.config_for(SCENARIO),
            "model": work_model,
            "output_tokens": uniform_output,
            "engine": _engine_block(arm, options, url, server),
            **({"compare_to": [ARM_LABELS[SECONDARY_ARM]]} if compare_to_static else {}),
        },
    )
    load.into(run)
    derived = {
        **engine_derived_stats(backend),
        "max_in_flight_observed": load.max_in_flight_observed,
        "warmup_requests": len(warm_requests),
    }
    path = write_run(
        run,
        result_path(SCENARIO, base_dir, label=f"{arm}-c{level}"),
        slo=slo,
        derived=derived,
    )
    return ArmOutcome(
        label=label, backend=arm, path=path, summary=dict(run.summary), derived=derived
    )


def _mean_output(work: NaiveVsCBProfile) -> int:
    """The single completion length every request asks for.

    The mean of the profile's range rather than its maximum, so the run generates the token
    volume the profile describes instead of the largest volume it permits.
    """
    return (work.output_tokens.min + work.output_tokens.max + 1) // 2


def _engine_block(
    arm: str, options: EngineOptions, url: str | None, server: Mapping[str, Any]
) -> dict[str, Any]:
    """What served this arm, recorded so a result file can be re-run from itself.

    A remote arm records the engine's *name* as well as its URL, because the URL says where
    the server was and not what it was, and ``server`` carries whatever the server itself
    reported (see :func:`remote_server_info`) -- empty when it reported nothing.
    """
    if arm in REMOTE_ENGINE_LABELS:
        block: dict[str, Any] = {"kind": "openai", "engine": arm, "url": url}
        if server:
            block["server"] = dict(server)
        return block
    return {
        "kind": arm,
        "dtype": options.dtype,
        "device": options.device,
        "max_num_seqs": options.max_num_seqs,
        "max_num_batched_tokens": options.max_num_batched_tokens,
        "block_size": options.block_size,
        "num_blocks": options.num_blocks,
        "enable_prefix_caching": options.enable_prefix_caching,
        "enable_chunked_prefill": options.enable_chunked_prefill,
    }


def naive_vs_cb_command(
    profile: Annotated[
        str, typer.Option("--profile", help="Profile in configs/bench/profiles.yaml.")
    ] = "h100",
    profiles_path: Annotated[
        Path | None,
        typer.Option("--profiles", help="Alternative profiles file."),
    ] = None,
    arm: Annotated[
        list[str] | None,
        typer.Option("--arm", help="Restrict to these arms; repeatable."),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Override the profile's checkpoint.")
    ] = None,
    num_requests: Annotated[
        int | None, typer.Option("--num-requests", min=1, help="Override the request count.")
    ] = None,
    concurrency: Annotated[
        list[int] | None,
        typer.Option("--concurrency", min=1, help="Override the concurrency sweep; repeatable."),
    ] = None,
    output_tokens: Annotated[
        int | None,
        typer.Option(
            "--output-tokens",
            min=1,
            help=(
                "Completion length every request asks for; defaults to the mean of the "
                "profile's range. Uniform because the baselines cannot batch mixed limits."
            ),
        ),
    ] = None,
    warmup: Annotated[
        int, typer.Option("--warmup", min=0, help="Unmeasured requests sent before each run.")
    ] = 2,
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            help=(
                "OpenAI-compatible server for a remote arm (vllm, sglang). One server per "
                "invocation: run the scenario once per engine."
            ),
        ),
    ] = None,
    api_key: Annotated[
        str | None, typer.Option("--api-key", help="Bearer token for --url.")
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
    prefix_caching: Annotated[
        bool,
        typer.Option(
            "--prefix-caching/--no-prefix-caching",
            help="Prefix caching for the engine arms; off by default so the sweep is not warmed.",
        ),
    ] = False,
    request_timeout_s: Annotated[
        float | None, typer.Option("--request-timeout", min=0.0, help="Per-request deadline.")
    ] = None,
    local_files_only: Annotated[
        bool,
        typer.Option("--local-files-only/--allow-download", help="Never contact the Hub."),
    ] = False,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", help="Where result files are written.")
    ] = None,
    slo_ttft_ms: Annotated[float | None, typer.Option("--slo-ttft-ms", min=0.0)] = None,
    slo_tpot_ms: Annotated[float | None, typer.Option("--slo-tpot-ms", min=0.0)] = None,
    slo_e2e_ms: Annotated[float | None, typer.Option("--slo-e2e-ms", min=0.0)] = None,
) -> None:
    """Compare sequential, static-batched and continuously batched serving."""
    try:
        loaded = load_profile(profile, path=profiles_path)
    except (ProfileError, KeyError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from exc
    work = loaded.scenarios.naive_vs_cb
    options = EngineOptions(
        dtype=dtype or work.dtype,
        device=device or loaded.device,
        seed=loaded.seed,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
        num_blocks=kv_blocks,
        enable_prefix_caching=prefix_caching,
    )
    try:
        outcomes = asyncio.run(
            run_scenario(
                loaded,
                arms=arm,
                model=model,
                num_requests=num_requests,
                concurrencies=concurrency,
                output_tokens=output_tokens,
                warmup=warmup,
                url=url,
                api_key=api_key,
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
