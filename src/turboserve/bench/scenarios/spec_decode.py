"""Scenario 3: what speculative decoding is worth, and what it costs when it misses.

The question is narrow and the design follows from it: *for the same prompts, the same
target model and the same sampling, how many tokens per second does the engine produce with
speculation and without it, and what fraction of drafted tokens survives?* Everything else is
held fixed so that the comparison means something.

* **Same prompts.** One seeded pool per target/draft pair, built through
  :func:`turboserve.bench.scenarios.common.build_prompt_pool` and sent as token ids to every
  arm of that pair. Two arms that drew different prompts are not comparable.
* **Same sampling.** Greedy with ``ignore_eos``, which is
  :func:`turboserve.bench.loadgen.build_requests`'s default. Greedy is right here for two
  reasons: every request then produces exactly the number of output tokens the profile asked
  for, so throughput is measured over the workload that was specified; and greedy
  verification is deterministic, so a change in acceptance rate between runs is a change in
  the models, not in the dice.
* **Same client.** Every arm is driven through the gateway's backend interface by
  :mod:`turboserve.bench.loadgen`, so the reference engine and a vLLM server configured with
  its own speculative decoding are measured by the same code against the same clock.

**The arms.** For each pair in the profile, a baseline arm runs the target alone and one arm
per ``k`` runs it with the pair's drafter (a draft model, or prompt lookup for a pair whose
``drafter`` is ``ngram``). Every arm of a pair carries the same ``baseline_label``, so the
report renderer can still compute relative figures if one arm's file is missing.

**What lands in the result.** The standard record set -- per-request TTFT, ITL, TPOT and E2E
from the client's clock -- plus, under ``summary["derived"]``, the engine's own speculation
counters: acceptance rate, mean accepted length, drafted and accepted token counts and how
many target forward passes actually ran. Those come from
:meth:`turboserve.engine.spec.spec_engine.SpeculativeLLMEngine.spec_stats` and are what
explains the throughput figure next to them. An arm served by a remote server has no derived
block: its acceptance rate is not visible from this process and the scenario does not invent
one.

Nothing here has been run on a GPU in this repository; see ``docs/benchmarking.md`` for how
results are produced and ``docs/speculative-decoding.md`` for the algorithm.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer

from turboserve.bench.loadgen import build_requests, run_load
from turboserve.bench.profiles import ProfileError, SpecDecodePair, load_profile
from turboserve.bench.scenarios.common import (
    ArmOutcome,
    EngineOptions,
    ScenarioError,
    arm_table,
    build_engine_config,
    build_load_spec,
    build_prompt_pool,
    close_backend,
    engine_derived_stats,
    load_tokenizer,
    normalise_base_url,
    open_run,
    openai_backend,
    reference_backend,
    resolve_slo,
    result_path,
    write_run,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence as SequenceABC
    from datetime import datetime

    from turboserve.bench.profiles import BenchProfile, SpecDecodeProfile
    from turboserve.bench.prompts import BenchPrompt
    from turboserve.bench.records import SLO
    from turboserve.bench.scenarios.common import AnyBackend

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_LABEL",
    "SCENARIO",
    "SPEC_STAT_KEYS",
    "Arm",
    "ArmMethod",
    "build_backend",
    "plan_arms",
    "run_arm",
    "run_scenario",
    "spec_decode_command",
]

SCENARIO = "spec_decode"
BASELINE_LABEL = "target only"
"""Setting name of the arm every other arm of the same pair is compared against."""

ArmMethod = Literal["none", "model", "ngram"]

SPEC_STAT_KEYS: tuple[str, ...] = (
    "drafter",
    "num_speculative_tokens",
    "num_spec_steps",
    "num_verified_seqs",
    "num_drafted",
    "num_accepted",
    "num_emitted",
    "num_target_forwards",
    "num_draft_calls",
    "acceptance_rate",
    "mean_accepted_len",
    "mean_emitted_len",
    "draft_forwards",
    "draft_tokens_proposed",
    "draft_starved",
    "draft_lookups",
    "draft_lookup_hits",
    "draft_lookup_hit_rate",
)
"""Engine counters this scenario adds to the shared ``derived`` block, when present."""


@dataclass(frozen=True, slots=True)
class Arm:
    """One measured configuration: a pair, a speculation setting and a concurrency.

    ``label`` names the pair as well as the setting, because two pairs measured in the same
    run would otherwise both be called "k=4" and the renderer deduplicates on the label.
    """

    pair: SpecDecodePair
    method: ArmMethod
    num_speculative_tokens: int
    concurrency: int

    @property
    def is_baseline(self) -> bool:
        """Whether this arm runs the target model on its own."""
        return self.method == "none"

    @property
    def setting(self) -> str:
        """The speculation setting, without the pair name."""
        if self.method == "none":
            return BASELINE_LABEL
        if self.method == "ngram":
            return f"ngram k={self.num_speculative_tokens}"
        return f"k={self.num_speculative_tokens}"

    @property
    def label(self) -> str:
        """Human-readable name of the arm, unique within a scenario run."""
        return f"{self.pair.name} / {self.setting}"

    @property
    def baseline_label(self) -> str:
        """Label of this arm's baseline, stamped into every arm of the pair."""
        return f"{self.pair.name} / {BASELINE_LABEL}"

    def speculative_config(self) -> dict[str, Any] | None:
        """The mapping handed to ``EngineConfig.speculative``, or ``None`` for the baseline."""
        if self.method == "none":
            return None
        if self.method == "ngram":
            return {"method": "ngram", "num_speculative_tokens": self.num_speculative_tokens}
        return {
            "method": "model",
            "draft_model": self.pair.draft,
            "num_speculative_tokens": self.num_speculative_tokens,
        }

    def to_dict(self) -> dict[str, Any]:
        """The arm's identity, embedded in its result file so a run explains itself."""
        return {
            "pair": self.pair.name,
            "target": self.pair.target,
            "draft": self.pair.draft,
            "drafter": self.pair.drafter,
            "method": self.method,
            "num_speculative_tokens": self.num_speculative_tokens,
            "concurrency": self.concurrency,
        }


def plan_arms(
    work: SpecDecodeProfile,
    *,
    pairs: SequenceABC[str] | None = None,
    k_values: SequenceABC[int] | None = None,
    concurrencies: SequenceABC[int] | None = None,
) -> list[Arm]:
    """Expand a profile into the arms to run, baseline first within each pair.

    The baseline comes first on purpose: it is the arm every other one is divided by, and
    running it while the machine is coldest is the conservative order -- any warm-up the
    device has not finished counts against speculation rather than for it.

    Raises :class:`ScenarioError` when a filter selects nothing, so a mistyped pair name
    fails immediately instead of producing an empty, plausible-looking result directory.
    """
    selected = list(work.pairs)
    if pairs:
        wanted = set(pairs)
        selected = [pair for pair in selected if pair.name in wanted]
        missing = wanted - {pair.name for pair in selected}
        if missing:
            known = ", ".join(pair.name for pair in work.pairs)
            raise ScenarioError(f"unknown pair(s) {sorted(missing)}; the profile has {known}")
    if not selected:
        raise ScenarioError("no target/draft pairs selected")
    ks = list(k_values) if k_values else list(work.speculative_tokens)
    if not ks:
        raise ScenarioError("no speculative token counts selected")
    if any(k < 1 for k in ks):
        raise ScenarioError(f"speculative token counts must be positive, got {ks}")
    loads = list(concurrencies) if concurrencies else list(work.concurrencies)
    if not loads:
        raise ScenarioError("no concurrencies selected")

    arms: list[Arm] = []
    for concurrency in loads:
        for pair in selected:
            method: ArmMethod = "ngram" if pair.drafter == "ngram" else "model"
            if method == "model" and not pair.draft:
                raise ScenarioError(f"pair {pair.name!r} has drafter 'model' but no draft model")
            arms.append(
                Arm(pair=pair, method="none", num_speculative_tokens=0, concurrency=concurrency)
            )
            arms.extend(
                Arm(pair=pair, method=method, num_speculative_tokens=k, concurrency=concurrency)
                for k in ks
            )
    return arms


# ---------------------------------------------------------------------------------------------
# Running one arm
# ---------------------------------------------------------------------------------------------


def build_backend(
    arm: Arm,
    options: EngineOptions,
    *,
    url: str | None = None,
    local_files_only: bool = False,
) -> AnyBackend:
    """The backend that serves one arm.

    With ``url`` the arm goes to an already-running OpenAI-compatible server. Otherwise an
    in-process engine is used: the baseline is left for
    :class:`~turboserve.gateway.backends.local_engine.LocalEngineBackend` to build lazily,
    while a speculative arm is constructed here, because choosing the
    :class:`~turboserve.engine.spec.spec_engine.SpeculativeLLMEngine` subclass is a decision
    the backend's own lazy construction cannot make. The backend still owns closing it.
    """
    if url:
        return openai_backend(normalise_base_url(url), name="vllm")
    config = build_engine_config(arm.pair.target, options)
    speculative = arm.speculative_config()
    if speculative is None:
        return reference_backend(config, local_files_only=local_files_only)

    from turboserve.engine.runtime.async_engine import AsyncLLMEngine
    from turboserve.engine.spec.spec_engine import SpeculativeLLMEngine
    from turboserve.gateway.backends.local_engine import LocalEngineBackend

    engine = SpeculativeLLMEngine(
        config.model_copy(update={"speculative": speculative}),
        local_files_only=local_files_only,
    )
    return LocalEngineBackend(
        name="reference",
        engine=AsyncLLMEngine(engine),
        served_models=[arm.pair.target],
    )


def derived_stats(backend: AnyBackend) -> dict[str, Any]:
    """Engine counters for the result's ``derived`` block: the shared ones plus speculation."""
    stats_fn = getattr(backend, "stats", None)
    derived = dict(engine_derived_stats(backend))
    if not callable(stats_fn):
        return derived
    values = stats_fn()
    if not isinstance(values, dict):
        return derived
    derived.update({key: values[key] for key in SPEC_STAT_KEYS if key in values})
    return derived


async def run_arm(
    arm: Arm,
    profile: BenchProfile,
    prompts: SequenceABC[BenchPrompt],
    *,
    options: EngineOptions,
    url: str | None = None,
    label: str | None = None,
    num_warmup: int = 2,
    request_timeout_s: float | None = None,
    local_files_only: bool = False,
    slo: SLO | None = None,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    now: datetime | None = None,
) -> ArmOutcome:
    """Measure one arm, write its result file, and return where it landed.

    ``label`` overrides the arm's own name, which is what a remote server needs: its
    speculative settings are fixed at its launch, so calling the run "k=4" because this
    process asked for ``k=4`` would be a claim about something this process did not control.
    """
    backend_name = "vllm" if url else "reference"
    backend = build_backend(arm, options, url=url, local_files_only=local_files_only)
    spec = build_load_spec(
        concurrency=arm.concurrency,
        seed=profile.seed,
        backend_name=backend_name,
        request_timeout_s=request_timeout_s,
    )
    arm_label = label or arm.label
    run = open_run(
        SCENARIO,
        profile.name,
        label=arm_label,
        backend=backend_name,
        baseline_label=arm.baseline_label if label is None else None,
        spec=spec,
        config={**profile.config_for(SCENARIO), **arm.to_dict()},
    )
    try:
        requests = build_requests(
            list(prompts), model=arm.pair.target, id_prefix=f"{_slugish(arm_label)}-"
        )
        warmup = build_requests(
            list(prompts[:num_warmup]),
            model=arm.pair.target,
            id_prefix=f"warmup-{_slugish(arm_label)}-",
        )
        result = await run_load(requests, backend.generate, spec=spec, warmup=warmup)
        result.into(run)
        derived = derived_stats(backend)
    finally:
        await close_backend(backend)
    path = write_run(
        run,
        out_dir / result_path(SCENARIO, results_dir, label=arm_label, now=now).name
        if out_dir is not None
        else result_path(SCENARIO, results_dir, label=arm_label, now=now),
        slo=slo,
        derived=derived,
    )
    logger.info(
        "arm %s finished: %d requests, %d failed",
        arm_label,
        int(run.summary.get("num_requests", 0)),
        int(run.summary.get("num_failed", 0)),
    )
    return ArmOutcome(
        label=arm_label,
        backend=backend_name,
        path=path,
        summary=dict(run.summary),
        derived=derived,
    )


def _slugish(label: str) -> str:
    """A compact request-id prefix derived from an arm label."""
    return "".join(char if char.isalnum() else "-" for char in label.lower()).strip("-")


# ---------------------------------------------------------------------------------------------
# Running the scenario
# ---------------------------------------------------------------------------------------------


async def run_scenario(
    profile: BenchProfile,
    arms: SequenceABC[Arm],
    *,
    options: EngineOptions,
    num_requests: int | None = None,
    url: str | None = None,
    label: str | None = None,
    num_warmup: int = 2,
    request_timeout_s: float | None = None,
    local_files_only: bool = False,
    slo: SLO | None = None,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
) -> list[ArmOutcome]:
    """Run every arm in order and return one outcome each.

    Prompt pools are built once per pair and reused by all of that pair's arms, which is the
    condition that makes the arms comparable at all.
    """
    work = profile.scenarios.spec_decode
    count = num_requests or work.num_requests
    pools: dict[str, list[BenchPrompt]] = {}
    outcomes: list[ArmOutcome] = []
    for arm in arms:
        if arm.pair.name not in pools:
            tokenizer = load_tokenizer(arm.pair.target, local_files_only=local_files_only)
            pools[arm.pair.name] = build_prompt_pool(
                count=count,
                input_tokens=work.input_tokens.as_tuple(),
                output_tokens=work.output_tokens.as_tuple(),
                seed=profile.seed,
                tenants=tuple(profile.tenants),
                tokenizer=tokenizer,
                id_prefix=f"{SCENARIO}-{arm.pair.name}",
            )
        outcomes.append(
            await run_arm(
                arm,
                profile,
                pools[arm.pair.name],
                options=options,
                url=url,
                label=label,
                num_warmup=num_warmup,
                request_timeout_s=request_timeout_s,
                local_files_only=local_files_only,
                slo=slo,
                results_dir=results_dir,
                out_dir=out_dir,
            )
        )
    return outcomes


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
#
# A plain command function rather than a ``typer.Typer``: ``turboserve.bench.cli`` mounts a
# Typer object as a sub-*group*, which would make the invocation ``bench spec-decode run``,
# and mounts a callable as a single command, which is the documented ``bench spec-decode``.
# Every other scenario in this package exports its command the same way.


def spec_decode_command(
    profile: Annotated[
        str, typer.Option("--profile", help="Profile in configs/bench/profiles.yaml.")
    ] = "h100",
    profiles_path: Annotated[
        Path | None, typer.Option("--profiles", help="Alternative profiles file.")
    ] = None,
    pair: Annotated[
        list[str] | None, typer.Option("--pair", help="Only these target/draft pairs.")
    ] = None,
    k: Annotated[
        list[int] | None,
        typer.Option("--k", help="Speculative token counts; defaults to the profile's."),
    ] = None,
    concurrency: Annotated[
        list[int] | None,
        typer.Option("--concurrency", help="Concurrencies; defaults to the profile's."),
    ] = None,
    num_requests: Annotated[
        int | None, typer.Option("--num-requests", min=1, help="Override the request count.")
    ] = None,
    warmup: Annotated[
        int, typer.Option("--warmup", min=0, help="Requests sent before the measured phase.")
    ] = 2,
    url: Annotated[
        str | None,
        typer.Option("--url", help="OpenAI-compatible server with its own speculative setup."),
    ] = None,
    backend: Annotated[
        str, typer.Option("--backend", help="reference (in-process engine) or vllm (--url).")
    ] = "reference",
    label: Annotated[
        str | None, typer.Option("--label", help="Arm name for a remote server's own settings.")
    ] = None,
    dtype: Annotated[
        str | None, typer.Option("--dtype", help="Override the profile's dtype.")
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="Override the profile's device.")
    ] = None,
    max_num_seqs: Annotated[int | None, typer.Option("--max-num-seqs", min=1)] = None,
    max_num_batched_tokens: Annotated[int, typer.Option("--max-num-batched-tokens", min=1)] = 2048,
    block_size: Annotated[int, typer.Option("--block-size", min=1)] = 16,
    kv_blocks: Annotated[
        int | None,
        typer.Option("--kv-blocks", min=1, help="Fix the KV pool size instead of profiling it."),
    ] = None,
    max_model_len: Annotated[int | None, typer.Option("--max-model-len", min=1)] = None,
    request_timeout_s: Annotated[
        float | None, typer.Option("--request-timeout", min=0.0, help="Per-request deadline.")
    ] = None,
    local_files_only: Annotated[
        bool, typer.Option("--local-files-only/--allow-download", help="Never contact the Hub.")
    ] = False,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", help="Base directory for result files.")
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Exact directory for this scenario's result files."),
    ] = None,
    slo_ttft_ms: Annotated[float | None, typer.Option("--slo-ttft-ms", min=0.0)] = None,
    slo_tpot_ms: Annotated[float | None, typer.Option("--slo-tpot-ms", min=0.0)] = None,
    slo_e2e_ms: Annotated[float | None, typer.Option("--slo-e2e-ms", min=0.0)] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the arms that would run and stop.")
    ] = False,
) -> None:
    """Compare the target model alone against the same model with speculative decoding.

    The default sweep comes from the profile: every target/draft pair, the baseline plus
    every ``k``, at every concurrency, one result file per arm. Narrow it with ``--pair``,
    ``--k`` and ``--concurrency`` when re-measuring one cell, and check the plan with
    ``--dry-run`` before committing a machine to it.
    """
    if backend not in ("reference", "vllm"):
        raise typer.BadParameter("--backend must be 'reference' or 'vllm'")
    if backend == "vllm" and not url:
        raise typer.BadParameter("--backend vllm needs --url")
    if url and backend != "vllm":
        raise typer.BadParameter("--url is only meaningful with --backend vllm")

    try:
        loaded = load_profile(profile, path=profiles_path)
    except (ProfileError, KeyError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from exc
    work = loaded.scenarios.spec_decode
    try:
        arms = plan_arms(work, pairs=pair, k_values=k, concurrencies=concurrency)
    except ScenarioError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if url:
        # A server's speculative settings are fixed at its launch, so there is exactly one
        # arm per pair and concurrency and the operator names it.
        arms = [arm for arm in arms if arm.is_baseline]

    if dry_run:
        _print_plan(loaded.name, arms)
        return

    options = EngineOptions(
        dtype=dtype or work.dtype,
        device=device or loaded.device,
        seed=loaded.seed,
        max_num_seqs=max_num_seqs or max(arm.concurrency for arm in arms),
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
        num_blocks=kv_blocks,
        max_model_len=max_model_len,
    )
    try:
        outcomes = asyncio.run(
            run_scenario(
                loaded,
                arms,
                options=options,
                num_requests=num_requests,
                url=url,
                label=label,
                num_warmup=warmup,
                request_timeout_s=request_timeout_s,
                local_files_only=local_files_only,
                slo=resolve_slo(
                    work.slo, ttft_ms=slo_ttft_ms, tpot_ms=slo_tpot_ms, e2e_ms=slo_e2e_ms
                ),
                results_dir=results_dir,
                out_dir=out,
            )
        )
    except ScenarioError as exc:
        raise typer.BadParameter(str(exc)) from exc
    arm_table(f"{SCENARIO} ({loaded.name})", outcomes)
    typer.echo("render the tables with `turboserve bench render`")


def _print_plan(profile_name: str, arms: SequenceABC[Arm]) -> None:
    """Show the arms a run would execute, without loading anything."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"{SCENARIO} arms for profile {profile_name}")
    for column in ("arm", "target", "draft", "method", "k", "concurrency"):
        table.add_column(column)
    for arm in arms:
        table.add_row(
            arm.label,
            arm.pair.target,
            arm.pair.draft or "-",
            arm.method,
            str(arm.num_speculative_tokens),
            str(arm.concurrency),
        )
    console = Console()
    console.print(table)
    console.print(f"[dim]{len(arms)} arms; nothing was run[/dim]")
