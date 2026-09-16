"""``turboserve bench loadgen``: point the load generator at a running server.

The other scenarios own an experiment -- they build the thing under test, sweep it and
write one file per arm. This command owns none: it takes a URL, sends load at it, and
writes one result file. That makes it the tool for the cases where the system under test is
already running and was started by someone else:

* the Kubernetes end-to-end job, which installs the chart and then drives the deployed
  gateway while pods are being deleted underneath it;
* a smoke test against a vLLM server before committing GPU hours to a full scenario;
* any ad-hoc "is this deployment healthy under load" question.

The driver follows from the flags, and the rule is worth stating because it changes what
the numbers mean. With ``--rps`` the load is **open loop**: arrivals are a Poisson process
at that rate and the client does not slow down when the server does, so queueing delay and
the latency tail are free to grow -- which is the only way a saturated or partially broken
service looks different from a healthy one. ``--concurrency`` then only caps the number of
simultaneous requests, as a safety valve against an unresponsive server accumulating
unbounded tasks, and is recorded in the result when it is used. Without ``--rps`` the load
is **closed loop** at ``--concurrency``: a capacity measurement, where latency cannot run
away because the client is the brake.

Prompts are synthetic token ids by default and no tokenizer is loaded. A server addressed
by URL may be serving a model whose tokenizer is not on this machine at all (the mock
gateway in the kind job serves no real model), and the load generator's job here is to
produce requests of the right *shape*. Pass ``--tokenizer`` to build prompts from a real
vocabulary, which is required for ``--prompt-set sharegpt`` and ``--prompt-set file``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer

from turboserve.bench.loadgen import build_requests, run_load
from turboserve.bench.records import RunResult
from turboserve.bench.scenarios.common import (
    ScenarioError,
    build_load_spec,
    build_prompt_pool,
    close_backend,
    gpu_price,
    load_tokenizer,
    normalise_base_url,
    openai_backend,
    resolve_slo,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from turboserve.bench.prompts import BenchPrompt
    from turboserve.bench.records import SLO

logger = logging.getLogger(__name__)

__all__ = ["SCENARIO", "LoadgenOutcome", "loadgen_command", "parse_token_range", "run_loadgen"]

SCENARIO = "loadgen"

#: Upper bound on the generated prompt pool. An open-loop run of an hour at a high rate
#: does not need a distinct prompt per request -- the pool is cycled with unique ids -- and
#: building a million prompts would cost more than the run.
MAX_POOL = 1024

PromptSet = Literal["synthetic", "sharegpt", "file"]


class LoadgenOutcome:
    """What one ``loadgen`` invocation produced."""

    __slots__ = ("path", "run")

    def __init__(self, run: RunResult, path: Path) -> None:
        self.run = run
        self.path = path

    @property
    def summary(self) -> dict[str, Any]:
        """The run's computed summary."""
        return self.run.summary

    @property
    def error_rate(self) -> float:
        """Fraction of requests that failed, the figure the end-to-end job asserts on."""
        value = self.run.summary.get("error_rate", 0.0)
        return float(value) if isinstance(value, int | float) else 0.0


def parse_token_range(text: str, *, option: str) -> tuple[int, int]:
    """Parse ``"128"`` or ``"128:512"`` into an inclusive range.

    A single number is a fixed length; a colon-separated pair is sampled uniformly, which
    is what makes a run's request mix resemble real traffic rather than one point of a
    curve.
    """
    raw = text.strip()
    parts = raw.split(":") if ":" in raw else [raw, raw]
    if len(parts) != 2:
        raise ScenarioError(f"{option} must be N or MIN:MAX, got {text!r}")
    try:
        low, high = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ScenarioError(f"{option} must be N or MIN:MAX, got {text!r}") from exc
    if low < 1 or high < low:
        raise ScenarioError(f"{option} must satisfy 1 <= min <= max, got {text!r}")
    return low, high


def _pool_size(
    requested: int | None,
    *,
    rate_rps: float | None,
    duration_s: float | None,
    concurrency: int,
) -> int:
    """How many distinct prompts to build when the caller did not say."""
    if requested is not None:
        return requested
    if rate_rps is not None and duration_s is not None:
        return max(1, min(MAX_POOL, math.ceil(rate_rps * duration_s)))
    if duration_s is not None:
        return max(1, min(MAX_POOL, concurrency * 8))
    return max(1, min(MAX_POOL, concurrency * 4))


def _prompts(
    *,
    prompt_set: PromptSet,
    count: int,
    input_tokens: tuple[int, int],
    output_tokens: tuple[int, int],
    seed: int,
    tenant: str,
    shared_prefix_tokens: int,
    tokenizer: Any | None,
    path: Path | None,
) -> list[BenchPrompt]:
    """Build the prompt pool for the requested source."""
    if prompt_set == "synthetic":
        return build_prompt_pool(
            count=count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            seed=seed,
            tenants=(tenant,),
            shared_prefix_tokens=shared_prefix_tokens,
            tokenizer=tokenizer,
            id_prefix=SCENARIO,
        )
    if path is None:
        raise ScenarioError(f"--prompt-set {prompt_set} needs --prompt-path")
    if tokenizer is None:
        raise ScenarioError(
            f"--prompt-set {prompt_set} needs --tokenizer: the prompts are text and their "
            "token lengths cannot be known without one"
        )
    from turboserve.bench.prompts import PromptError, build_prompts
    from turboserve.bench.prompts import PromptSpec as _PromptSpec

    spec = _PromptSpec(
        source=prompt_set,
        count=count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        shared_prefix_tokens=shared_prefix_tokens,
        path=path,
        seed=seed,
        tenants=(tenant,),
    )
    try:
        return build_prompts(spec, tokenizer)
    except PromptError as exc:
        raise ScenarioError(str(exc)) from exc


async def run_loadgen(
    *,
    url: str,
    out: Path,
    rate_rps: float | None = None,
    duration_s: float | None = None,
    concurrency: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
    prompt_set: PromptSet = "synthetic",
    prompt_path: Path | None = None,
    tokenizer_id: str | None = None,
    num_requests: int | None = None,
    input_tokens: tuple[int, int] = (128, 128),
    output_tokens: tuple[int, int] = (64, 64),
    shared_prefix_tokens: int = 0,
    seed: int = 0,
    tenant: str = "bench",
    profile_name: str = "adhoc",
    label: str = "",
    request_timeout_s: float | None = None,
    warmup: int = 0,
    update_index: bool = True,
    local_files_only: bool = False,
    slo: SLO | None = None,
) -> LoadgenOutcome:
    """Drive load at ``url`` and write the result to ``out``."""
    if rate_rps is None and concurrency is None:
        raise ScenarioError("give --rps for an open-loop run, or --concurrency for a closed one")
    if rate_rps is not None and duration_s is None:
        raise ScenarioError("--rps needs --duration, or the run would never end")
    base_url = normalise_base_url(url)
    backend = openai_backend(base_url, name="gateway", api_key=api_key)
    try:
        served = model or await _first_model(backend)
        level = concurrency if concurrency is not None else 1
        count = _pool_size(
            num_requests, rate_rps=rate_rps, duration_s=duration_s, concurrency=level
        )
        tokenizer = (
            load_tokenizer(tokenizer_id, local_files_only=local_files_only)
            if tokenizer_id
            else None
        )
        pool = _prompts(
            prompt_set=prompt_set,
            count=count + warmup,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            seed=seed,
            tenant=tenant,
            shared_prefix_tokens=shared_prefix_tokens,
            tokenizer=tokenizer,
            path=prompt_path,
        )
        warm_prompts = pool[:warmup]
        measured = pool[warmup:]
        spec = build_load_spec(
            mode="open" if rate_rps is not None else "closed",
            concurrency=level,
            rate_rps=rate_rps,
            duration_s=duration_s,
            seed=seed,
            backend_name="gateway",
            request_timeout_s=request_timeout_s,
            repeat_requests=rate_rps is not None,
        )
        requests = build_requests(measured, model=served, id_prefix=f"{SCENARIO}-")
        warm_requests = build_requests(warm_prompts, model=served, id_prefix=f"{SCENARIO}-warm-")
        logger.info(
            "loadgen: url=%s model=%s mode=%s pool=%d", base_url, served, spec.mode, len(requests)
        )
        load = await run_load(requests, backend.generate, spec=spec, warmup=warm_requests)
    finally:
        await close_backend(backend)

    price, source = gpu_price()
    run = RunResult.start(
        SCENARIO,
        profile_name,
        config={
            "label": label or f"{spec.mode} loop",
            "backend": "gateway",
            "scenario": SCENARIO,
            "url": base_url,
            "model": served,
            "seed": seed,
            "tenants": [tenant],
            "prompt_set": prompt_set,
            "input_tokens": list(input_tokens),
            "output_tokens": list(output_tokens),
            "shared_prefix_tokens": shared_prefix_tokens,
            "num_prompts": len(requests),
            "load": {
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
            },
        },
        gpu_price_per_hour=price,
        price_source=source,
    )
    load.into(run)
    run.finish(slo=slo)
    run.summary["derived"] = {
        "requests_sent": load.num_sent,
        "max_in_flight_observed": load.max_in_flight_observed,
        "warmup_requests": len(warm_requests),
    }
    path = run.save(out, update_index=update_index)
    return LoadgenOutcome(run, path)


async def _first_model(backend: Any) -> str:
    """The server's first advertised model, used when ``--model`` was not given."""
    try:
        names = await backend.models()
    except Exception as exc:  # noqa: BLE001 - surfaced as a scenario error with the cause
        raise ScenarioError(f"cannot list models at {backend.base_url}: {exc}") from exc
    if not names:
        raise ScenarioError(
            f"{backend.base_url} advertises no models; pass --model to name one explicitly"
        )
    return str(names[0])


def loadgen_command(
    url: Annotated[
        str, typer.Option("--url", help="Base URL of an OpenAI-compatible server.")
    ] = "http://127.0.0.1:8000",
    rps: Annotated[
        float | None,
        typer.Option("--rps", min=0.0, help="Open-loop arrival rate; needs --duration."),
    ] = None,
    duration: Annotated[
        float | None, typer.Option("--duration", min=0.0, help="How long to drive load, seconds.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Result JSON to write.")] = Path(
        "results/loadgen/run.json"
    ),
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency",
            min=1,
            help="Closed-loop concurrency; with --rps it caps simultaneous requests instead.",
        ),
    ] = None,
    prompt_set: Annotated[
        str, typer.Option("--prompt-set", help="synthetic | sharegpt | file.")
    ] = "synthetic",
    prompt_path: Annotated[
        Path | None, typer.Option("--prompt-path", help="Input file for sharegpt/file.")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Model name; defaults to the server's first.")
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key", help="Bearer token.")] = None,
    tokenizer: Annotated[
        str | None,
        typer.Option("--tokenizer", help="Tokenizer for real-vocabulary prompts."),
    ] = None,
    num_requests: Annotated[
        int | None, typer.Option("--num-requests", min=1, help="Size of the prompt pool.")
    ] = None,
    input_tokens: Annotated[
        str, typer.Option("--input-tokens", help="Prompt length, N or MIN:MAX.")
    ] = "128",
    output_tokens: Annotated[
        str, typer.Option("--output-tokens", help="Completion length, N or MIN:MAX.")
    ] = "64",
    shared_prefix_tokens: Annotated[
        int, typer.Option("--shared-prefix-tokens", min=0, help="Prefix every prompt shares.")
    ] = 0,
    seed: Annotated[int, typer.Option("--seed", help="Seed for prompts and arrivals.")] = 0,
    tenant: Annotated[str, typer.Option("--tenant", help="Tenant id on every request.")] = "bench",
    profile_name: Annotated[
        str, typer.Option("--profile-name", help="Profile name recorded in the result.")
    ] = "adhoc",
    label: Annotated[str, typer.Option("--label", help="Arm name in rendered tables.")] = "",
    warmup: Annotated[
        int, typer.Option("--warmup", min=0, help="Unmeasured requests sent first.")
    ] = 0,
    request_timeout_s: Annotated[
        float | None, typer.Option("--request-timeout", min=0.0, help="Per-request deadline.")
    ] = None,
    update_index: Annotated[
        bool, typer.Option("--index/--no-index", help="Append to the results index.")
    ] = True,
    local_files_only: Annotated[
        bool, typer.Option("--local-files-only/--allow-download", help="Never contact the Hub.")
    ] = False,
    slo_ttft_ms: Annotated[float | None, typer.Option("--slo-ttft-ms", min=0.0)] = None,
    slo_tpot_ms: Annotated[float | None, typer.Option("--slo-tpot-ms", min=0.0)] = None,
    slo_e2e_ms: Annotated[float | None, typer.Option("--slo-e2e-ms", min=0.0)] = None,
) -> None:
    """Drive load at a running OpenAI-compatible server and write a result file."""
    if prompt_set not in ("synthetic", "sharegpt", "file"):
        raise typer.BadParameter(
            f"--prompt-set must be synthetic, sharegpt or file, got {prompt_set!r}",
            param_hint="--prompt-set",
        )
    source: PromptSet = prompt_set  # type: ignore[assignment]
    try:
        outcome = asyncio.run(
            run_loadgen(
                url=url,
                out=out,
                rate_rps=rps,
                duration_s=duration,
                concurrency=concurrency,
                model=model,
                api_key=api_key,
                prompt_set=source,
                prompt_path=prompt_path,
                tokenizer_id=tokenizer,
                num_requests=num_requests,
                input_tokens=parse_token_range(input_tokens, option="--input-tokens"),
                output_tokens=parse_token_range(output_tokens, option="--output-tokens"),
                shared_prefix_tokens=shared_prefix_tokens,
                seed=seed,
                tenant=tenant,
                profile_name=profile_name,
                label=label,
                request_timeout_s=request_timeout_s,
                warmup=warmup,
                update_index=update_index,
                local_files_only=local_files_only,
                slo=resolve_slo(None, ttft_ms=slo_ttft_ms, tpot_ms=slo_tpot_ms, e2e_ms=slo_e2e_ms),
            )
        )
    except ScenarioError as exc:
        raise typer.BadParameter(str(exc)) from exc
    summary = outcome.summary
    typer.echo(
        f"{summary.get('num_requests', 0)} request(s), "
        f"{summary.get('num_failed', 0)} failed, "
        f"error rate {outcome.error_rate:.4%}"
    )
    typer.echo(f"wrote {outcome.path}")
