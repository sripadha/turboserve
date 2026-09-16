"""Scenario 5: steady offered load through a fleet that is being broken on purpose.

This scenario owns no fault injection of its own. Everything it needs already exists in
:mod:`turboserve.chaos`: a fault-schedule grammar, replicas that can really be SIGKILLed,
and a harness that drives open-loop load through the gateway's real router while applying
the schedule. What the scenario adds is the *benchmark* framing -- the fleet size, arrival
rate, duration and request shape come from a bench profile instead of from flags, the
result lands under ``results/chaos/`` beside the other scenarios' files, and its ``config``
carries the label and backend keys the report renderer groups tables by.

Delegation rather than duplication is deliberate. A second implementation of the fault
timeline would be a second thing to keep honest, and the one number this scenario exists to
produce -- the error rate a client sees while replicas are dying -- must be computed by the
same code the chaos tests pin.

Load is open-loop (Poisson arrivals at the profile's rate) because that is the only driver
that can show a fault at all: a closed loop slows down exactly when the fleet does, so
queueing never grows and the tail never moves. Offered load is held constant and the
question is what the gateway manages to hide.

The replicas are mock servers, which is why the result file records
``replica_engine: "mock"``. The experiment is about the *gateway's* retry, health and
routing behaviour under failure; putting a real 7B checkpoint behind each replica would
measure a GPU, not a failure policy, and would need three of them.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from turboserve.bench.profiles import ProfileError, load_profile
from turboserve.bench.scenarios.common import (
    ArmOutcome,
    ScenarioError,
    arm_table,
    default_results_dir,
    gpu_price,
    resolve_slo,
    result_path,
)
from turboserve.chaos.faults import FaultError, FaultSchedule
from turboserve.chaos.harness import ChaosHarness, ChaosSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from turboserve.bench.profiles import BenchProfile
    from turboserve.chaos.harness import ChaosReport, WorkerMode

logger = logging.getLogger(__name__)

__all__ = ["SCENARIO", "build_spec", "chaos_command", "run_scenario"]

SCENARIO = "chaos"

#: Shown as the arm name in the rendered table when the run has no faults at all, so a
#: control run is not silently indistinguishable from a broken one.
NO_FAULTS_LABEL = "no faults"


def _parse_schedule(specs: Sequence[str] | None, *, fault_interval_s: float) -> FaultSchedule:
    """The fault schedule: the caller's, or the profile's kill cadence as a default.

    A profile sizes the experiment, including how often something dies; expressing that as
    an actual ``kill:every=Ns`` specification rather than as a hidden behaviour means the
    schedule printed in the result file is the schedule that ran.
    """
    if specs:
        try:
            return FaultSchedule.parse(specs)
        except FaultError as exc:
            raise ScenarioError(str(exc)) from exc
    return FaultSchedule.parse([f"kill:every={fault_interval_s:g}s"])


def build_spec(
    profile: BenchProfile,
    *,
    faults: Sequence[str] | None = None,
    replicas: int | None = None,
    duration_s: float | None = None,
    rate_rps: float | None = None,
    mode: WorkerMode = "subprocess",
    model: str = "mock-model",
    base_port: int = 9100,
    max_attempts: int = 3,
    health_ttl_s: float = 1.0,
    ttft_ms: float = 20.0,
    itl_ms: float = 5.0,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    slo_e2e_ms: float | None = None,
) -> ChaosSpec:
    """Turn a profile's ``chaos`` sizing plus overrides into a :class:`ChaosSpec`.

    The profile's *model* is deliberately not used: the replicas are mock servers, and
    naming a checkpoint in a run that never loaded one would misdescribe the file.
    """
    work = profile.scenarios.chaos
    schedule = _parse_schedule(faults, fault_interval_s=work.fault_interval_s)
    try:
        return ChaosSpec(
            faults=schedule,
            replicas=replicas if replicas is not None else work.num_workers,
            duration_s=duration_s if duration_s is not None else work.duration_s,
            rate_rps=rate_rps if rate_rps is not None else work.rate_rps,
            mode=mode,
            model=model,
            seed=profile.seed,
            profile=profile.name,
            input_tokens=work.input_tokens.as_tuple(),
            output_tokens=work.output_tokens.as_tuple(),
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            max_attempts=max_attempts,
            health_ttl_s=health_ttl_s,
            base_port=base_port,
            slo=resolve_slo(work.slo, ttft_ms=slo_ttft_ms, tpot_ms=slo_tpot_ms, e2e_ms=slo_e2e_ms),
            gpu_price_per_hour=gpu_price()[0],
            price_source=gpu_price()[1],
        )
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc


def _label(spec: ChaosSpec) -> str:
    """A short arm name describing the faults that were applied."""
    described = spec.faults.describe()
    return described or NO_FAULTS_LABEL


def _decorate(report: ChaosReport, spec: ChaosSpec) -> dict[str, Any]:
    """Add the renderer's conventions to the harness's result, and pull out the figures.

    The harness writes a complete, self-describing ``config``; what it does not write are
    the three keys the report renderer uses to name a row (``label``), colour it by backend
    (``backend``) and place the load level on it (``load``), because those are benchmark
    conventions rather than chaos ones.
    """
    report.run.config.update(
        {
            "label": _label(spec),
            "backend": "mock",
            "load": {
                "mode": "open",
                "concurrency": spec.replicas,
                "rate_rps": spec.rate_rps,
                "duration_s": spec.duration_s,
                "seed": spec.seed,
                "backend": "mock",
            },
        }
    )
    chaos_block: Mapping[str, Any] = report.chaos
    derived: dict[str, Any] = {
        "replicas": spec.replicas,
        "faults": _label(spec),
        "retries": report.retries,
        "disruptions": len(report.disruptions),
        "requests_never_routed": chaos_block.get("requests_never_routed", 0),
        "requests_during_faults": chaos_block.get("requests_during_faults", 0),
        "impaired_seconds": chaos_block.get("impaired_seconds", 0.0),
    }
    recovery = chaos_block.get("recovery_s")
    if isinstance(recovery, dict):
        derived["recovery_s_p95"] = recovery.get("p95")
    report.run.summary["derived"] = derived
    return derived


async def run_scenario(
    profile: BenchProfile,
    *,
    spec: ChaosSpec | None = None,
    results_dir: Path | None = None,
    out: Path | None = None,
    **spec_kwargs: Any,
) -> ArmOutcome:
    """Run the chaos experiment and write its result file.

    Either pass a fully built ``spec`` (tests do, so they can inject a tiny one) or let the
    profile and ``spec_kwargs`` build it.
    """
    resolved = spec if spec is not None else build_spec(profile, **spec_kwargs)
    logger.info(
        "chaos: replicas=%d rate=%.3g rps duration=%.3gs faults=%s",
        resolved.replicas,
        resolved.rate_rps,
        resolved.duration_s,
        _label(resolved),
    )
    report = await ChaosHarness(resolved).run()
    derived = _decorate(report, resolved)
    base_dir = results_dir if results_dir is not None else default_results_dir()
    target = out if out is not None else result_path(SCENARIO, base_dir, label=_label(resolved))
    path = report.save(target)
    return ArmOutcome(
        label=_label(resolved),
        backend="mock",
        path=path,
        summary=dict(report.run.summary),
        derived=derived,
    )


def chaos_command(
    profile: Annotated[
        str, typer.Option("--profile", help="Profile in configs/bench/profiles.yaml.")
    ] = "h100",
    profiles_path: Annotated[
        Path | None, typer.Option("--profiles", help="Alternative profiles file.")
    ] = None,
    faults: Annotated[
        list[str] | None,
        typer.Option(
            "--faults",
            help=(
                "Fault specification, repeatable; defaults to the profile's kill cadence. "
                "kill:every=10s[,grace=5s,restart=2s,target=replica-0], latency:p=0.05,ms=500, "
                "error:p=0.01, partition:at=20s,for=5s."
            ),
        ),
    ] = None,
    replicas: Annotated[
        int | None, typer.Option("--replicas", min=1, help="Override the fleet size.")
    ] = None,
    duration: Annotated[
        float | None, typer.Option("--duration", min=0.0, help="Override the run length (s).")
    ] = None,
    rps: Annotated[
        float | None, typer.Option("--rps", min=0.0, help="Override the arrival rate.")
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help=(
                "'subprocess' runs each replica as a real HTTP server that can be SIGKILLed; "
                "'inprocess' is faster but cannot exercise the HTTP client's error handling."
            ),
        ),
    ] = "subprocess",
    model: Annotated[
        str, typer.Option("--model", help="Model name the mock replicas serve.")
    ] = "mock-model",
    base_port: Annotated[int, typer.Option("--base-port", min=1, max=65535)] = 9100,
    max_attempts: Annotated[int, typer.Option("--max-attempts", min=1)] = 3,
    health_ttl_s: Annotated[float, typer.Option("--health-ttl", min=0.0)] = 1.0,
    ttft_ms: Annotated[float, typer.Option("--ttft-ms", min=0.0)] = 20.0,
    itl_ms: Annotated[float, typer.Option("--itl-ms", min=0.0)] = 5.0,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", help="Where result files are written.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Exact result file path, overriding --results-dir.")
    ] = None,
    slo_ttft_ms: Annotated[float | None, typer.Option("--slo-ttft-ms", min=0.0)] = None,
    slo_tpot_ms: Annotated[float | None, typer.Option("--slo-tpot-ms", min=0.0)] = None,
    slo_e2e_ms: Annotated[float | None, typer.Option("--slo-e2e-ms", min=0.0)] = None,
) -> None:
    """Drive steady load through a replica fleet while faults are injected into it."""
    try:
        loaded = load_profile(profile, path=profiles_path)
    except (ProfileError, KeyError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from exc
    if mode not in ("inprocess", "subprocess"):
        raise typer.BadParameter(
            f"--mode must be 'inprocess' or 'subprocess', got {mode!r}", param_hint="--mode"
        )
    worker_mode: WorkerMode = "inprocess" if mode == "inprocess" else "subprocess"
    try:
        outcome = asyncio.run(
            run_scenario(
                loaded,
                results_dir=results_dir,
                out=out,
                faults=faults,
                replicas=replicas,
                duration_s=duration,
                rate_rps=rps,
                mode=worker_mode,
                model=model,
                base_port=base_port,
                max_attempts=max_attempts,
                health_ttl_s=health_ttl_s,
                ttft_ms=ttft_ms,
                itl_ms=itl_ms,
                slo_ttft_ms=slo_ttft_ms,
                slo_tpot_ms=slo_tpot_ms,
                slo_e2e_ms=slo_e2e_ms,
            )
        )
    except ScenarioError as exc:
        raise typer.BadParameter(str(exc)) from exc
    arm_table(f"{SCENARIO} ({loaded.name})", [outcome])
