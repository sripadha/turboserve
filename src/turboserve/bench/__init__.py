"""Benchmarking: load generation, metric aggregation, scenarios and report rendering.

The package is layered, and the layers are worth knowing before reading any one of them:

``profiles``
    How big a run is on a given machine -- models, request counts, lengths, concurrencies.
``prompts``
    What is sent: seeded synthetic prompts of exact token length, ShareGPT, or a file.
``loadgen``
    When it is sent: an open (Poisson) or closed (fixed concurrency) driver over anything
    implementing the gateway's ``Backend.generate`` contract.
``metrics``
    What is observed: one token stream becomes one record, and records become comparable
    summaries, ratios and deltas.
``records``
    The versioned on-disk schema every run is written in.
``report`` / ``plots``
    How the recorded runs become ``results/README.md``, ``docs/results.md`` and the PNGs
    beside them -- the only place a number in this repository is allowed to be printed.
``scenarios``
    The experiments themselves, one CLI subcommand each.

Names are re-exported lazily. ``loadgen`` imports the engine's types, which import torch,
and a tool that only wants to read a result file (the canary controller, the report
renderer, a CI check) should not pay several seconds for that.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from turboserve.bench.loadgen import (
        LoadResult,
        LoadSpec,
        build_requests,
        poisson_offsets,
        run_closed_loop,
        run_load,
        run_open_loop,
    )
    from turboserve.bench.metrics import (
        Comparison,
        RecordBuilder,
        compare_runs,
        group_by,
        run_label,
        summarize_by,
        summarize_records,
    )
    from turboserve.bench.profiles import BenchProfile, load_profile, load_profiles
    from turboserve.bench.prompts import (
        BenchPrompt,
        PromptSpec,
        SyntheticPromptBuilder,
        build_prompts,
    )
    from turboserve.bench.records import SLO, Percentiles, RequestRecord, RunResult
    from turboserve.bench.report import render_index, render_run, results_app

__all__ = [
    "SLO",
    "BenchProfile",
    "BenchPrompt",
    "Comparison",
    "LoadResult",
    "LoadSpec",
    "Percentiles",
    "PromptSpec",
    "RecordBuilder",
    "RequestRecord",
    "RunResult",
    "SyntheticPromptBuilder",
    "bench_app",
    "build_prompts",
    "build_requests",
    "compare_runs",
    "group_by",
    "load_profile",
    "load_profiles",
    "poisson_offsets",
    "render_index",
    "render_run",
    "results_app",
    "run_closed_loop",
    "run_label",
    "run_load",
    "run_open_loop",
    "summarize_by",
    "summarize_records",
]

#: Which submodule each exported name lives in, resolved on first attribute access.
_EXPORTS: dict[str, str] = {
    "BenchProfile": "turboserve.bench.profiles",
    "BenchPrompt": "turboserve.bench.prompts",
    "Comparison": "turboserve.bench.metrics",
    "LoadResult": "turboserve.bench.loadgen",
    "LoadSpec": "turboserve.bench.loadgen",
    "Percentiles": "turboserve.bench.records",
    "PromptSpec": "turboserve.bench.prompts",
    "RecordBuilder": "turboserve.bench.metrics",
    "RequestRecord": "turboserve.bench.records",
    "RunResult": "turboserve.bench.records",
    "SLO": "turboserve.bench.records",
    "SyntheticPromptBuilder": "turboserve.bench.prompts",
    "bench_app": "turboserve.bench.cli",
    "build_prompts": "turboserve.bench.prompts",
    "build_requests": "turboserve.bench.loadgen",
    "compare_runs": "turboserve.bench.metrics",
    "group_by": "turboserve.bench.metrics",
    "load_profile": "turboserve.bench.profiles",
    "load_profiles": "turboserve.bench.profiles",
    "poisson_offsets": "turboserve.bench.loadgen",
    "render_index": "turboserve.bench.report",
    "render_run": "turboserve.bench.report",
    "results_app": "turboserve.bench.report",
    "run_closed_loop": "turboserve.bench.loadgen",
    "run_label": "turboserve.bench.metrics",
    "run_load": "turboserve.bench.loadgen",
    "run_open_loop": "turboserve.bench.loadgen",
    "summarize_by": "turboserve.bench.metrics",
    "summarize_records": "turboserve.bench.metrics",
}


def __getattr__(name: str) -> Any:
    """Import the submodule that owns ``name`` on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    """Make the lazy names discoverable by ``dir()`` and by tab completion."""
    return sorted(__all__)
