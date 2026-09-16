"""Benchmark scenarios, each a CLI subcommand writing a versioned JSON result file.

Every module here answers one question about serving and writes its answer as
``results/<scenario>/<timestamp>-<arm>.json``; nothing in this package prints a number of
its own. The shared harness -- prompt pools, backend construction, the result-file
conventions the report renderer reads -- lives in :mod:`turboserve.bench.scenarios.common`.

Names are re-exported lazily (PEP 562). Importing a scenario pulls in the engine and
therefore torch, and a tool that only wants to know which scenarios exist, or to render a
result file that was written last week, should not pay for that.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from turboserve.bench.scenarios.chaos import chaos_command
    from turboserve.bench.scenarios.common import (
        ArmOutcome,
        BaselineBackend,
        EngineOptions,
        ScenarioError,
    )
    from turboserve.bench.scenarios.loadgen_cli import loadgen_command, run_loadgen
    from turboserve.bench.scenarios.naive_vs_cb import naive_vs_cb_command
    from turboserve.bench.scenarios.prefix_cache import prefix_cache_command

#: The scenarios implemented in this package, in the order they are meant to be read.
#: ``spec_decode`` and ``multi_lora`` are registered by :mod:`turboserve.bench.cli` when
#: their modules are present; they are owned by the engine's speculative and LoRA groups.
SCENARIO_MODULES: tuple[str, ...] = ("naive_vs_cb", "prefix_cache", "chaos")

__all__ = [
    "SCENARIO_MODULES",
    "ArmOutcome",
    "BaselineBackend",
    "EngineOptions",
    "ScenarioError",
    "chaos_command",
    "loadgen_command",
    "naive_vs_cb_command",
    "prefix_cache_command",
    "run_loadgen",
]

_EXPORTS: dict[str, str] = {
    "ArmOutcome": "turboserve.bench.scenarios.common",
    "BaselineBackend": "turboserve.bench.scenarios.common",
    "EngineOptions": "turboserve.bench.scenarios.common",
    "ScenarioError": "turboserve.bench.scenarios.common",
    "chaos_command": "turboserve.bench.scenarios.chaos",
    "loadgen_command": "turboserve.bench.scenarios.loadgen_cli",
    "naive_vs_cb_command": "turboserve.bench.scenarios.naive_vs_cb",
    "prefix_cache_command": "turboserve.bench.scenarios.prefix_cache",
    "run_loadgen": "turboserve.bench.scenarios.loadgen_cli",
}


def __getattr__(name: str) -> Any:
    """Import the module that owns ``name`` on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    """Make the lazy names visible to ``dir()`` and tab completion."""
    return sorted(__all__)
