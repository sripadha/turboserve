"""Fault injection and the chaos harness that drives load through failing replicas.

Three layers, and it is worth knowing which one you want before opening any of them:

``faults``
    The schedule. Pure data: parses ``kill:every=10s`` and friends and expands them into a
    dated timeline of actions over a named fleet. No clock, no I/O, no dependencies beyond
    the standard library.
``worker``
    The breakable replica. A mock backend wrapped in a fault injector, either in this
    process or served over HTTP by a child process that can be ``SIGKILL``ed.
``harness``
    The experiment. Starts the fleet, puts the gateway's router in front of it, drives
    open-loop load through it while applying the timeline, and writes the result in the
    ordinary benchmark schema with retries, recovery times and per-window latency under
    ``summary["chaos"]``.

Only :mod:`turboserve.chaos.faults` is imported eagerly. The other two pull in the gateway,
the benchmark stack and (through them) torch, and a tool that only wants to parse or print a
schedule -- the CLI's ``chaos plan``, a config check, a test -- should not pay for that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from turboserve.chaos.faults import (
    Fault,
    FaultAction,
    FaultError,
    FaultEvent,
    FaultKind,
    FaultSchedule,
    SteadyFaults,
    outage_windows,
    parse_duration,
    parse_probability,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from turboserve.chaos.harness import (
        ChaosHarness,
        ChaosReport,
        ChaosSpec,
        Disruption,
        chaos_app,
    )
    from turboserve.chaos.worker import (
        ChaosWorker,
        FaultingBackend,
        InProcessWorker,
        SubprocessWorker,
        WorkerFaults,
        WorkerSpec,
    )

__all__ = [
    "ChaosHarness",
    "ChaosReport",
    "ChaosSpec",
    "ChaosWorker",
    "Disruption",
    "Fault",
    "FaultAction",
    "FaultError",
    "FaultEvent",
    "FaultKind",
    "FaultSchedule",
    "FaultingBackend",
    "InProcessWorker",
    "SteadyFaults",
    "SubprocessWorker",
    "WorkerFaults",
    "WorkerSpec",
    "chaos_app",
    "outage_windows",
    "parse_duration",
    "parse_probability",
]

_LAZY: dict[str, str] = {
    "ChaosHarness": "turboserve.chaos.harness",
    "ChaosReport": "turboserve.chaos.harness",
    "ChaosSpec": "turboserve.chaos.harness",
    "Disruption": "turboserve.chaos.harness",
    "chaos_app": "turboserve.chaos.harness",
    "ChaosWorker": "turboserve.chaos.worker",
    "FaultingBackend": "turboserve.chaos.worker",
    "InProcessWorker": "turboserve.chaos.worker",
    "SubprocessWorker": "turboserve.chaos.worker",
    "WorkerFaults": "turboserve.chaos.worker",
    "WorkerSpec": "turboserve.chaos.worker",
}


def __getattr__(name: str) -> Any:
    """Import the harness and worker layers on first use (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    """Include the lazily imported names in ``dir()`` and tab completion."""
    return sorted(__all__)
