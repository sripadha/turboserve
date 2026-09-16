"""SLO-gated progressive delivery: sliding-window lane statistics and the canary controller.

Import surface:

``controller``
    :class:`CanaryController`, the pure state machine, plus its configuration, decisions and
    per-lane sliding windows. Nothing in it performs I/O, so the gateway can embed one.
``prometheus``
    Turns four PromQL queries over the gateway's metrics into the same
    :class:`LaneSummary` the in-process windows produce.
``k8s``
    ``kubectl`` drivers (Argo Rollouts or two Deployments behind weighted Services), the
    control loop that connects controller to cluster, and the ``turboserve canary`` CLI.

The controller and the Prometheus client are re-exported here. ``canary_app`` is resolved
lazily through :func:`__getattr__` so that importing the controller inside the gateway does
not drag in typer, rich and the subprocess machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from turboserve.canary.controller import (
    CanaryConfig,
    CanaryController,
    CanaryError,
    CanaryState,
    Decision,
    DecisionKind,
    Lane,
    LaneSummary,
    LaneWindow,
    Sample,
)
from turboserve.canary.prometheus import (
    PrometheusClient,
    PrometheusError,
    PrometheusLaneSource,
    PrometheusSettings,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from turboserve.canary.k8s import canary_app

__all__ = [
    "CanaryConfig",
    "CanaryController",
    "CanaryError",
    "CanaryState",
    "Decision",
    "DecisionKind",
    "Lane",
    "LaneSummary",
    "LaneWindow",
    "PrometheusClient",
    "PrometheusError",
    "PrometheusLaneSource",
    "PrometheusSettings",
    "Sample",
    "canary_app",
]

_LAZY: dict[str, str] = {"canary_app": "turboserve.canary.k8s"}


def __getattr__(name: str) -> Any:
    """Resolve the CLI app on first use, keeping the library import light."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
