"""Drive a real rollout: kubectl for the effects, Prometheus for the evidence, and the CLI.

:mod:`turboserve.canary.controller` decides; this module acts. It contains four things:

``Kubectl``
    A thin, auditable wrapper over the ``kubectl`` binary. Every command is logged before
    it runs and kept in :attr:`Kubectl.commands`, and ``dry_run=True`` executes reads while
    refusing to execute anything that changes the cluster. The subprocess call is behind an
    injectable ``runner``, which is how the tests exercise the exact argv of a rollout
    without a cluster.

``ArgoRolloutsDriver`` / ``WeightedServiceDriver``
    The two ways to move traffic. The first delegates to Argo Rollouts
    (``kubectl argo rollouts set weight|promote|abort``), which is what a cluster that
    already runs Argo should use. The second needs no CRDs: two Deployments, two Services,
    and a weight annotation the ingress splits on -- promotion rolls the canary's image onto
    the stable Deployment and then drains the canary to zero, which is the same end state
    Argo reaches.

``CanaryRunner``
    The loop: start, set weight, poll metrics, tick, apply. Its sleep and its clock are
    injected, so a rollout that would take an hour in the cluster runs instantly in a test.

``canary_app``
    The ``turboserve canary`` typer sub-app: ``run`` replays a recorded stream of request
    outcomes through the controller (no cluster, no metrics backend, useful for tuning the
    gates and for CI), and ``run --kube`` performs the real thing.

Why promotion is not just "weight = 100": at 100% the canary is serving everything but the
stable Deployment still exists on the old image, and the next rollout would compare against
it. Promotion has to make the canary *the* version -- Argo does it inside the controller,
and the Service driver does it by patching the image and scaling the canary down.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, runtime_checkable

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field

from turboserve.canary.controller import (
    CanaryConfig,
    CanaryController,
    CanaryError,
    CanaryState,
    Decision,
    DecisionKind,
    Lane,
    LaneSummary,
)
from turboserve.canary.prometheus import (
    PrometheusClient,
    PrometheusError,
    PrometheusLaneSource,
    PrometheusSettings,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ArgoRolloutsDriver",
    "CanaryRunner",
    "CommandResult",
    "Kubectl",
    "KubectlError",
    "KubeSettings",
    "OutcomeEvent",
    "RolloutDriver",
    "RolloutOutcome",
    "RolloutReport",
    "WeightedServiceDriver",
    "canary_app",
    "load_outcomes",
    "simulate",
]

DEFAULT_CONFIG_PATH = Path("configs/canary.yaml")


# ---------------------------------------------------------------------------------------
# kubectl
# ---------------------------------------------------------------------------------------


class KubectlError(RuntimeError):
    """A ``kubectl`` invocation failed, timed out, or could not be started."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The outcome of one command, including the ones dry-run refused to execute."""

    args: tuple[str, ...]
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False
    """True when ``dry_run`` suppressed a cluster-changing command."""

    @property
    def ok(self) -> bool:
        """Whether the command succeeded (a skipped command counts as success)."""
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict for the rollout report's command log."""
        return {
            "args": list(self.args),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "skipped": self.skipped,
        }


class Kubectl:
    """Runs ``kubectl`` and remembers what it ran.

    ``dry_run`` is the important flag. It does not pass ``--dry-run=client`` to kubectl --
    that would still contact the API server and would not work for ``argo rollouts``
    subcommands or for ``scale``. Instead it classifies each command: reads still execute
    (the plan has to be computed from the real cluster), and anything that changes state or
    waits for a change is logged and skipped. The result is a rehearsal that shows the
    operator the exact argv a real run would execute.
    """

    def __init__(
        self,
        *,
        namespace: str | None = None,
        binary: str = "kubectl",
        dry_run: bool = False,
        timeout_s: float = 60.0,
        runner: Callable[[Sequence[str], float], CommandResult] | None = None,
    ) -> None:
        self.namespace = namespace
        self.binary = binary
        self.dry_run = dry_run
        self.timeout_s = timeout_s
        self._runner = runner or _subprocess_runner
        self._commands: list[CommandResult] = []

    @property
    def commands(self) -> tuple[CommandResult, ...]:
        """Every command attempted, in order -- the audit trail of a rollout."""
        return tuple(self._commands)

    def argv(self, args: Sequence[str], *, namespaced: bool = True) -> tuple[str, ...]:
        """Full argv for ``args``, with ``-n <namespace>`` appended when configured."""
        full = [self.binary, *args]
        if namespaced and self.namespace:
            full += ["-n", self.namespace]
        return tuple(full)

    def run(
        self,
        args: Sequence[str],
        *,
        mutating: bool,
        namespaced: bool = True,
        check: bool = True,
    ) -> CommandResult:
        """Run one kubectl command.

        ``mutating`` means "changes cluster state, or blocks waiting for a change"; those
        are the commands ``dry_run`` suppresses.
        """
        full = self.argv(args, namespaced=namespaced)
        if mutating and self.dry_run:
            logger.info("[dry-run] %s", " ".join(full))
            result = CommandResult(args=full, skipped=True)
            self._commands.append(result)
            return result
        logger.info("$ %s", " ".join(full))
        result = self._runner(full, self.timeout_s)
        self._commands.append(result)
        if check and not result.ok:
            detail = result.stderr.strip() or result.stdout.strip()
            raise KubectlError(f"{' '.join(full)} exited {result.returncode}: {detail}")
        return result

    def get_jsonpath(self, resource: str, name: str, jsonpath: str) -> str:
        """Read one field of one object; returns the raw (stripped) stdout."""
        result = self.run(
            [
                "get",
                resource,
                name,
                "-o",
                f"jsonpath={jsonpath}",
            ],
            mutating=False,
        )
        return result.stdout.strip()

    def patch(self, resource: str, name: str, patch: Mapping[str, Any]) -> CommandResult:
        """Strategic-merge patch an object with a JSON document."""
        return self.run(
            ["patch", resource, name, "--type", "merge", "-p", json.dumps(patch, sort_keys=True)],
            mutating=True,
        )


def _subprocess_runner(args: Sequence[str], timeout_s: float) -> CommandResult:
    """Default runner: ``subprocess.run`` with no shell and a hard timeout."""
    try:
        completed = subprocess.run(  # noqa: S603 - argv is built here, never from a shell string
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise KubectlError(f"{args[0]} not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise KubectlError(f"{' '.join(args)} timed out after {timeout_s:g}s") from exc
    return CommandResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


# ---------------------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------------------


@runtime_checkable
class RolloutDriver(Protocol):
    """What the runner needs from a traffic-shifting mechanism."""

    def set_weight(self, percent: int) -> None:
        """Send ``percent`` of traffic to the canary lane."""

    def promote(self) -> None:
        """Make the canary the new stable version."""

    def abort(self) -> None:
        """Send all traffic back to stable and stand the canary down."""

    def describe(self) -> dict[str, Any]:
        """Identify the objects this driver acts on, for the rollout report."""


def _validate_percent(percent: int) -> int:
    if not 0 <= percent <= 100:
        raise ValueError(f"weight must be a percentage in [0, 100], got {percent}")
    return int(percent)


class ArgoRolloutsDriver:
    """Drives an Argo Rollouts ``Rollout`` through the ``kubectl argo rollouts`` plugin.

    The plugin is used rather than patching the CR directly because the weight lives in the
    Rollout's status machinery, and the plugin is the supported way to move it without
    fighting the Argo controller for ownership of the object.
    """

    def __init__(self, kubectl: Kubectl, rollout: str, *, full_promote: bool = True) -> None:
        if not rollout:
            raise ValueError("rollout name must not be empty")
        self.kubectl = kubectl
        self.rollout = rollout
        self.full_promote = full_promote

    def set_weight(self, percent: int) -> None:
        """``kubectl argo rollouts set weight <rollout> <percent>``."""
        pct = _validate_percent(percent)
        self.kubectl.run(
            ["argo", "rollouts", "set", "weight", self.rollout, str(pct)], mutating=True
        )

    def promote(self) -> None:
        """``kubectl argo rollouts promote <rollout> [--full]``.

        ``--full`` skips the remaining analysis steps: our controller has already decided,
        and leaving Argo to re-run its own steps would gate the rollout twice.
        """
        args = ["argo", "rollouts", "promote", self.rollout]
        if self.full_promote:
            args.append("--full")
        self.kubectl.run(args, mutating=True)

    def abort(self) -> None:
        """``kubectl argo rollouts abort <rollout>``, which returns traffic to stable."""
        self.kubectl.run(["argo", "rollouts", "abort", self.rollout], mutating=True)

    def describe(self) -> dict[str, Any]:
        """The Rollout this driver acts on."""
        return {
            "mode": "argo",
            "rollout": self.rollout,
            "namespace": self.kubectl.namespace,
        }


class WeightedServiceDriver:
    """Two Deployments and two Services, split by a weight annotation.

    The annotation defaults to ``nginx.ingress.kubernetes.io/canary-weight`` because that is
    the split an NGINX ingress understands out of the box; a second, vendor-neutral
    annotation carries the same number so a gateway (or a service mesh with its own
    controller) can read the lane weights without depending on the ingress vendor.
    """

    def __init__(
        self,
        kubectl: Kubectl,
        *,
        stable_deployment: str,
        canary_deployment: str,
        stable_service: str,
        canary_service: str,
        container: str = "gateway",
        weight_annotation: str = "nginx.ingress.kubernetes.io/canary-weight",
        lane_annotation: str = "turboserve.io/lane-weight",
        rollout_timeout_s: float = 600.0,
    ) -> None:
        self.kubectl = kubectl
        self.stable_deployment = stable_deployment
        self.canary_deployment = canary_deployment
        self.stable_service = stable_service
        self.canary_service = canary_service
        self.container = container
        self.weight_annotation = weight_annotation
        self.lane_annotation = lane_annotation
        self.rollout_timeout_s = rollout_timeout_s

    def set_weight(self, percent: int) -> None:
        """Patch both Services' annotations so the two weights always sum to 100."""
        pct = _validate_percent(percent)
        self.kubectl.patch(
            "service",
            self.canary_service,
            {
                "metadata": {
                    "annotations": {
                        self.weight_annotation: str(pct),
                        self.lane_annotation: str(pct),
                    }
                }
            },
        )
        self.kubectl.patch(
            "service",
            self.stable_service,
            {"metadata": {"annotations": {self.lane_annotation: str(100 - pct)}}},
        )

    def canary_image(self) -> str:
        """The image the canary Deployment is running, read from the live object."""
        jsonpath = (
            "{.spec.template.spec.containers[?(@.name==" + f'"{self.container}"' + ")].image}"
        )
        image = self.kubectl.get_jsonpath("deployment", self.canary_deployment, jsonpath)
        if not image:
            raise KubectlError(
                f"deployment/{self.canary_deployment} has no container named "
                f"{self.container!r}; set kubernetes.container in the canary config"
            )
        return image

    def promote(self) -> None:
        """Roll the canary image onto stable, wait for it, then drain the canary."""
        image = self.canary_image()
        self.kubectl.run(
            ["set", "image", f"deployment/{self.stable_deployment}", f"{self.container}={image}"],
            mutating=True,
        )
        self.kubectl.run(
            [
                "rollout",
                "status",
                f"deployment/{self.stable_deployment}",
                f"--timeout={self.rollout_timeout_s:g}s",
            ],
            mutating=True,
        )
        self.set_weight(0)
        self._scale_canary(0)

    def abort(self) -> None:
        """Take the canary out of the split and scale it to zero."""
        self.set_weight(0)
        self._scale_canary(0)

    def _scale_canary(self, replicas: int) -> None:
        self.kubectl.run(
            ["scale", f"deployment/{self.canary_deployment}", f"--replicas={replicas}"],
            mutating=True,
        )

    def describe(self) -> dict[str, Any]:
        """The Deployments and Services this driver acts on."""
        return {
            "mode": "services",
            "namespace": self.kubectl.namespace,
            "stable_deployment": self.stable_deployment,
            "canary_deployment": self.canary_deployment,
            "stable_service": self.stable_service,
            "canary_service": self.canary_service,
            "container": self.container,
        }


# ---------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------


class KubeSettings(BaseModel):
    """The ``kubernetes:`` section of ``configs/canary.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["argo", "services"] = "argo"
    namespace: str = "turboserve"
    kubectl_binary: str = "kubectl"
    rollout: str = "turboserve-gateway"
    stable_deployment: str = "turboserve-gateway-stable"
    canary_deployment: str = "turboserve-gateway-canary"
    stable_service: str = "turboserve-gateway-stable"
    canary_service: str = "turboserve-gateway-canary"
    container: str = "gateway"
    weight_annotation: str = "nginx.ingress.kubernetes.io/canary-weight"
    lane_annotation: str = "turboserve.io/lane-weight"
    poll_interval_s: float = Field(default=15.0, gt=0.0)
    command_timeout_s: float = Field(default=60.0, gt=0.0)
    rollout_timeout_s: float = Field(default=600.0, gt=0.0)
    deadline_s: float | None = Field(default=None, gt=0.0)
    """Hard limit on a whole rollout; exceeded means abort. ``None`` disables it."""

    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path, *, key: str | None = "kubernetes") -> KubeSettings:
        """Load the ``kubernetes:`` section, defaulting every absent field."""
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(document, dict):
            raise ValueError(f"{path}: expected a YAML mapping, got {type(document).__name__}")
        section: Any = document
        if key is not None:
            section = document.get(key, {})
        if not isinstance(section, dict):
            raise ValueError(f"{path}: section {key!r} must be a mapping")
        return cls.model_validate(section)

    def build_kubectl(
        self,
        *,
        dry_run: bool | None = None,
        runner: Callable[[Sequence[str], float], CommandResult] | None = None,
    ) -> Kubectl:
        """A :class:`Kubectl` configured from these settings."""
        return Kubectl(
            namespace=self.namespace,
            binary=self.kubectl_binary,
            dry_run=self.dry_run if dry_run is None else dry_run,
            timeout_s=self.command_timeout_s,
            runner=runner,
        )

    def build_driver(self, kubectl: Kubectl) -> RolloutDriver:
        """The driver named by :attr:`mode`."""
        if self.mode == "argo":
            return ArgoRolloutsDriver(kubectl, self.rollout)
        return WeightedServiceDriver(
            kubectl,
            stable_deployment=self.stable_deployment,
            canary_deployment=self.canary_deployment,
            stable_service=self.stable_service,
            canary_service=self.canary_service,
            container=self.container,
            weight_annotation=self.weight_annotation,
            lane_annotation=self.lane_annotation,
            rollout_timeout_s=self.rollout_timeout_s,
        )


# ---------------------------------------------------------------------------------------
# Running a rollout
# ---------------------------------------------------------------------------------------


class RolloutOutcome(StrEnum):
    """How a rollout ended."""

    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    TIMED_OUT = "timed_out"
    RUNNING = "running"


@dataclass(slots=True)
class RolloutReport:
    """Everything a rollout did: its decisions, its commands, and how it ended.

    Written as JSON by ``turboserve canary run --out``, which is what the kind e2e job and
    a post-incident review read. It carries no latency figures of its own -- the decisions
    embed the :class:`LaneSummary` each gate saw, so the report explains itself.
    """

    version: str
    outcome: RolloutOutcome = RolloutOutcome.RUNNING
    final_weight: int = 0
    decisions: list[Decision] = field(default_factory=list)
    commands: list[CommandResult] = field(default_factory=list)
    driver: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def promoted(self) -> bool:
        """Whether the canary became the new stable version."""
        return self.outcome is RolloutOutcome.PROMOTED

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict."""
        return {
            "version": self.version,
            "outcome": str(self.outcome),
            "final_weight": self.final_weight,
            "dry_run": self.dry_run,
            "driver": self.driver,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "commands": [command.to_dict() for command in self.commands],
        }


@runtime_checkable
class LaneSource(Protocol):
    """Anything that can report both lanes' health for one evaluation."""

    def fetch_all(self, *, at: float | None = None) -> dict[Lane, LaneSummary]:
        """Both lane summaries, read at the same instant."""


class CanaryRunner:
    """The control loop that turns decisions into cluster changes.

    Sleep and clock are injected so the whole loop is testable without waiting: the tests
    run rollouts that would take an hour of wall-clock in well under a second.

    Metric-backend failures are tolerated up to ``max_metric_failures`` consecutive polls
    and then roll the canary back. Holding for ever because Prometheus is down leaves a
    half-shifted rollout unattended, which is worse than returning to a known-good state.
    """

    def __init__(
        self,
        controller: CanaryController,
        driver: RolloutDriver,
        source: LaneSource | None = None,
        *,
        poll_interval_s: float = 15.0,
        deadline_s: float | None = None,
        max_metric_failures: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s}")
        self.controller = controller
        self.driver = driver
        self.source = source
        self.poll_interval_s = poll_interval_s
        self.deadline_s = deadline_s
        self.max_metric_failures = max(1, max_metric_failures)
        self._sleep = sleep
        self._clock: Callable[[], float] = clock or time.monotonic

    def run(self, version: str) -> RolloutReport:
        """Start a rollout of ``version`` and drive it to a terminal state."""
        from turboserve.bench.records import utc_now_iso

        report = RolloutReport(
            version=version,
            driver=self.driver.describe(),
            started_at=utc_now_iso(),
        )
        started = self._clock()
        report.decisions.append(self.controller.start(version, now=started))
        self.driver.set_weight(self.controller.canary_weight)
        failures = 0

        while self.controller.is_running:
            self._sleep(self.poll_interval_s)
            now = self._clock()
            if self.deadline_s is not None and now - started >= self.deadline_s:
                report.decisions.append(
                    self.controller.abort(
                        f"rollout exceeded its deadline of {self.deadline_s:g}s", now=now
                    )
                )
                self.driver.abort()
                report.outcome = RolloutOutcome.TIMED_OUT
                break

            summaries: dict[Lane, LaneSummary] = {}
            if self.source is not None:
                try:
                    summaries = self.source.fetch_all(at=None)
                except PrometheusError as exc:
                    failures += 1
                    logger.warning(
                        "canary metrics unavailable (%d/%d): %s",
                        failures,
                        self.max_metric_failures,
                        exc,
                    )
                    if failures >= self.max_metric_failures:
                        report.decisions.append(
                            self.controller.abort(
                                f"metrics unavailable for {failures} consecutive polls: {exc}",
                                now=now,
                            )
                        )
                        self.driver.abort()
                        report.outcome = RolloutOutcome.ROLLED_BACK
                        break
                    continue
                failures = 0

            decision = self.controller.tick(
                now, stable=summaries.get("stable"), canary=summaries.get("canary")
            )
            report.decisions.append(decision)
            self._apply(decision, report)

        report.final_weight = self.controller.canary_weight
        kubectl = getattr(self.driver, "kubectl", None)
        if isinstance(kubectl, Kubectl):
            report.commands = list(kubectl.commands)
            report.dry_run = kubectl.dry_run
        report.finished_at = utc_now_iso()
        return report

    def _apply(self, decision: Decision, report: RolloutReport) -> None:
        """Turn one decision into the matching cluster action."""
        match decision.kind:
            case DecisionKind.ADVANCE:
                self.driver.set_weight(self.controller.canary_weight)
            case DecisionKind.PROMOTE:
                self.driver.promote()
                report.outcome = RolloutOutcome.PROMOTED
            case DecisionKind.ROLLBACK:
                self.driver.abort()
                report.outcome = RolloutOutcome.ROLLED_BACK
            case _:
                logger.debug("canary hold: %s", decision.reason)


# ---------------------------------------------------------------------------------------
# In-process simulation
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    """One recorded request outcome, replayed into the controller by :func:`simulate`."""

    t: float
    lane: Lane
    ok: bool = True
    ttft_ms: float | None = None
    e2e_ms: float | None = None


def _as_bool(value: Any) -> bool:
    """Parse the ``ok`` column of a CSV or a JSON value that may be a string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "t", "yes", "y", "1", "ok"):
        return True
    if text in ("false", "f", "no", "n", "0", "error", "err"):
        return False
    raise ValueError(f"cannot read {value!r} as a boolean")


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _event_from_mapping(row: Mapping[str, Any]) -> OutcomeEvent:
    lane = str(row.get("lane", "canary")).strip()
    if lane not in ("stable", "canary"):
        raise ValueError(f"lane must be 'stable' or 'canary', got {lane!r}")
    return OutcomeEvent(
        t=float(row["t"]),
        lane=lane,  # type: ignore[arg-type]
        ok=_as_bool(row.get("ok", True)),
        ttft_ms=_as_optional_float(row.get("ttft_ms")),
        e2e_ms=_as_optional_float(row.get("e2e_ms")),
    )


def load_outcomes(path: str | Path) -> list[OutcomeEvent]:
    """Read a recorded outcome stream from JSON, JSON-lines or CSV.

    Accepted shapes, all with the columns ``t, lane, ok, ttft_ms, e2e_ms``:

    * ``.json``  -- a list of objects, or ``{"events": [...]}``
    * ``.jsonl`` -- one object per line
    * ``.csv``   -- a header row and one record per line

    The stream is what a real gateway's request log looks like once it has been projected
    onto the five fields the gate cares about, which is what makes replaying it a
    meaningful rehearsal of a rollout rather than a toy.
    """
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    rows: list[Mapping[str, Any]]
    suffix = file.suffix.lower()
    if suffix == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
    elif suffix in (".jsonl", ".ndjson"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        document = json.loads(text)
        if isinstance(document, dict):
            document = document.get("events", [])
        if not isinstance(document, list):
            raise ValueError(f"{file}: expected a list of events or an 'events' key")
        rows = document
    events = [_event_from_mapping(row) for row in rows]
    events.sort(key=lambda event: event.t)
    return events


def simulate(
    controller: CanaryController,
    events: Iterable[OutcomeEvent],
    *,
    version: str = "candidate",
    tick_interval_s: float = 10.0,
) -> RolloutReport:
    """Replay ``events`` through ``controller`` on a virtual clock and report what happened.

    The controller is ticked every ``tick_interval_s`` of *event* time, so the simulation is
    fully deterministic: the same file and the same config always produce the same
    decisions, which is what makes this usable as a regression test for a gate change.
    """
    from turboserve.bench.records import utc_now_iso

    if tick_interval_s <= 0:
        raise ValueError(f"tick_interval_s must be > 0, got {tick_interval_s}")
    ordered = sorted(events, key=lambda event: event.t)
    report = RolloutReport(version=version, started_at=utc_now_iso())
    start_t = ordered[0].t if ordered else 0.0
    report.decisions.append(controller.start(version, now=start_t))
    next_tick = start_t + tick_interval_s
    last_t = start_t

    for event in ordered:
        while controller.is_running and event.t >= next_tick:
            report.decisions.append(controller.tick(next_tick))
            next_tick += tick_interval_s
        if not controller.is_running:
            break
        controller.observe(event.lane, event.ok, event.ttft_ms, event.e2e_ms, now=event.t)
        last_t = event.t

    while controller.is_running and next_tick <= last_t + tick_interval_s:
        report.decisions.append(controller.tick(next_tick))
        next_tick += tick_interval_s

    report.final_weight = controller.canary_weight
    report.outcome = _outcome_for(controller.state)
    report.finished_at = utc_now_iso()
    return report


def _outcome_for(state: CanaryState) -> RolloutOutcome:
    match state:
        case CanaryState.PROMOTED:
            return RolloutOutcome.PROMOTED
        case CanaryState.ROLLED_BACK:
            return RolloutOutcome.ROLLED_BACK
        case _:
            return RolloutOutcome.RUNNING


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

canary_app = typer.Typer(
    name="canary",
    help="SLO-gated progressive delivery: replay a recorded rollout or drive a real one.",
    no_args_is_help=True,
    add_completion=False,
)


def _load_config(path: Path) -> CanaryConfig:
    if path.exists():
        return CanaryConfig.from_yaml(path)
    logger.info("canary config %s not found; using built-in defaults", path)
    return CanaryConfig()


def _decisions_table(report: RolloutReport) -> Any:
    from rich.table import Table

    table = Table(title=f"canary rollout: {report.version}", show_lines=False)
    table.add_column("t", justify="right")
    table.add_column("decision")
    table.add_column("weight", justify="right")
    table.add_column("reason", overflow="fold")
    for decision in report.decisions:
        table.add_row(
            f"{decision.at:.1f}",
            str(decision.kind),
            f"{decision.weight}%",
            decision.reason,
        )
    return table


def _emit(report: RolloutReport, out: Path | None) -> None:
    from rich.console import Console

    console = Console()
    console.print(_decisions_table(report))
    console.print(f"outcome: [bold]{report.outcome}[/bold] at {report.final_weight}%")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"wrote {out}")


@canary_app.command()
def plan(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="YAML file holding the canary policy."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Print the rollout policy: the weight steps and every gate that can fail them."""
    from rich.console import Console

    settings = _load_config(config)
    console = Console()
    console.print_json(json.dumps(settings.model_dump(mode="json"), sort_keys=True))


@canary_app.command()
def run(
    version: Annotated[
        str, typer.Option("--version", help="Version label recorded with the rollout.")
    ] = "candidate",
    config: Annotated[
        Path, typer.Option("--config", "-c", help="YAML file holding the canary policy.")
    ] = DEFAULT_CONFIG_PATH,
    outcomes: Annotated[
        Path | None,
        typer.Option(
            "--outcomes",
            help="Recorded request outcomes (.json/.jsonl/.csv); required without --kube.",
        ),
    ] = None,
    tick_interval_s: Annotated[
        float,
        typer.Option("--tick-interval", min=0.001, help="Seconds of event time between gates."),
    ] = 10.0,
    kube: Annotated[
        bool, typer.Option("--kube", help="Drive a real rollout with kubectl and Prometheus.")
    ] = False,
    namespace: Annotated[
        str | None, typer.Option("--namespace", "-n", help="Override the configured namespace.")
    ] = None,
    prometheus_url: Annotated[
        str | None, typer.Option("--prometheus-url", help="Override the Prometheus base URL.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Log cluster-changing commands instead of running them."),
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the rollout report as JSON here.")
    ] = None,
) -> None:
    """Roll a canary out: replay recorded outcomes, or drive a cluster with ``--kube``.

    Exits non-zero when the rollout did not promote, so a pipeline step fails when the gate
    rejected the candidate.
    """
    policy = _load_config(config)
    controller = CanaryController(policy)
    if kube:
        report = _run_kube(
            controller,
            config_path=config,
            version=version,
            namespace=namespace,
            prometheus_url=prometheus_url,
            dry_run=dry_run,
        )
    else:
        if outcomes is None:
            raise typer.BadParameter("--outcomes is required unless --kube is given")
        try:
            events = load_outcomes(outcomes)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise typer.BadParameter(f"cannot read {outcomes}: {exc}") from exc
        report = simulate(controller, events, version=version, tick_interval_s=tick_interval_s)
    _emit(report, out)
    if not report.promoted:
        raise typer.Exit(code=1)


@canary_app.command()
def abort(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="YAML file holding the canary policy.")
    ] = DEFAULT_CONFIG_PATH,
    namespace: Annotated[
        str | None, typer.Option("--namespace", "-n", help="Override the configured namespace.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Log cluster-changing commands instead of running them."),
    ] = False,
) -> None:
    """Take the traffic off the canary lane now, without waiting for the gate.

    The break-glass command: it drives the configured rollout driver's ``abort`` directly
    (``kubectl argo rollouts abort``, or weight 0 plus scaling the canary Deployment down)
    and does not consult Prometheus, because the operator reaching for this has already
    decided. ``turboserve canary run`` does the same thing by itself when a gate fails; this
    exists for the case where nobody is running it, or where it is running somewhere you
    cannot reach.
    """
    from rich.console import Console

    kube_settings = KubeSettings.from_yaml(config) if config.exists() else KubeSettings()
    if namespace:
        kube_settings = kube_settings.model_copy(update={"namespace": namespace})
    kubectl = kube_settings.build_kubectl(dry_run=dry_run or kube_settings.dry_run)
    driver = kube_settings.build_driver(kubectl)
    try:
        driver.abort()
    except KubectlError as exc:
        logger.error("canary abort failed: %s", exc)
        raise typer.Exit(code=2) from exc
    console = Console()
    for command in kubectl.commands:
        console.print(
            f"[dim]{'skipped' if command.skipped else 'ran'}[/dim] {' '.join(command.args)}"
        )
    console.print("[bold]canary aborted[/bold]: the canary lane is at 0% and scaled down")


def _run_kube(
    controller: CanaryController,
    *,
    config_path: Path,
    version: str,
    namespace: str | None,
    prometheus_url: str | None,
    dry_run: bool,
) -> RolloutReport:
    """Build the kubectl driver and the Prometheus source, then run the loop."""
    kube_settings = KubeSettings.from_yaml(config_path) if config_path.exists() else KubeSettings()
    if namespace:
        kube_settings = kube_settings.model_copy(update={"namespace": namespace})
    prom_settings = (
        PrometheusSettings.from_yaml(config_path) if config_path.exists() else PrometheusSettings()
    )
    if prometheus_url:
        prom_settings = prom_settings.model_copy(update={"base_url": prometheus_url})

    kubectl = kube_settings.build_kubectl(dry_run=dry_run or kube_settings.dry_run)
    driver = kube_settings.build_driver(kubectl)
    with PrometheusClient(prom_settings.base_url, timeout_s=prom_settings.timeout_s) as client:
        source = PrometheusLaneSource(
            client, prom_settings, default_lookback_s=controller.config.window_s
        )
        runner = CanaryRunner(
            controller,
            driver,
            source,
            poll_interval_s=kube_settings.poll_interval_s,
            deadline_s=kube_settings.deadline_s,
        )
        try:
            return runner.run(version)
        except (KubectlError, CanaryError) as exc:
            logger.error("canary rollout failed: %s", exc)
            raise typer.Exit(code=2) from exc
