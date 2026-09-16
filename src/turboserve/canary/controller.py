"""The SLO gate: a pure, clock-injectable state machine for progressive delivery.

A canary rollout is a control loop. Traffic is split between a ``stable`` lane and a
``canary`` lane, the canary's share is raised one step at a time, and after every step the
loop asks a single question: *is the canary still allowed to keep this traffic?* This
module answers that question and nothing else. It does not talk to Kubernetes, it does not
talk to Prometheus, it does not sleep, and it does not read the wall clock unless it is
asked to. :mod:`turboserve.canary.k8s` supplies the effects; this file supplies the
decision, which is why it can be tested exhaustively on a fake clock in milliseconds.

Why the observations live here
------------------------------
The controller keeps its own sliding window of per-request outcomes (a ring buffer per
lane) so that the in-process path -- the gateway router calling :meth:`CanaryController.observe`
as requests finish -- needs no metrics backend at all. The Kubernetes path has the same
data, but pre-aggregated by Prometheus, so :meth:`CanaryController.tick` also accepts
:class:`LaneSummary` overrides. Both paths then run the *identical* gate code, which is the
point: a rollout must not be judged by one rule in a test and another in production.

The gates
---------
A step is only judged once the canary lane has at least ``min_requests`` samples in the
window; below that the controller holds, because a handful of requests cannot distinguish a
bad build from noise. Once there is enough evidence three things can fail a canary:

1. **Error rate.** The canary's failure fraction exceeds ``max_error_rate``.
2. **Absolute latency.** The canary's p95 TTFT exceeds ``max_p95_ttft_ms`` (opt-in; the
   default is ``None``, meaning "not asserted", because an absolute millisecond budget is a
   property of a deployment's model and hardware, not of this code).
3. **Relative latency.** The canary's p95 is more than ``max_p95_ratio_vs_stable`` times
   the stable lane's p95, measured over the same window and therefore under the same load.
   This is the gate that survives a traffic spike: both lanes get slower together, the
   ratio does not move, and a healthy canary is not rolled back for someone else's noise.

Percentiles come from :func:`turboserve.bench.records.percentile` so that the figure the
controller gates on is computed exactly like the figure the benchmark reports.

References: the step/analysis/promote structure follows Argo Rollouts' canary strategy and
Flagger's progressive-delivery loop; the "compare the canary against the baseline under the
same load rather than against a fixed threshold" rule is the central lesson of Netflix's
Kayenta/automated canary analysis.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from turboserve.bench.records import percentile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

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
    "Sample",
]

#: The two traffic lanes. Every request, metric and window in the repository is labelled
#: with one of these (see ``RequestRecord.lane`` and the gateway's ``lane`` metric label).
#:
#: :data:`turboserve.gateway.router.Lane` declares the identical alias, on purpose: the
#: gateway must not import this package (the dependency runs canary -> gateway, never the
#: reverse), and a shared two-element ``Literal`` is not worth a third module to hold it.
Lane = Literal["stable", "canary"]

LANES: tuple[Lane, ...] = ("stable", "canary")


class CanaryError(RuntimeError):
    """Raised when a rollout operation is impossible in the controller's current state."""


class CanaryState(StrEnum):
    """Where a rollout is. ``IDLE`` and the two terminal states carry no traffic split."""

    IDLE = "idle"
    CANARY = "canary"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


class DecisionKind(StrEnum):
    """What :meth:`CanaryController.tick` concluded.

    ``ADVANCE`` raises the canary's weight to the next step, ``PROMOTE`` ends the rollout
    with the canary serving everything, ``ROLLBACK`` ends it with the canary serving
    nothing, and ``HOLD`` means "keep the current weight and ask again later".
    """

    HOLD = "hold"
    ADVANCE = "advance"
    PROMOTE = "promote"
    ROLLBACK = "rollback"


# ---------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Sample:
    """One finished request as the controller sees it.

    Latencies are optional because a request that failed before its first token has no
    TTFT, and excluding it from the latency percentiles while still counting it in the
    error rate is the honest treatment: otherwise a fast failure would *improve* p95.
    """

    t: float
    ok: bool
    ttft_ms: float | None = None
    e2e_ms: float | None = None


@dataclass(frozen=True, slots=True)
class LaneSummary:
    """What one lane did inside one window: the only input the gates consume.

    Produced either by :meth:`LaneWindow.summary` (in-process observations) or by
    :class:`turboserve.canary.prometheus.PrometheusLaneSource` (a PromQL query over the
    gateway's metrics). Having one type for both is what lets the Kubernetes path reuse the
    gate logic verbatim.
    """

    lane: Lane
    requests: int = 0
    errors: int = 0
    p95_ttft_ms: float | None = None
    p95_e2e_ms: float | None = None
    window_s: float | None = None

    def __post_init__(self) -> None:
        if self.requests < 0:
            raise ValueError(f"requests must be >= 0, got {self.requests}")
        if not 0 <= self.errors <= self.requests:
            raise ValueError(f"errors must be in [0, {self.requests}], got {self.errors}")

    @property
    def error_rate(self) -> float:
        """Failed fraction of the window; ``0.0`` when nothing was observed.

        An empty window is *not* an error rate of one: "no traffic" is handled by the
        ``min_requests`` hold, not by the error gate.
        """
        return self.errors / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, JSON-ready, with ``error_rate`` materialised for readers."""
        data = asdict(self)
        data["error_rate"] = self.error_rate
        return data


class LaneWindow:
    """A bounded sliding window of one lane's samples.

    Implemented as a ``deque`` used as a ring buffer: appends and expiry are both O(1)
    amortised, and ``max_samples`` caps memory under a traffic burst, so a controller
    embedded in a gateway cannot grow without bound if a window is configured generously.
    Samples are expired lazily -- on observation and on summary -- because a window with no
    traffic costs nothing to keep.
    """

    def __init__(self, lane: Lane, window_s: float, *, max_samples: int = 100_000) -> None:
        if lane not in LANES:
            raise ValueError(f"lane must be one of {LANES}, got {lane!r}")
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {max_samples}")
        self.lane: Lane = lane
        self.window_s = float(window_s)
        self._samples: deque[Sample] = deque(maxlen=max_samples)
        self._now = float("-inf")

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    @property
    def max_samples(self) -> int:
        """Ring-buffer capacity; the oldest sample is dropped when it is exceeded."""
        return self._samples.maxlen or 0

    def observe(
        self,
        ok: bool,
        *,
        ttft_ms: float | None = None,
        e2e_ms: float | None = None,
        now: float,
    ) -> None:
        """Record one finished request at time ``now`` (seconds, any monotone origin)."""
        self.prune(now)
        if now < self._now - self.window_s:
            # Arrived so late it is already outside the window; nothing would ever read it.
            return
        self._samples.append(Sample(t=float(now), ok=bool(ok), ttft_ms=ttft_ms, e2e_ms=e2e_ms))

    def prune(self, now: float) -> None:
        """Drop samples older than ``window_s`` relative to the latest time seen.

        The latest time seen -- not ``now`` itself -- is the reference, so an out-of-order
        timestamp from a concurrent caller can never un-expire samples that were already
        dropped, and the window can only move forwards.
        """
        self._now = max(self._now, float(now))
        cutoff = self._now - self.window_s
        samples = self._samples
        while samples and samples[0].t < cutoff:
            samples.popleft()

    def clear(self) -> None:
        """Forget every sample, used when a new rollout starts."""
        self._samples.clear()
        self._now = float("-inf")

    def summary(self, now: float) -> LaneSummary:
        """Expire, then aggregate: counts, error count and p95 TTFT/E2E."""
        self.prune(now)
        requests = len(self._samples)
        errors = sum(1 for sample in self._samples if not sample.ok)
        ttft = [s.ttft_ms for s in self._samples if s.ok and s.ttft_ms is not None]
        e2e = [s.e2e_ms for s in self._samples if s.ok and s.e2e_ms is not None]
        return LaneSummary(
            lane=self.lane,
            requests=requests,
            errors=errors,
            p95_ttft_ms=percentile(ttft, 95.0),
            p95_e2e_ms=percentile(e2e, 95.0),
            window_s=self.window_s,
        )


# ---------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------


class CanaryConfig(BaseModel):
    """The rollout's policy: how far, how fast, and what fails it.

    Frozen and ``extra="forbid"`` so that a typo in ``configs/canary.yaml`` is a loud error
    at load time rather than a gate that silently never fires.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: tuple[int, ...] = (1, 5, 25, 50, 100)
    """Canary traffic percentages, strictly increasing and ending at 100."""

    step_hold_s: float = Field(default=300.0, gt=0.0)
    """Seconds a step must survive before the controller will advance past it."""

    window_s: float = Field(default=300.0, gt=0.0)
    """Width of the sliding window the gates are evaluated over."""

    min_requests: int = Field(default=200, ge=1)
    """Canary samples required in the window before any gate is evaluated."""

    max_error_rate: float = Field(default=0.005, ge=0.0, le=1.0)
    """Failed fraction of canary requests that fails the rollout."""

    max_p95_ttft_ms: float | None = Field(default=None, gt=0.0)
    """Absolute p95 TTFT budget; ``None`` (the default) means the gate is not asserted."""

    max_p95_ratio_vs_stable: float = Field(default=1.25, gt=0.0)
    """How much slower at p95 the canary may be than the stable lane in the same window."""

    stall_timeout_s: float | None = Field(default=None, gt=0.0)
    """Roll back if a step never reaches ``min_requests`` within this many seconds.

    ``None`` disables it. It exists because a canary that takes *no* traffic -- a crash
    loop, a bad readiness probe, a router that never picked it -- would otherwise hold for
    ever, looking healthy while serving nobody.
    """

    max_samples_per_lane: int = Field(default=100_000, ge=1)
    """Ring-buffer capacity per lane, bounding memory under a burst."""

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("steps")
    @classmethod
    def _steps_are_monotone_and_complete(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("steps must not be empty")
        for step in value:
            if not 1 <= step <= 100:
                raise ValueError(f"every step must be in [1, 100], got {step}")
        if any(later <= earlier for earlier, later in zip(value, value[1:], strict=False)):
            raise ValueError(f"steps must be strictly increasing, got {value}")
        if value[-1] != 100:
            raise ValueError(f"the last step must be 100, got {value[-1]}")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path, *, key: str | None = "canary") -> CanaryConfig:
        """Load the ``canary:`` section of a YAML file (or the whole document).

        ``configs/canary.yaml`` also carries ``prometheus:`` and ``kubernetes:`` sections,
        which are parsed separately by :mod:`turboserve.canary.k8s`; keeping one file with
        three independently-validated sections means the operator edits one thing.
        """
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise ValueError(f"{path}: expected a YAML mapping, got {type(document).__name__}")
        section: Any = document
        if key is not None and key in document:
            section = document[key]
        if not isinstance(section, dict):
            raise ValueError(f"{path}: section {key!r} must be a mapping")
        return cls.model_validate(section)


# ---------------------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision:
    """One evaluation of the gates: the verdict, why, and the resulting rollout position.

    ``state`` and ``weight`` are the values *after* the decision was applied, so a stored
    history replays the rollout exactly -- which is what the CLI prints and what the
    Kubernetes driver acts on.
    """

    kind: DecisionKind
    reason: str
    at: float
    state: CanaryState
    weight: int
    step_index: int
    version: str | None = None
    stable: LaneSummary | None = None
    canary: LaneSummary | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether this decision ended the rollout."""
        return self.kind in (DecisionKind.PROMOTE, DecisionKind.ROLLBACK)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; lane summaries are inlined as objects or ``None``."""
        return {
            "kind": str(self.kind),
            "reason": self.reason,
            "at": self.at,
            "state": str(self.state),
            "weight": self.weight,
            "step_index": self.step_index,
            "version": self.version,
            "stable": self.stable.to_dict() if self.stable else None,
            "canary": self.canary.to_dict() if self.canary else None,
        }


# ---------------------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------------------


class CanaryController:
    """The rollout state machine: ``IDLE → CANARY(step) → PROMOTED | ROLLED_BACK``.

    Pure in the sense that matters for testing: every time it needs the clock it either
    takes ``now`` from the caller or calls the injected ``clock``. Nothing in it sleeps,
    retries or performs I/O, so the tests drive whole rollouts -- including the ones that
    take hours of simulated time -- in microseconds.

    Thread-safety is not provided. The gateway owns one controller and mutates it from its
    event loop; a threaded caller must serialise access itself.
    """

    def __init__(
        self,
        config: CanaryConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or CanaryConfig()
        self._clock: Callable[[], float] = clock or monotonic
        self._state = CanaryState.IDLE
        self._version: str | None = None
        self._step_index = 0
        self._step_started_at = 0.0
        self._history: list[Decision] = []
        self._windows: dict[Lane, LaneWindow] = {
            lane: LaneWindow(
                lane,
                self.config.window_s,
                max_samples=self.config.max_samples_per_lane,
            )
            for lane in LANES
        }

    # -- introspection -----------------------------------------------------------------

    @property
    def state(self) -> CanaryState:
        """Current state of the rollout."""
        return self._state

    @property
    def version(self) -> str | None:
        """The version string passed to :meth:`start`, or ``None`` before the first one."""
        return self._version

    @property
    def step_index(self) -> int:
        """Index into :attr:`CanaryConfig.steps` of the step being evaluated."""
        return self._step_index

    @property
    def canary_weight(self) -> int:
        """Percentage of traffic the canary lane should receive right now, ``0..100``.

        This is the value the gateway router and the Kubernetes driver read. It is a
        percentage rather than a fraction because that is the unit Argo Rollouts, the
        ingress canary annotations and the operator's mental model all use; the fraction is
        available as :attr:`canary_fraction` for code that weights a random draw.
        """
        match self._state:
            case CanaryState.CANARY:
                return self.config.steps[self._step_index]
            case CanaryState.PROMOTED:
                return 100
            case _:
                return 0

    @property
    def canary_fraction(self) -> float:
        """:attr:`canary_weight` as a fraction in ``[0.0, 1.0]``."""
        return self.canary_weight / 100.0

    @property
    def stable_weight(self) -> int:
        """Percentage of traffic the stable lane should receive, ``100 - canary_weight``."""
        return 100 - self.canary_weight

    def lane_weights(self) -> dict[Lane, int]:
        """Both lane weights, summing to 100 -- convenient for a weighted router."""
        return {"stable": self.stable_weight, "canary": self.canary_weight}

    @property
    def is_running(self) -> bool:
        """Whether a rollout is in progress."""
        return self._state is CanaryState.CANARY

    @property
    def is_terminal(self) -> bool:
        """Whether the last rollout finished (promoted or rolled back)."""
        return self._state in (CanaryState.PROMOTED, CanaryState.ROLLED_BACK)

    @property
    def history(self) -> tuple[Decision, ...]:
        """Every decision that moved or held this rollout, oldest first."""
        return tuple(self._history)

    @property
    def last_decision(self) -> Decision | None:
        """The most recent decision, or ``None`` before the first one."""
        return self._history[-1] if self._history else None

    def window(self, lane: Lane) -> LaneWindow:
        """The sliding window of one lane, for inspection and tests."""
        if lane not in self._windows:
            raise ValueError(f"lane must be one of {LANES}, got {lane!r}")
        return self._windows[lane]

    def summary(self, lane: Lane, now: float | None = None) -> LaneSummary:
        """Aggregate one lane's window as the gates would see it."""
        return self.window(lane).summary(self._now(now))

    def snapshot(self) -> dict[str, Any]:
        """A JSON-ready view of the rollout, used by the CLI and by structured logs."""
        return {
            "state": str(self._state),
            "version": self._version,
            "weight": self.canary_weight,
            "step_index": self._step_index,
            "steps": list(self.config.steps),
            "decisions": len(self._history),
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
        }

    # -- lifecycle ---------------------------------------------------------------------

    def start(self, version: str, *, now: float | None = None) -> Decision:
        """Begin a rollout of ``version`` at the first step.

        Both lane windows are cleared: samples from a previous build must not be able to
        promote or sink the new one. Raises :class:`CanaryError` if a rollout is already
        running -- restarting one silently would lose the audit trail of why the first
        rollout was abandoned.
        """
        if self._state is CanaryState.CANARY:
            raise CanaryError(
                f"a rollout of {self._version!r} is already running at {self.canary_weight}%"
            )
        if not version:
            raise ValueError("version must be a non-empty string")
        moment = self._now(now)
        self._version = version
        self._step_index = 0
        self._step_started_at = moment
        self._state = CanaryState.CANARY
        self._history = []
        for lane_window in self._windows.values():
            lane_window.clear()
        logger.info(
            "canary start version=%s weight=%d%% steps=%s",
            version,
            self.canary_weight,
            list(self.config.steps),
        )
        return self._record(
            DecisionKind.ADVANCE,
            f"started rollout of {version} at {self.canary_weight}%",
            moment,
        )

    def abort(self, reason: str = "aborted by operator", *, now: float | None = None) -> Decision:
        """End a running rollout immediately with the canary at zero traffic."""
        if self._state is not CanaryState.CANARY:
            raise CanaryError(f"no rollout to abort (state is {self._state})")
        moment = self._now(now)
        self._state = CanaryState.ROLLED_BACK
        logger.warning("canary abort version=%s reason=%s", self._version, reason)
        return self._record(DecisionKind.ROLLBACK, reason, moment)

    def reset(self) -> None:
        """Return to ``IDLE``, clearing history and both windows.

        Used between rollouts (and by the simulator between runs) so a fresh rollout starts
        from a known state without constructing a new controller and re-wiring the router.
        """
        self._state = CanaryState.IDLE
        self._version = None
        self._step_index = 0
        self._step_started_at = 0.0
        self._history = []
        for lane_window in self._windows.values():
            lane_window.clear()

    # -- observation -------------------------------------------------------------------

    def observe(
        self,
        lane: Lane,
        ok: bool,
        ttft_ms: float | None = None,
        e2e_ms: float | None = None,
        now: float | None = None,
    ) -> None:
        """Record one finished request.

        Accepted in every state: the stable lane keeps producing traffic before and after a
        rollout, and having its window already warm when :meth:`start` is called is what
        makes the very first ratio comparison meaningful.
        """
        self.window(lane).observe(ok, ttft_ms=ttft_ms, e2e_ms=e2e_ms, now=self._now(now))

    # -- the gate ----------------------------------------------------------------------

    def tick(
        self,
        now: float | None = None,
        *,
        stable: LaneSummary | None = None,
        canary: LaneSummary | None = None,
    ) -> Decision:
        """Evaluate the gates once and return the resulting :class:`Decision`.

        ``stable``/``canary`` override the internally-kept windows and are how the
        Kubernetes driver feeds Prometheus-derived numbers through the same logic. When a
        rollout is not running the call is a no-op ``HOLD`` that is *not* appended to
        :attr:`history`, so a control loop may poll freely without diluting the record.
        """
        moment = self._now(now)
        if self._state is not CanaryState.CANARY:
            return Decision(
                kind=DecisionKind.HOLD,
                reason=f"no rollout in progress (state is {self._state})",
                at=moment,
                state=self._state,
                weight=self.canary_weight,
                step_index=self._step_index,
                version=self._version,
            )

        stable_summary = stable if stable is not None else self.summary("stable", moment)
        canary_summary = canary if canary is not None else self.summary("canary", moment)
        config = self.config
        elapsed = moment - self._step_started_at

        if canary_summary.requests < config.min_requests:
            if config.stall_timeout_s is not None and elapsed >= config.stall_timeout_s:
                self._state = CanaryState.ROLLED_BACK
                return self._record(
                    DecisionKind.ROLLBACK,
                    (
                        f"step {self.canary_weight}% saw only {canary_summary.requests} of the "
                        f"{config.min_requests} required canary requests after "
                        f"{elapsed:.1f}s; treating the lane as stalled"
                    ),
                    moment,
                    stable_summary,
                    canary_summary,
                )
            return self._record(
                DecisionKind.HOLD,
                (
                    f"waiting for canary traffic: {canary_summary.requests} of "
                    f"{config.min_requests} requests in the {config.window_s:g}s window"
                ),
                moment,
                stable_summary,
                canary_summary,
            )

        breach = self._breach(stable_summary, canary_summary)
        if breach is not None:
            self._state = CanaryState.ROLLED_BACK
            logger.warning("canary rollback version=%s reason=%s", self._version, breach)
            return self._record(
                DecisionKind.ROLLBACK, breach, moment, stable_summary, canary_summary
            )

        if elapsed < config.step_hold_s:
            return self._record(
                DecisionKind.HOLD,
                (
                    f"step {self.canary_weight}% healthy, held {elapsed:.1f}s of "
                    f"{config.step_hold_s:g}s"
                ),
                moment,
                stable_summary,
                canary_summary,
            )

        if self._step_index >= len(config.steps) - 1:
            self._state = CanaryState.PROMOTED
            logger.info("canary promote version=%s", self._version)
            return self._record(
                DecisionKind.PROMOTE,
                f"step 100% held {elapsed:.1f}s within every gate; promoting {self._version}",
                moment,
                stable_summary,
                canary_summary,
            )

        self._step_index += 1
        self._step_started_at = moment
        logger.info("canary advance version=%s weight=%d%%", self._version, self.canary_weight)
        return self._record(
            DecisionKind.ADVANCE,
            f"advancing to {self.canary_weight}% after {elapsed:.1f}s within every gate",
            moment,
            stable_summary,
            canary_summary,
        )

    # -- internals ---------------------------------------------------------------------

    def _breach(self, stable: LaneSummary, canary: LaneSummary) -> str | None:
        """Return the reason the canary fails its gates, or ``None`` if it passes."""
        config = self.config
        if canary.error_rate > config.max_error_rate:
            return (
                f"canary error rate {canary.error_rate:.4f} "
                f"({canary.errors}/{canary.requests}) exceeds the limit "
                f"{config.max_error_rate:.4f}"
            )
        if (
            config.max_p95_ttft_ms is not None
            and canary.p95_ttft_ms is not None
            and canary.p95_ttft_ms > config.max_p95_ttft_ms
        ):
            return (
                f"canary p95 TTFT {canary.p95_ttft_ms:.1f}ms exceeds the limit "
                f"{config.max_p95_ttft_ms:.1f}ms"
            )
        if stable.requests < config.min_requests:
            # Not enough baseline to compare against; the absolute gates above still applied.
            return None
        pairs = (
            ("TTFT", canary.p95_ttft_ms, stable.p95_ttft_ms),
            ("E2E", canary.p95_e2e_ms, stable.p95_e2e_ms),
        )
        for name, canary_p95, stable_p95 in pairs:
            if canary_p95 is None or stable_p95 is None or stable_p95 <= 0.0:
                continue
            ratio = canary_p95 / stable_p95
            if ratio > config.max_p95_ratio_vs_stable:
                return (
                    f"canary p95 {name} is {ratio:.2f}x the stable lane "
                    f"({canary_p95:.1f}ms vs {stable_p95:.1f}ms), over the limit "
                    f"{config.max_p95_ratio_vs_stable:.2f}x"
                )
        return None

    def _record(
        self,
        kind: DecisionKind,
        reason: str,
        at: float,
        stable: LaneSummary | None = None,
        canary: LaneSummary | None = None,
    ) -> Decision:
        decision = Decision(
            kind=kind,
            reason=reason,
            at=at,
            state=self._state,
            weight=self.canary_weight,
            step_index=self._step_index,
            version=self._version,
            stable=stable,
            canary=canary,
        )
        self._history.append(decision)
        return decision

    def _now(self, now: float | None) -> float:
        return float(now) if now is not None else float(self._clock())
