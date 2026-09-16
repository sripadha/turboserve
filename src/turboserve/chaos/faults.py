"""Fault schedules: what breaks, when, and which replica it happens to.

A chaos run is only useful if it is *repeatable*, and a schedule written as prose is not.
So a schedule is a list of short strings, parsed here into :class:`Fault` objects and
expanded into an explicit, sorted timeline of :class:`FaultEvent` before a single request is
sent. Expanding up front rather than deciding as the run proceeds buys three things: the
timeline can be printed and reviewed (``turboserve chaos plan``), it is embedded verbatim in
the result file so a run can be replayed, and a test can assert on it without running
anything.

The grammar is ``kind:key=value,key=value``, and there are four kinds:

``kill:every=10s``
    Take a replica out every ten seconds and bring it back. ``grace`` (default zero) is a
    drain period: the replica is marked unready and refuses *new* requests for that long
    before it is actually killed, which is what ``kubectl delete pod`` does with a preStop
    hook and a readiness gate. With ``grace=0`` the process dies with requests in flight and
    those streams are lost -- that difference is the single biggest lever on the error rate a
    run reports, which is exactly why it is explicit here rather than assumed.
``latency:p=0.05,ms=500``
    Add 500 ms before the first token on 5 % of requests. A slow replica is more dangerous
    than a dead one: a dead one is retried immediately, a slow one holds the request and
    shows up only in the tail.
``error:p=0.01``
    Fail 1 % of requests before the first token. Retryable by construction, so what this
    measures is whether the router's retry path actually hides it.
``partition:at=20s,for=5s``
    Make one replica unreachable for a window -- every endpoint, health probe included,
    which is what a network partition looks like from the gateway.

Durations accept ``ms``/``s``/``m``/``h`` suffixes (a bare number means seconds) and
probabilities accept ``0.05`` or ``5%``. Nothing here performs I/O or looks at a clock: the
timeline is pure data, and the harness in :mod:`turboserve.chaos.harness` is what turns it
into signals and HTTP calls.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GRACE_S",
    "DEFAULT_RESTART_DELAY_S",
    "FAULT_KINDS",
    "Fault",
    "FaultAction",
    "FaultError",
    "FaultEvent",
    "FaultKind",
    "FaultSchedule",
    "SteadyFaults",
    "outage_windows",
    "parse_duration",
    "parse_milliseconds",
    "parse_probability",
]

FaultKind = Literal["kill", "latency", "error", "partition"]

#: Every kind the parser accepts, in the order ``--help`` lists them.
FAULT_KINDS: Final[tuple[FaultKind, ...]] = ("kill", "latency", "error", "partition")

#: Seconds between a replica being killed and being started again.
DEFAULT_RESTART_DELAY_S: Final = 2.0

#: Seconds of drain before the kill. Zero -- an unannounced death -- is the default because
#: that is what the word "kill" means; a graceful rollout is the special case and has to be
#: asked for, so that a result file can never quietly describe an easier experiment than its
#: schedule string suggests.
DEFAULT_GRACE_S: Final = 0.0

_DURATION_UNITS: Final[dict[str, float]] = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

#: Parameters each kind accepts. Anything else is a typo and is rejected rather than
#: ignored: a silently dropped ``ms=500`` would produce a run that measured nothing.
_ALLOWED_KEYS: Final[dict[str, frozenset[str]]] = {
    "kill": frozenset({"every", "start", "restart", "grace", "target"}),
    "latency": frozenset({"p", "ms"}),
    "error": frozenset({"p"}),
    "partition": frozenset({"at", "for", "target"}),
}


class FaultError(ValueError):
    """A fault specification is unparsable, incomplete or contradictory."""


class FaultAction(StrEnum):
    """One thing the harness does to one replica at one instant."""

    DRAIN = "drain"
    KILL = "kill"
    RESTART = "restart"
    PARTITION_START = "partition_start"
    PARTITION_END = "partition_end"


#: Tie-break order for events sharing a timestamp: recoveries are applied before
#: disruptions, so a schedule that brings one replica back exactly as another goes down
#: never passes through a moment with the whole fleet down.
_ACTION_ORDER: Final[dict[FaultAction, int]] = {
    FaultAction.RESTART: 0,
    FaultAction.PARTITION_END: 1,
    FaultAction.DRAIN: 2,
    FaultAction.KILL: 3,
    FaultAction.PARTITION_START: 4,
}

#: Actions that make a replica unable to take new traffic, and those that undo them.
_DOWN_ACTIONS: Final[frozenset[FaultAction]] = frozenset(
    {FaultAction.DRAIN, FaultAction.KILL, FaultAction.PARTITION_START}
)
_UP_ACTIONS: Final[frozenset[FaultAction]] = frozenset(
    {FaultAction.RESTART, FaultAction.PARTITION_END}
)


def parse_duration(text: str) -> float:
    """Seconds from ``"500ms"``, ``"10s"``, ``"2m"``, ``"1h"`` or a bare number.

    A bare number is seconds, which is what every such field in this repository means;
    the suffixes exist because ``every=10s`` reads as intended and ``every=10`` does not.
    """
    raw = str(text).strip().lower()
    if not raw:
        raise FaultError("empty duration")
    number, scale = raw, 1.0
    # Longest suffix first, so "500ms" is milliseconds rather than 500 metres of "m".
    for suffix in sorted(_DURATION_UNITS, key=len, reverse=True):
        if raw.endswith(suffix):
            number, scale = raw[: -len(suffix)].strip(), _DURATION_UNITS[suffix]
            break
    try:
        value = float(number)
    except ValueError as exc:
        raise FaultError(f"cannot read {text!r} as a duration") from exc
    if value < 0:
        raise FaultError(f"duration must not be negative, got {text!r}")
    return value * scale


def parse_probability(text: str) -> float:
    """A probability in ``[0, 1]`` from ``"0.05"`` or ``"5%"``."""
    raw = str(text).strip()
    if not raw:
        raise FaultError("empty probability")
    scale = 1.0
    if raw.endswith("%"):
        raw, scale = raw[:-1].strip(), 0.01
    try:
        value = float(raw) * scale
    except ValueError as exc:
        raise FaultError(f"cannot read {text!r} as a probability") from exc
    if not 0.0 <= value <= 1.0:
        raise FaultError(f"probability must be in [0, 1], got {text!r}")
    return value


def parse_milliseconds(text: str) -> float:
    """Milliseconds from a bare number, or from any duration with an explicit unit.

    ``ms=500`` is the documented spelling and means 500 milliseconds, but ``ms=1.5s``
    should not silently become 1.5 milliseconds, so a value carrying a unit is read as a
    duration and converted.
    """
    raw = str(text).strip().lower()
    if not raw:
        raise FaultError("empty latency")
    if any(raw.endswith(unit) for unit in _DURATION_UNITS):
        return parse_duration(raw) * 1000.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise FaultError(f"cannot read {text!r} as a number of milliseconds") from exc
    if value < 0.0:
        raise FaultError(f"latency must not be negative, got {text!r}")
    return value


def _split_params(text: str) -> dict[str, str]:
    """Parse the ``key=value,key=value`` tail of a fault specification."""
    params: dict[str, str] = {}
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        key, separator, value = item.partition("=")
        if not separator:
            raise FaultError(f"fault parameter {item!r} is not of the form key=value")
        name = key.strip().lower()
        if name in params:
            raise FaultError(f"fault parameter {name!r} given twice")
        params[name] = value.strip()
    return params


@dataclass(frozen=True, slots=True)
class Fault:
    """One parsed fault specification.

    A single flat record rather than a class per kind: the fields a kind does not use stay
    at their defaults, the whole thing serialises into a result file as one JSON object, and
    :meth:`FaultSchedule.events` can dispatch on :attr:`kind` in one place. Validation is in
    ``__post_init__`` so that a fault built in code is checked exactly like a parsed one.
    """

    kind: FaultKind
    every_s: float | None = None
    start_s: float | None = None
    at_s: float | None = None
    for_s: float | None = None
    restart_delay_s: float = DEFAULT_RESTART_DELAY_S
    grace_s: float = DEFAULT_GRACE_S
    probability: float = 0.0
    latency_ms: float = 0.0
    target: str | None = None
    raw: str = ""

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise FaultError(f"unknown fault kind {self.kind!r}; known: {', '.join(FAULT_KINDS)}")
        if self.restart_delay_s < 0.0 or self.grace_s < 0.0:
            raise FaultError("restart and grace must not be negative")
        if not 0.0 <= self.probability <= 1.0:
            raise FaultError(f"probability must be in [0, 1], got {self.probability}")
        if self.kind == "kill":
            if self.every_s is None or self.every_s <= 0.0:
                raise FaultError("kill needs every=<duration> greater than zero")
            if self.start_s is not None and self.start_s < 0.0:
                raise FaultError("kill start must not be negative")
        elif self.kind == "partition":
            if self.at_s is None:
                raise FaultError("partition needs at=<duration>")
            if self.for_s is None or self.for_s <= 0.0:
                raise FaultError("partition needs for=<duration> greater than zero")
        elif self.kind == "latency":
            if self.latency_ms <= 0.0:
                raise FaultError("latency needs ms=<milliseconds> greater than zero")
            if self.probability <= 0.0:
                raise FaultError("latency needs p=<probability> greater than zero")
        elif self.probability <= 0.0:  # error
            raise FaultError("error needs p=<probability> greater than zero")

    @classmethod
    def parse(cls, text: str) -> Fault:
        """Parse one ``kind:key=value,...`` specification."""
        spec = str(text).strip()
        if not spec:
            raise FaultError("empty fault specification")
        head, separator, tail = spec.partition(":")
        kind = head.strip().lower()
        if kind not in FAULT_KINDS:
            raise FaultError(
                f"unknown fault kind {head.strip()!r}; known: {', '.join(FAULT_KINDS)}"
            )
        params = _split_params(tail) if separator else {}
        unknown = set(params) - _ALLOWED_KEYS[kind]
        if unknown:
            raise FaultError(
                f"{kind} takes {', '.join(sorted(_ALLOWED_KEYS[kind]))}; "
                f"got unknown parameter(s) {', '.join(sorted(unknown))}"
            )
        target = params.get("target") or None
        if kind == "kill":
            return cls(
                kind="kill",
                every_s=parse_duration(params["every"]) if "every" in params else None,
                start_s=parse_duration(params["start"]) if "start" in params else None,
                restart_delay_s=parse_duration(params.get("restart", ""))
                if "restart" in params
                else DEFAULT_RESTART_DELAY_S,
                grace_s=parse_duration(params["grace"]) if "grace" in params else DEFAULT_GRACE_S,
                target=target,
                raw=spec,
            )
        if kind == "partition":
            return cls(
                kind="partition",
                at_s=parse_duration(params["at"]) if "at" in params else None,
                for_s=parse_duration(params["for"]) if "for" in params else None,
                target=target,
                raw=spec,
            )
        if kind == "latency":
            return cls(
                kind="latency",
                probability=parse_probability(params.get("p", "0")),
                latency_ms=parse_milliseconds(params["ms"]) if "ms" in params else 0.0,
                raw=spec,
            )
        return cls(kind="error", probability=parse_probability(params.get("p", "0")), raw=spec)

    @property
    def first_at_s(self) -> float:
        """When this fault first fires, in seconds from the start of the run."""
        if self.kind == "kill" and self.every_s is not None:
            return self.every_s if self.start_s is None else self.start_s
        return self.at_s or 0.0

    def describe(self) -> str:
        """The canonical specification string, which re-parses to an equal fault."""
        if self.raw:
            return self.raw
        parts: list[str] = []
        if self.kind == "kill":
            parts.append(f"every={_fmt_duration(self.every_s or 0.0)}")
            if self.start_s is not None:
                parts.append(f"start={_fmt_duration(self.start_s)}")
            if self.restart_delay_s != DEFAULT_RESTART_DELAY_S:
                parts.append(f"restart={_fmt_duration(self.restart_delay_s)}")
            if self.grace_s != DEFAULT_GRACE_S:
                parts.append(f"grace={_fmt_duration(self.grace_s)}")
        elif self.kind == "partition":
            parts.append(f"at={_fmt_duration(self.at_s or 0.0)}")
            parts.append(f"for={_fmt_duration(self.for_s or 0.0)}")
        elif self.kind == "latency":
            parts.append(f"p={self.probability:g}")
            parts.append(f"ms={self.latency_ms:g}")
        else:
            parts.append(f"p={self.probability:g}")
        if self.target:
            parts.append(f"target={self.target}")
        return f"{self.kind}:{','.join(parts)}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description, embedded in the run's ``config`` block."""
        data: dict[str, Any] = {"kind": self.kind, "spec": self.describe()}
        for key, value in (
            ("every_s", self.every_s),
            ("start_s", self.start_s),
            ("at_s", self.at_s),
            ("for_s", self.for_s),
            ("target", self.target),
        ):
            if value is not None:
                data[key] = value
        if self.kind == "kill":
            data["restart_delay_s"] = self.restart_delay_s
            data["grace_s"] = self.grace_s
        if self.kind in ("latency", "error"):
            data["probability"] = self.probability
        if self.kind == "latency":
            data["latency_ms"] = self.latency_ms
        return data


def _fmt_duration(seconds: float) -> str:
    """Render seconds the way the grammar accepts them back."""
    if seconds and seconds < 1.0:
        return f"{seconds * 1000.0:g}ms"
    return f"{seconds:g}s"


@dataclass(frozen=True, slots=True)
class FaultEvent:
    """One dated instruction in an expanded schedule."""

    t_s: float
    action: FaultAction
    target: str
    kind: FaultKind
    spec: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, stored in the result file's chaos block."""
        return {
            "t_s": self.t_s,
            "action": str(self.action),
            "target": self.target,
            "kind": self.kind,
            "spec": self.spec,
        }


@dataclass(frozen=True, slots=True)
class SteadyFaults:
    """The faults that are simply *on* for the whole run, as a worker is configured with them.

    ``latency`` and ``error`` are not events: they are a probability applied to every
    request, so they are pushed into each replica once at the start rather than driven from
    the timeline. Keeping them in their own object is what lets a worker be reconfigured in
    one call, including after a restart -- a replica that came back with its faults cleared
    would quietly make the second half of a run easier than the first.
    """

    latency_probability: float = 0.0
    latency_ms: float = 0.0
    error_probability: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to apply."""
        return self.error_probability <= 0.0 and (
            self.latency_probability <= 0.0 or self.latency_ms <= 0.0
        )

    def to_dict(self) -> dict[str, float]:
        """JSON-ready form."""
        return {
            "latency_probability": self.latency_probability,
            "latency_ms": self.latency_ms,
            "error_probability": self.error_probability,
        }


@dataclass(frozen=True, slots=True)
class FaultSchedule:
    """An ordered set of faults, expandable into a timeline for a given fleet and duration."""

    faults: tuple[Fault, ...] = ()

    def __post_init__(self) -> None:
        for kind in ("latency", "error"):
            if sum(1 for fault in self.faults if fault.kind == kind) > 1:
                raise FaultError(
                    f"at most one {kind!r} fault per schedule: two of them would have to be "
                    "combined into a single probability, and the combination a reader "
                    "expects is not the one the arithmetic gives"
                )

    @classmethod
    def parse(cls, specs: Iterable[str]) -> FaultSchedule:
        """Build a schedule from ``["kill:every=10s", "error:p=0.01"]``."""
        return cls(tuple(Fault.parse(spec) for spec in specs if str(spec).strip()))

    @classmethod
    def from_string(cls, text: str, *, separator: str = ";") -> FaultSchedule:
        """Parse several specifications from one string.

        Semicolon-separated, because a comma already separates a fault's own parameters.
        """
        return cls.parse(text.split(separator))

    def of_kind(self, kind: FaultKind) -> tuple[Fault, ...]:
        """Every fault of one kind, in declaration order."""
        return tuple(fault for fault in self.faults if fault.kind == kind)

    @property
    def is_empty(self) -> bool:
        """Whether the schedule has no faults at all.

        Prefer this to truthiness: the class defines :meth:`__len__`, so an empty schedule
        is falsey, and a bug that swapped a schedule for ``None`` would read the same.
        """
        return not self.faults

    def steady(self) -> SteadyFaults:
        """The per-request probabilities to configure every replica with."""
        latency = self.of_kind("latency")
        errors = self.of_kind("error")
        return SteadyFaults(
            latency_probability=latency[0].probability if latency else 0.0,
            latency_ms=latency[0].latency_ms if latency else 0.0,
            error_probability=errors[0].probability if errors else 0.0,
        )

    def events(
        self,
        *,
        duration_s: float,
        workers: Sequence[str],
        seed: int = 0,
    ) -> tuple[FaultEvent, ...]:
        """Expand the schedule into a sorted timeline over ``workers``.

        Victims are assigned round-robin from an offset drawn by a seeded RNG: round-robin
        so that a repeated ``kill`` spreads over the fleet instead of picking the same
        replica by chance and measuring nothing, and the random offset so that two runs of
        the same schedule with different seeds do not always start on the same replica.

        Events whose trigger falls at or after ``duration_s`` are not generated; the
        ``restart`` that follows a kill *is*, even when it lands past the end, so that the
        timeline says what would have happened and the harness can report a replica that was
        still down when the run ended.
        """
        if duration_s <= 0.0:
            raise FaultError(f"duration_s must be positive, got {duration_s}")
        if not workers:
            raise FaultError("a fault schedule needs at least one worker to target")
        known = set(workers)
        rng = random.Random(seed)
        events: list[FaultEvent] = []
        for fault in self.faults:
            if fault.target is not None and fault.target not in known:
                raise FaultError(
                    f"fault {fault.describe()!r} targets unknown worker {fault.target!r}; "
                    f"known workers: {', '.join(workers)}"
                )
            if fault.kind == "kill":
                events.extend(self._kill_events(fault, duration_s, workers, rng))
            elif fault.kind == "partition":
                events.extend(self._partition_events(fault, duration_s, workers, rng))
        events.sort(key=lambda event: (event.t_s, _ACTION_ORDER[event.action], event.target))
        return tuple(events)

    @staticmethod
    def _kill_events(
        fault: Fault,
        duration_s: float,
        workers: Sequence[str],
        rng: random.Random,
    ) -> list[FaultEvent]:
        """Drain/kill/restart triples for one repeating kill fault."""
        every_s = fault.every_s
        if every_s is None or every_s <= 0.0:  # pragma: no cover - __post_init__ guarantees it
            return []
        offset = rng.randrange(len(workers))
        events: list[FaultEvent] = []
        occurrence = 0
        moment = fault.first_at_s
        while moment < duration_s:
            target = fault.target or workers[(offset + occurrence) % len(workers)]
            killed_at = moment + fault.grace_s
            if fault.grace_s > 0.0:
                events.append(
                    FaultEvent(moment, FaultAction.DRAIN, target, fault.kind, fault.describe())
                )
            events.append(
                FaultEvent(killed_at, FaultAction.KILL, target, fault.kind, fault.describe())
            )
            events.append(
                FaultEvent(
                    killed_at + fault.restart_delay_s,
                    FaultAction.RESTART,
                    target,
                    fault.kind,
                    fault.describe(),
                )
            )
            occurrence += 1
            moment += every_s
        return events

    @staticmethod
    def _partition_events(
        fault: Fault,
        duration_s: float,
        workers: Sequence[str],
        rng: random.Random,
    ) -> list[FaultEvent]:
        """Start/end pair for one partition window."""
        at_s, for_s = fault.at_s, fault.for_s
        if at_s is None or for_s is None:  # pragma: no cover - __post_init__ guarantees them
            return []
        if at_s >= duration_s:
            logger.warning(
                "partition at %.3fs falls outside the %.3fs run and will not be applied",
                at_s,
                duration_s,
            )
            return []
        target = fault.target or workers[rng.randrange(len(workers))]
        return [
            FaultEvent(at_s, FaultAction.PARTITION_START, target, fault.kind, fault.describe()),
            FaultEvent(
                at_s + for_s,
                FaultAction.PARTITION_END,
                target,
                fault.kind,
                fault.describe(),
            ),
        ]

    def describe(self) -> str:
        """The schedule as one string, re-parsable by :meth:`from_string`."""
        return "; ".join(fault.describe() for fault in self.faults)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for the result file."""
        return {
            "spec": self.describe(),
            "faults": [fault.to_dict() for fault in self.faults],
            "steady": self.steady().to_dict(),
        }

    def __iter__(self) -> Iterator[Fault]:
        return iter(self.faults)

    def __len__(self) -> int:
        return len(self.faults)


def outage_windows(
    events: Sequence[FaultEvent],
    workers: Sequence[str],
    *,
    duration_s: float,
) -> tuple[tuple[float, float], ...]:
    """Intervals during which *every* replica is out of service.

    A schedule that takes the whole fleet down is a legitimate experiment -- it measures
    what a client sees during a total outage -- but it is almost never what someone meant to
    write, and it turns the error rate a run reports into a statement about the schedule
    rather than about the gateway. The harness warns when this returns anything, and the
    windows are recorded alongside the results so the number can be read in context.
    """
    if not workers:
        return ()
    fleet = len(set(workers))
    down: set[str] = set()
    windows: list[tuple[float, float]] = []
    started: float | None = None
    for event in sorted(events, key=lambda item: (item.t_s, _ACTION_ORDER[item.action])):
        if event.t_s > duration_s:
            break
        if event.action in _DOWN_ACTIONS:
            down.add(event.target)
        elif event.action in _UP_ACTIONS:
            down.discard(event.target)
        if len(down) >= fleet and started is None:
            started = event.t_s
        elif len(down) < fleet and started is not None:
            windows.append((started, event.t_s))
            started = None
    if started is not None:
        windows.append((started, duration_s))
    return tuple(windows)
