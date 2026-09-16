"""The on-disk result schema: per-request records, percentile maths, and the run file.

Every benchmark scenario ends by writing one JSON file described by :class:`RunResult`,
and every table, plot and README figure in this repository is rendered from those files.
Nothing downstream re-derives a number from prose, so the schema has to carry enough to
reconstruct the run: the raw per-request records (not just their percentiles), the machine
and software versions that produced them, the configuration, the git sha, and the price of
the GPU-hour the run consumed.

Two fields exist purely for honesty. ``provenance`` distinguishes a ``"measured"`` run
from a ``"projected"`` reference table, and ``provenance_note`` says in one line how a
projected file is to be replaced by a real one. The report renderer prints that line under
every table it draws, so a reader never has to guess which they are looking at.

Timestamps in the records are ``time.monotonic_ns()`` integers from the load generator's
process: latency is a difference, and a monotonic clock cannot make one negative. The
human-readable ``started_at``/``finished_at`` on the run are UTC wall-clock, for indexing.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "PERCENTILE_QUANTILES",
    "SCHEMA_VERSION",
    "SOFTWARE_PACKAGES",
    "SLO",
    "Percentiles",
    "Provenance",
    "RequestRecord",
    "RunResult",
    "percentile",
    "software_versions",
    "utc_now_iso",
]

#: Bumped whenever a field is removed or its meaning changes, so the renderer can refuse a
#: file it would misread. Adding an optional field does not bump it.
SCHEMA_VERSION = "1"

Provenance = Literal["measured", "projected"]

#: The quantiles every latency distribution in a result file reports.
PERCENTILE_QUANTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)

#: Packages whose versions are recorded with every run. A missing package is recorded as
#: ``None`` rather than omitted, so a run made without one of the production engines
#: installed is distinguishable from an older file that never looked.
SOFTWARE_PACKAGES: tuple[str, ...] = (
    "turboserve",
    "torch",
    "transformers",
    "triton",
    "fastapi",
    "httpx",
    "vllm",
    "sglang",
)

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string, the format used in run files."""
    return datetime.now(UTC).isoformat()


def software_versions(packages: Iterable[str] = SOFTWARE_PACKAGES) -> dict[str, str | None]:
    """Installed versions of the packages that can change a measurement's meaning."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q``-th percentile of ``values`` by linear interpolation between ranks.

    This is the definition numpy calls ``method="linear"`` and the one every other tool in
    this repository uses: with ``n`` sorted samples, rank ``r = q/100 * (n - 1)``, and the
    result interpolates between ``sorted[floor(r)]`` and ``sorted[ceil(r)]``. Nearest-rank
    would be defensible too, but mixing the two across the load generator, the canary
    controller and the report renderer would make p95 figures disagree with each other, so
    one definition lives here and everything else calls it.

    Returns ``None`` for an empty sample rather than raising or inventing a zero: a
    percentile of nothing is not a number, and a zero would silently improve a table.
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


@dataclass(frozen=True, slots=True)
class Percentiles:
    """A latency distribution summarised the same way everywhere in the repo."""

    count: int = 0
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: float | None = None
    max: float | None = None

    @classmethod
    def from_values(cls, values: Sequence[float]) -> Percentiles:
        """Summarise a sample; an empty sample yields ``count=0`` and ``None`` elsewhere."""
        usable = [float(value) for value in values if value is not None and math.isfinite(value)]
        if not usable:
            return cls()
        return cls(
            count=len(usable),
            mean=math.fsum(usable) / len(usable),
            p50=percentile(usable, 50.0),
            p90=percentile(usable, 90.0),
            p95=percentile(usable, 95.0),
            p99=percentile(usable, 99.0),
            min=min(usable),
            max=max(usable),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for embedding in the run's ``summary``."""
        return asdict(self)

    @property
    def is_empty(self) -> bool:
        """Whether the sample had no usable value."""
        return self.count == 0


@dataclass(frozen=True, slots=True)
class SLO:
    """Latency objectives a request must meet to count towards goodput.

    Any field left ``None`` is not asserted, so ``SLO(ttft_ms=500)`` means "we only care
    about time to first token". Goodput -- throughput restricted to requests that met the
    objective -- is the number a serving system is actually judged on: raising throughput
    by letting tail latency explode shows up here as a drop.
    """

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Plain dict for embedding in the run's ``summary``."""
        return asdict(self)

    def is_met_by(self, record: RequestRecord) -> bool:
        """Whether ``record`` succeeded *and* satisfied every asserted objective."""
        if not record.ok:
            return False
        checks = (
            (self.ttft_ms, record.ttft_ms),
            (self.tpot_ms, record.tpot_ms),
            (self.e2e_ms, record.e2e_ms),
        )
        for limit, measured in checks:
            if limit is None:
                continue
            if measured is None or measured > limit:
                return False
        return True


@dataclass(slots=True)
class RequestRecord:
    """Everything the load generator observed about one request.

    Raw timestamps rather than pre-computed latencies, because the aggregate a reader
    wants later is rarely the one the scenario chose to compute: keeping ``t_send_ns``,
    ``t_first_ns``, ``t_last_ns`` and the full inter-token series lets the report code
    re-derive any percentile, and lets a reviewer check the arithmetic.

    ``lane`` is ``"stable"`` or ``"canary"`` during a progressive rollout and ``"stable"``
    otherwise; it is kept per request so a canary's latency can be compared against the
    stable lane measured under the identical load.
    """

    request_id: str
    tenant: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    t_send_ns: int = 0
    t_first_ns: int | None = None
    t_last_ns: int | None = None
    itl_ns: list[int] = field(default_factory=list)
    """Gaps between consecutive output tokens, in nanoseconds; ``output_tokens - 1`` long."""

    ok: bool = True
    error: str | None = None
    backend: str = ""
    lane: str = "stable"

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token in milliseconds, or ``None`` if none arrived."""
        if self.t_first_ns is None:
            return None
        return (self.t_first_ns - self.t_send_ns) / NS_PER_MS

    @property
    def e2e_ms(self) -> float | None:
        """End-to-end latency in milliseconds, or ``None`` if the request never finished."""
        if self.t_last_ns is None:
            return None
        return (self.t_last_ns - self.t_send_ns) / NS_PER_MS

    @property
    def tpot_ms(self) -> float | None:
        """Mean milliseconds per output token after the first.

        ``None`` below two output tokens: the decode phase has no measurable slope yet,
        and the first token's cost is TTFT, reported separately.

        Also ``None`` when the last token was observed at the same instant as the first,
        which is what a blocking backend produces -- ``transformers.generate`` returns
        nothing until the whole completion exists, so a client observes every token at
        once. The decode phase has no measurable slope there either, and a literal
        ``0.0 ms`` per token would render as the fastest decoder ever measured.
        """
        if self.output_tokens < 2 or self.t_first_ns is None or self.t_last_ns is None:
            return None
        if self.t_last_ns <= self.t_first_ns:
            return None
        return (self.t_last_ns - self.t_first_ns) / (self.output_tokens - 1) / NS_PER_MS

    @property
    def itl_ms(self) -> list[float]:
        """The inter-token gaps in milliseconds."""
        return [gap / NS_PER_MS for gap in self.itl_ns]

    @property
    def total_tokens(self) -> int:
        """Prompt plus output tokens."""
        return self.prompt_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the JSON object stored in the run file."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestRecord:
        """Rebuild a record, ignoring fields a newer writer added."""
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(slots=True)
class RunResult:
    """One benchmark run: its inputs, its machine, its raw records and its summary.

    Build it with :meth:`start` (which captures the hardware and software blocks), append
    :class:`RequestRecord` objects as the run proceeds, then call :meth:`finish` and
    :meth:`save`.
    """

    scenario: str
    profile: str
    config: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    git_sha: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    gpu_price_per_hour: float | None = None
    """$/GPU-hour of the rented instance, used to derive cost per million tokens."""

    price_source: str | None = None
    """Where that price came from, e.g. the instance's own price at run time."""

    provenance: Provenance = "measured"
    provenance_note: str | None = None
    requests: list[RequestRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.provenance not in ("measured", "projected"):
            raise ValueError(
                f"provenance must be 'measured' or 'projected', got {self.provenance!r}"
            )

    # -- construction ------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        scenario: str,
        profile: str,
        *,
        config: dict[str, Any] | None = None,
        provenance: Provenance = "measured",
        provenance_note: str | None = None,
        gpu_price_per_hour: float | None = None,
        price_source: str | None = None,
        hardware: dict[str, Any] | None = None,
    ) -> RunResult:
        """Open a run, capturing the machine and the software stack up front.

        Captured at the start, not at the end, so that a run which crashes half way still
        leaves a file that says what it was running on.
        """
        if hardware is None:
            from turboserve.hwinfo import collect

            hardware = collect()
        git = hardware.get("git") if isinstance(hardware, dict) else None
        return cls(
            scenario=scenario,
            profile=profile,
            config=dict(config or {}),
            hardware=hardware,
            software=dict(software_versions()),
            git_sha=git.get("sha") if isinstance(git, dict) else None,
            started_at=utc_now_iso(),
            gpu_price_per_hour=gpu_price_per_hour,
            price_source=price_source,
            provenance=provenance,
            provenance_note=provenance_note,
        )

    def add(self, record: RequestRecord) -> None:
        """Append one observed request."""
        self.requests.append(record)

    def finish(self, *, slo: SLO | None = None) -> dict[str, Any]:
        """Stamp the finish time and compute :attr:`summary`; returns the summary."""
        self.finished_at = utc_now_iso()
        return self.summarize(slo=slo)

    # -- aggregation -------------------------------------------------------------------

    @property
    def ok_requests(self) -> list[RequestRecord]:
        """Requests that completed without an error."""
        return [record for record in self.requests if record.ok]

    def wall_seconds(self) -> float:
        """Seconds from the first request sent to the last token received.

        Derived from the records rather than from ``started_at``/``finished_at`` so that
        process start-up, model loading and result writing do not depress the throughput
        figures. Zero when nothing completed.
        """
        completed = [record for record in self.requests if record.t_last_ns is not None]
        if not completed:
            return 0.0
        first_send = min(record.t_send_ns for record in self.requests)
        last_token = max(record.t_last_ns for record in completed if record.t_last_ns)
        return max(last_token - first_send, 0) / NS_PER_S

    def summarize(self, *, slo: SLO | None = None) -> dict[str, Any]:
        """Aggregate the records into the ``summary`` block and return it.

        Latency distributions are built from successful requests only -- a request that
        errored after 3 ms would otherwise flatter the p50 -- while ``error_rate`` counts
        every request, so the two together describe the run.
        """
        ok = self.ok_requests
        total = len(self.requests)
        wall_s = self.wall_seconds()
        output_tokens = sum(record.output_tokens for record in ok)
        total_tokens = sum(record.total_tokens for record in ok)
        itl_ms: list[float] = []
        for record in ok:
            itl_ms.extend(record.itl_ms)

        summary: dict[str, Any] = {
            "num_requests": total,
            "num_ok": len(ok),
            "num_failed": total - len(ok),
            "error_rate": (total - len(ok)) / total if total else 0.0,
            "wall_s": wall_s,
            "prompt_tokens": sum(record.prompt_tokens for record in ok),
            "output_tokens": output_tokens,
            "ttft_ms": Percentiles.from_values(
                [value for value in (record.ttft_ms for record in ok) if value is not None]
            ).to_dict(),
            "itl_ms": Percentiles.from_values(itl_ms).to_dict(),
            "tpot_ms": Percentiles.from_values(
                [value for value in (record.tpot_ms for record in ok) if value is not None]
            ).to_dict(),
            "e2e_ms": Percentiles.from_values(
                [value for value in (record.e2e_ms for record in ok) if value is not None]
            ).to_dict(),
            "output_tok_s": output_tokens / wall_s if wall_s > 0 else 0.0,
            "total_tok_s": total_tokens / wall_s if wall_s > 0 else 0.0,
            "req_s": len(ok) / wall_s if wall_s > 0 else 0.0,
        }
        summary["cost_per_1m_output_tokens_usd"] = self._cost_per_1m(output_tokens, wall_s)
        summary["goodput"] = self.goodput(slo) if slo is not None else None
        self.summary = summary
        return summary

    def goodput(self, slo: SLO) -> dict[str, Any]:
        """Requests per second that met ``slo``, and what fraction of the load that is."""
        wall_s = self.wall_seconds()
        met = [record for record in self.requests if slo.is_met_by(record)]
        total = len(self.requests)
        return {
            "slo": slo.to_dict(),
            "num_met": len(met),
            "ratio": len(met) / total if total else 0.0,
            "req_s": len(met) / wall_s if wall_s > 0 else 0.0,
        }

    def _cost_per_1m(self, output_tokens: int, wall_s: float) -> float | None:
        """Dollars per million output tokens, or ``None`` without a recorded GPU price."""
        if self.gpu_price_per_hour is None or output_tokens <= 0 or wall_s <= 0:
            return None
        cost = self.gpu_price_per_hour * (wall_s / 3600.0)
        return cost / output_tokens * 1_000_000

    # -- serialisation -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The exact JSON object written to disk, with the schema version first."""
        return {
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "profile": self.profile,
            "provenance": self.provenance,
            "provenance_note": self.provenance_note,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "git_sha": self.git_sha,
            "gpu_price_per_hour": self.gpu_price_per_hour,
            "price_source": self.price_source,
            "config": self.config,
            "hardware": self.hardware,
            "software": self.software,
            "summary": self.summary,
            "requests": [record.to_dict() for record in self.requests],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        """Rebuild a run from a parsed JSON object, rejecting an unreadable schema."""
        version = str(data.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"result schema_version {version!r} cannot be read by this build "
                f"(expected {SCHEMA_VERSION!r})"
            )
        return cls(
            scenario=str(data["scenario"]),
            profile=str(data["profile"]),
            config=dict(data.get("config") or {}),
            hardware=dict(data.get("hardware") or {}),
            software=dict(data.get("software") or {}),
            git_sha=data.get("git_sha"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            gpu_price_per_hour=data.get("gpu_price_per_hour"),
            price_source=data.get("price_source"),
            provenance=data.get("provenance", "measured"),
            provenance_note=data.get("provenance_note"),
            requests=[RequestRecord.from_dict(item) for item in data.get("requests") or []],
            summary=dict(data.get("summary") or {}),
            schema_version=version,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to JSON text."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> RunResult:
        """Parse JSON text produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: Path | str) -> RunResult:
        """Read a run file from disk."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(
        self,
        path: Path | str,
        *,
        index_path: Path | str | None = None,
        update_index: bool = True,
    ) -> Path:
        """Write the run file and register it in ``results/index.json``.

        Both writes are atomic (write a sibling temporary file, ``os.replace`` it into
        place) and the index update holds an exclusive ``flock`` for its read-modify-write,
        because scenarios are run in parallel on the measurement host and a torn index
        would lose runs that took minutes of GPU time to produce.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, self.to_json())
        if update_index:
            resolved_index = (
                Path(index_path) if index_path is not None else default_index_path(target)
            )
            append_to_index(resolved_index, self.index_entry(target, resolved_index))
        return target

    def index_entry(self, path: Path, index_path: Path) -> dict[str, Any]:
        """The compact row describing this run in ``results/index.json``.

        The path is stored relative to the index so a results directory stays valid when
        the repository is cloned somewhere else.
        """
        try:
            relative = path.resolve().relative_to(index_path.resolve().parent).as_posix()
        except ValueError:
            relative = path.resolve().as_posix()
        return {
            "path": relative,
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "profile": self.profile,
            "provenance": self.provenance,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "git_sha": self.git_sha,
            "gpu_name": self.hardware.get("gpu_name") if self.hardware else None,
            "num_requests": len(self.requests),
        }


def default_index_path(result_path: Path) -> Path:
    """Locate ``index.json`` for a run file written under a ``results`` directory.

    Scenarios write ``results/<scenario>/<timestamp>.json``, so the index normally sits two
    levels up; when a file is written somewhere else (a temporary directory in a test) the
    index lands next to it instead of escaping into an unrelated tree.
    """
    for parent in result_path.resolve().parents:
        if parent.name == "results":
            return parent / "index.json"
    return result_path.resolve().parent / "index.json"


def append_to_index(index_path: Path, entry: dict[str, Any]) -> None:
    """Append one entry to the JSON-array index under an exclusive lock."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_suffix(index_path.suffix + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            entries = _read_index(index_path)
            entries.append(entry)
            _atomic_write(index_path, json.dumps(entries, indent=2))
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_index(index_path: Path) -> list[dict[str, Any]]:
    """Existing index entries; an absent or unparsable index starts a fresh list."""
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("results index %s is not valid JSON; starting a new index", index_path)
        return []
    if not isinstance(data, list):
        logger.warning("results index %s is not a JSON array; starting a new index", index_path)
        return []
    return [item for item in data if isinstance(item, dict)]


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader sees either the old or the new file."""
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
