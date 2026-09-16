"""Turning an observed token stream into a record, and records into comparable numbers.

:mod:`turboserve.bench.records` owns the on-disk schema and the percentile maths. This
module owns the two things that sit either side of it:

* :class:`RecordBuilder` -- the *observer*. It watches one request's
  :class:`~turboserve.gateway.backends.protocol.TokenEvent` stream from the client side and
  produces one :class:`~turboserve.bench.records.RequestRecord`. All the awkward decisions
  about what a latency means when a chunk carries several tokens live here, in one place,
  documented, instead of being re-invented by each scenario.
* :func:`summarize_records`, :func:`group_by` and :class:`Comparison` -- the *reducer*.
  Scenarios in this repository are nearly all "A versus B": naive versus continuous
  batching, cache off versus on, base versus adapters. The ratio and delta arithmetic is
  therefore written once, returns ``None`` rather than infinity when a baseline is zero,
  and is what the report renderer prints.

Every timestamp taken here is ``time.monotonic_ns()`` read *by the client*, never the
producer's ``TokenEvent.t_ns``. The two agree for an in-process engine, but for an HTTP
backend the producer's stamp excludes serialisation and the network, and a benchmark that
quietly excluded them would flatter every remote engine it measured. The producer stamp is
still available on the event for anyone who wants to measure that difference explicitly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from turboserve.bench.records import SLO, Percentiles, RequestRecord, RunResult, percentile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable, Sequence

    from turboserve.gateway.backends.protocol import TokenEvent

logger = logging.getLogger(__name__)

__all__ = [
    "BACKEND_KEY",
    "BASELINE_KEY",
    "COMPARE_KEY",
    "DERIVED_KEY",
    "LABEL_KEY",
    "SLO",
    "Comparison",
    "Percentiles",
    "RecordBuilder",
    "RequestRecord",
    "baseline_label",
    "compare_runs",
    "compare_to_labels",
    "comparison_groups",
    "group_by",
    "pct_delta",
    "percentile",
    "ratio",
    "run_label",
    "run_summary",
    "saved_pct",
    "slo_from_mapping",
    "summarize_by",
    "summarize_records",
]


class RecordBuilder:
    """Accumulates one request's observed stream into a :class:`RequestRecord`.

    Usage is three calls: :meth:`sent` when the request goes out, :meth:`observe` for every
    event received, :meth:`build` at the end (or :meth:`failed` instead, if the stream
    died).

    Two conventions it fixes, both of which change the numbers:

    **Tokens per chunk.** A chunk may carry more than one token -- speculative decoding
    accepts several per step, and an HTTP backend may coalesce. The inter-token gap
    recorded for such a chunk is the interval since the previous chunk divided by the
    number of tokens in it: the tokens were produced over that interval even though they
    were observed at its end, and charging the whole interval to one token would put a
    spike into every ITL distribution measured with speculation on.

    **Tokens with no observed interval.** The first chunk's first token defines TTFT, so it
    contributes no gap. If that first chunk carries *k* tokens, the remaining *k-1* also
    contribute none: nothing was observed to elapse for them. The inter-token series is
    therefore ``output_tokens - k`` long, which for the usual one-token-per-chunk stream is
    the ``output_tokens - 1`` that :class:`RequestRecord` describes. Padding the difference
    with zeros would have been the alternative, and would have pulled every percentile of
    the distribution down towards a number nothing measured.

    **Token counting.** ``usage`` on the terminating event wins when the backend sends it,
    because the backend knows how many tokens it actually produced. Otherwise tokens are
    counted from ``token_ids``, and a backend that streams only text is counted one token
    per non-empty chunk -- the truth of what crossed the wire, and the reason the local
    engine and the OpenAI-compatible backend both return ids where they can.
    """

    __slots__ = (
        "_backend",
        "_counted_tokens",
        "_error",
        "_first_chunk_tokens",
        "_itl_ns",
        "_lane",
        "_last_event_ns",
        "_ok",
        "_prompt_tokens",
        "_request_id",
        "_sent",
        "_t_first_ns",
        "_t_last_ns",
        "_t_send_ns",
        "_tenant",
        "_usage",
    )

    def __init__(
        self,
        request_id: str,
        *,
        tenant: str = "",
        prompt_tokens: int = 0,
        backend: str = "",
        lane: str = "stable",
    ) -> None:
        self._request_id = request_id
        self._tenant = tenant
        self._prompt_tokens = prompt_tokens
        self._backend = backend
        self._lane = lane
        self._t_send_ns = 0
        self._sent = False
        self._t_first_ns: int | None = None
        self._t_last_ns: int | None = None
        self._last_event_ns: int | None = None
        self._itl_ns: list[int] = []
        self._counted_tokens = 0
        self._first_chunk_tokens = 0
        self._usage: dict[str, Any] | None = None
        self._ok = True
        self._error: str | None = None

    @property
    def request_id(self) -> str:
        """The request this builder is watching."""
        return self._request_id

    @property
    def started(self) -> bool:
        """Whether :meth:`sent` has been called.

        Tracked with a flag rather than by testing the timestamp against zero, because a
        caller passing an explicit ``t_ns`` (every test, and any replay of recorded
        timestamps) is free to use zero as its origin.
        """
        return self._sent

    def sent(self, t_ns: int | None = None) -> int:
        """Record the moment the request left the client; returns that timestamp."""
        self._t_send_ns = time.monotonic_ns() if t_ns is None else t_ns
        self._sent = True
        return self._t_send_ns

    def observe(self, event: TokenEvent, *, t_ns: int | None = None) -> None:
        """Fold one received event into the record."""
        received = time.monotonic_ns() if t_ns is None else t_ns
        if event.error is not None:
            self._ok = False
            self._error = event.error
        tokens = len(event.token_ids) or (1 if event.text else 0)
        if tokens:
            self._account_tokens(tokens, received)
        if event.usage:
            self._usage = dict(event.usage)
        if event.finished:
            self._t_last_ns = received

    def _account_tokens(self, tokens: int, received: int) -> None:
        """Update the token count, TTFT and the inter-token series for one chunk."""
        if self._t_first_ns is None:
            self._t_first_ns = received
            self._first_chunk_tokens = tokens
        else:
            previous = self._last_event_ns if self._last_event_ns is not None else self._t_first_ns
            gap = max(received - previous, 0)
            share = gap // tokens
            self._itl_ns.extend([share] * tokens)
        self._last_event_ns = received
        self._counted_tokens += tokens

    def failed(self, error: str, *, t_ns: int | None = None) -> None:
        """Mark the request failed; the stream ends here whatever it had produced.

        The finish timestamp is stamped even for a failure, so the run's wall clock covers
        the time the failed request occupied the system. Its latencies are still excluded
        from every distribution, because :meth:`RunResult.summarize` aggregates ``ok``
        records only.
        """
        self._ok = False
        self._error = error
        self._t_last_ns = time.monotonic_ns() if t_ns is None else t_ns

    def build(self) -> RequestRecord:
        """Produce the record. Safe to call more than once; it recomputes from state."""
        usage = self._usage or {}
        output_tokens = _as_int(usage.get("completion_tokens"), self._counted_tokens)
        prompt_tokens = _as_int(usage.get("prompt_tokens"), self._prompt_tokens)
        return RequestRecord(
            request_id=self._request_id,
            tenant=self._tenant,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            t_send_ns=self._t_send_ns,
            t_first_ns=self._t_first_ns,
            t_last_ns=self._t_last_ns,
            itl_ns=list(self._itl_ns),
            ok=self._ok,
            error=self._error,
            backend=self._backend,
            lane=self._lane,
        )


def _as_int(value: Any, fallback: int) -> int:
    """An int from a usage field, falling back when it is absent or not a number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return int(value)


def summarize_records(
    records: Sequence[RequestRecord],
    *,
    slo: SLO | None = None,
    gpu_price_per_hour: float | None = None,
    scenario: str = "",
    profile: str = "",
) -> dict[str, Any]:
    """Summarise an arbitrary set of records with the run summariser.

    Delegates to :meth:`RunResult.summarize` rather than repeating its arithmetic, so a
    per-tenant breakdown and the run-level summary can never drift apart in their
    definition of p95.
    """
    run = RunResult(
        scenario=scenario,
        profile=profile,
        gpu_price_per_hour=gpu_price_per_hour,
        requests=list(records),
    )
    return run.summarize(slo=slo)


def _attr_getter(name: str) -> Callable[[RequestRecord], str]:
    """A key function reading one attribute of a record as a string."""

    def getter(record: RequestRecord) -> str:
        return str(getattr(record, name))

    return getter


def group_by(
    records: Iterable[RequestRecord],
    key: str | Callable[[RequestRecord], str],
) -> dict[str, list[RequestRecord]]:
    """Partition records by an attribute name (``"tenant"``, ``"backend"``, ``"lane"``).

    Insertion-ordered, so the groups come out in the order the load generator first saw
    them and a rendered table is stable between runs of the same workload.
    """
    getter = key if callable(key) else _attr_getter(key)
    grouped: dict[str, list[RequestRecord]] = {}
    for record in records:
        grouped.setdefault(getter(record), []).append(record)
    return grouped


def summarize_by(
    records: Sequence[RequestRecord],
    key: str | Callable[[RequestRecord], str],
    *,
    slo: SLO | None = None,
    gpu_price_per_hour: float | None = None,
) -> dict[str, dict[str, Any]]:
    """A summary per group, for per-tenant and per-lane tables.

    Note that each group's throughput uses that group's own wall clock (first send to last
    token within the group), so the per-group ``req_s`` figures do not add up to the run's
    ``req_s`` unless the groups ran over the same window.
    """
    return {
        name: summarize_records(group, slo=slo, gpu_price_per_hour=gpu_price_per_hour)
        for name, group in group_by(records, key).items()
    }


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    """``candidate / baseline``, or ``None`` when that is not a meaningful number.

    A missing or zero baseline yields ``None`` rather than infinity: "infinitely faster
    than a run that produced nothing" is not a result, and ``None`` renders as an em dash.
    """
    if candidate is None or baseline is None or baseline == 0:
        return None
    return candidate / baseline


def pct_delta(candidate: float | None, baseline: float | None) -> float | None:
    """Signed percentage change from ``baseline`` to ``candidate``.

    Negative means the candidate is smaller, which for a latency is an improvement and for
    a throughput is a regression; the report labels each column so the sign is readable.
    """
    relative = ratio(candidate, baseline)
    if relative is None:
        return None
    return (relative - 1.0) * 100.0


def saved_pct(before: float | None, after: float | None) -> float | None:
    """Percentage of ``before`` that ``after`` no longer uses, e.g. memory saved."""
    if before is None or after is None or before == 0:
        return None
    return (before - after) / before * 100.0


def _stat(summary: Mapping[str, Any], metric: str, statistic: str) -> float | None:
    """One statistic out of a summary's percentile block, tolerating an absent block."""
    block = summary.get(metric)
    if not isinstance(block, Mapping):
        return None
    value = block.get(statistic)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _scalar(summary: Mapping[str, Any], name: str) -> float | None:
    """One top-level numeric field out of a summary."""
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


@dataclass(frozen=True, slots=True)
class Comparison:
    """One candidate measured against one baseline, in the terms the tables use.

    Ratios for throughput (a candidate that does 3x the work of the baseline), percentage
    deltas for latency (a candidate whose p95 is 20% lower). Every field is ``None`` when
    the baseline could not support the comparison, and the renderer prints an em dash.
    """

    baseline: str
    candidate: str
    output_tok_s_ratio: float | None = None
    total_tok_s_ratio: float | None = None
    req_s_ratio: float | None = None
    ttft_p50_delta_pct: float | None = None
    ttft_p95_delta_pct: float | None = None
    itl_p95_delta_pct: float | None = None
    tpot_p95_delta_pct: float | None = None
    e2e_p95_delta_pct: float | None = None
    error_rate_delta: float | None = None
    cost_ratio: float | None = None
    goodput_req_s_ratio: float | None = None

    @classmethod
    def from_summaries(
        cls,
        baseline_label: str,
        candidate_label: str,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> Comparison:
        """Build a comparison from two ``summarize()`` blocks."""
        base_goodput = baseline.get("goodput")
        cand_goodput = candidate.get("goodput")
        goodput_ratio: float | None = None
        if isinstance(base_goodput, Mapping) and isinstance(cand_goodput, Mapping):
            goodput_ratio = ratio(_scalar(cand_goodput, "req_s"), _scalar(base_goodput, "req_s"))
        base_error = _scalar(baseline, "error_rate")
        cand_error = _scalar(candidate, "error_rate")
        return cls(
            baseline=baseline_label,
            candidate=candidate_label,
            output_tok_s_ratio=ratio(
                _scalar(candidate, "output_tok_s"), _scalar(baseline, "output_tok_s")
            ),
            total_tok_s_ratio=ratio(
                _scalar(candidate, "total_tok_s"), _scalar(baseline, "total_tok_s")
            ),
            req_s_ratio=ratio(_scalar(candidate, "req_s"), _scalar(baseline, "req_s")),
            ttft_p50_delta_pct=pct_delta(
                _stat(candidate, "ttft_ms", "p50"), _stat(baseline, "ttft_ms", "p50")
            ),
            ttft_p95_delta_pct=pct_delta(
                _stat(candidate, "ttft_ms", "p95"), _stat(baseline, "ttft_ms", "p95")
            ),
            itl_p95_delta_pct=pct_delta(
                _stat(candidate, "itl_ms", "p95"), _stat(baseline, "itl_ms", "p95")
            ),
            tpot_p95_delta_pct=pct_delta(
                _stat(candidate, "tpot_ms", "p95"), _stat(baseline, "tpot_ms", "p95")
            ),
            e2e_p95_delta_pct=pct_delta(
                _stat(candidate, "e2e_ms", "p95"), _stat(baseline, "e2e_ms", "p95")
            ),
            error_rate_delta=(
                None if base_error is None or cand_error is None else cand_error - base_error
            ),
            cost_ratio=ratio(
                _scalar(candidate, "cost_per_1m_output_tokens_usd"),
                _scalar(baseline, "cost_per_1m_output_tokens_usd"),
            ),
            goodput_req_s_ratio=goodput_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, for embedding in a run summary or a comparison table."""
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "output_tok_s_ratio": self.output_tok_s_ratio,
            "total_tok_s_ratio": self.total_tok_s_ratio,
            "req_s_ratio": self.req_s_ratio,
            "ttft_p50_delta_pct": self.ttft_p50_delta_pct,
            "ttft_p95_delta_pct": self.ttft_p95_delta_pct,
            "itl_p95_delta_pct": self.itl_p95_delta_pct,
            "tpot_p95_delta_pct": self.tpot_p95_delta_pct,
            "e2e_p95_delta_pct": self.e2e_p95_delta_pct,
            "error_rate_delta": self.error_rate_delta,
            "cost_ratio": self.cost_ratio,
            "goodput_req_s_ratio": self.goodput_req_s_ratio,
        }


def slo_from_mapping(data: Mapping[str, Any] | None) -> SLO | None:
    """Build an :class:`SLO` from a config mapping, or ``None`` if nothing is asserted.

    Accepts the three keys the schema uses and ignores the rest, so a scenario can pass its
    whole config block without filtering it first. A key present but ``None`` means "do not
    assert this objective", which is how a CLI with three optional flags behaves.
    """
    if not data:
        return None
    values: dict[str, float] = {}
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if value <= 0:
            raise ValueError(f"SLO {name} must be positive, got {value}")
        values[name] = float(value)
    if not values:
        return None
    return SLO(**values)


# ---------------------------------------------------------------------------------------
# How a run describes itself
#
# A scenario runs the same workload under several configurations -- cache off and on, four
# adapter counts, three engines -- and writes one result file per configuration. Those files
# have to say which arm they are, and which arm the others are measured against, or the
# renderer is left guessing from filenames. The convention is three optional keys in
# ``RunResult.config``, read only through the helpers below so that a change of key name is
# a one-line change rather than a search across the scenarios.
# ---------------------------------------------------------------------------------------

#: ``config["label"]``: the arm's human name, e.g. ``"continuous batching"``.
LABEL_KEY = "label"

#: ``config["backend"]``: which engine served it, e.g. ``"reference"`` or ``"vllm"``.
BACKEND_KEY = "backend"

#: ``config["baseline_label"]``: the label of the arm the others are compared against.
BASELINE_KEY = "baseline_label"

#: ``config["compare_to"]``: further arms this run also wants to be measured against, as a
#: label or a list of them. A scenario declares it when the comparison a reader came for is
#: between two *non-baseline* arms -- continuous batching against a padded static batch, say,
#: where the baseline every arm shares is sequential decoding -- so that the ratio is
#: rendered rather than left to the reader to divide out of two other ratios.
COMPARE_KEY = "compare_to"

#: ``summary["derived"]``: scenario-specific figures (cache hit rate, acceptance rate,
#: adapter memory) that the generic summariser cannot compute, added after ``finish()``.
DERIVED_KEY = "derived"


def run_label(result: RunResult) -> str:
    """The arm's name: its explicit label, else its backend, else scenario/profile."""
    config = result.config
    for key in (LABEL_KEY, BACKEND_KEY):
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    return f"{result.scenario}/{result.profile}"


def run_summary(result: RunResult) -> dict[str, Any]:
    """The run's summary, computing it if the run was never finished."""
    if result.summary:
        return result.summary
    return result.summarize()


def baseline_label(results: Sequence[RunResult]) -> str | None:
    """Which arm the others should be compared against, or ``None`` for an empty set.

    A run states it by putting the baseline's label in its own ``config["baseline_label"]``
    -- every arm of a scenario carries the same value, so the answer survives one arm's
    file being deleted. With no declaration, the first run is used, which makes the
    comparison table appear rather than vanish; it is still labelled with what it compared
    against, so a reader is never misled about the reference point.
    """
    if not results:
        return None
    for result in results:
        declared = result.config.get(BASELINE_KEY)
        if isinstance(declared, str) and declared:
            return declared
    return run_label(results[0])


def compare_runs(results: Sequence[RunResult], *, baseline: str | None = None) -> list[Comparison]:
    """Compare every arm against the baseline arm, in the order the runs were given.

    The baseline compared against itself is omitted: a row of 1.00x and 0% carries no
    information and invites a reader to mistake it for a measurement.
    """
    if not results:
        return []
    reference = baseline or baseline_label(results) or run_label(results[0])
    by_label = {run_label(result): result for result in results}
    anchor = by_label.get(reference)
    if anchor is None:
        anchor = results[0]
        reference = run_label(anchor)
    anchor_summary = run_summary(anchor)
    comparisons: list[Comparison] = []
    for result in results:
        label = run_label(result)
        if label == reference:
            continue
        comparisons.append(
            Comparison.from_summaries(reference, label, anchor_summary, run_summary(result))
        )
    return comparisons


def compare_to_labels(result: RunResult) -> list[str]:
    """The extra arms ``result`` declares in ``config["compare_to"]``.

    Accepts a single label or a list of them, and ignores anything that is not a non-empty
    string, so a hand-written config cannot turn a render into a traceback.
    """
    value = result.config.get(COMPARE_KEY)
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str) and item]
    return []


def comparison_groups(results: Sequence[RunResult]) -> list[tuple[str, list[RunResult]]]:
    """The relative tables a set of comparable runs asks for: one per reference arm.

    A scenario does not always have *one* control. ``spec_decode`` sweeps several
    target/draft pairs in a single run and each pair is only meaningful against its own
    target-only arm; ``multi_lora`` measured on two engines has one base-only arm per
    engine. Comparing every arm against whichever control happened to be written first
    would produce ratios between things that were never alternatives to each other, so the
    runs are partitioned by the baseline each of them *declares*, and a run additionally
    naming arms in ``config["compare_to"]`` joins those arms' tables as well.

    Returns ``(reference label, runs to measure against it)`` pairs in the order the
    references were first declared, skipping any reference that is not itself among
    ``results`` -- a table against an arm nobody measured would be a row of em dashes.
    With no declaration anywhere the whole set is compared against
    :func:`baseline_label`'s answer, which is what a one-off run written by hand gets.
    """
    by_label = {run_label(result): result for result in results}
    order: list[str] = []
    members: dict[str, list[RunResult]] = {}

    def want(result: RunResult, reference: Any) -> None:
        if not isinstance(reference, str) or not reference:
            return
        if reference not in by_label or reference == run_label(result):
            return
        if reference not in members:
            order.append(reference)
            members[reference] = []
        members[reference].append(result)

    # Declared baselines first, so a scenario's own control heads its section and the extra
    # comparisons an arm asked for follow it.
    for result in results:
        want(result, result.config.get(BASELINE_KEY))
    for result in results:
        for reference in compare_to_labels(result):
            want(result, reference)
    if order:
        return [(reference, members[reference]) for reference in order]
    fallback = baseline_label(results)
    if fallback is None or fallback not in by_label:
        return []
    rest = [result for result in results if run_label(result) != fallback]
    return [(fallback, rest)] if rest else []
