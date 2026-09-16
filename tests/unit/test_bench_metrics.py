"""Metric tests: how an observed stream becomes a record, and records become comparisons."""

from __future__ import annotations

import pytest

from turboserve.bench.metrics import (
    SLO,
    Comparison,
    RecordBuilder,
    baseline_label,
    compare_runs,
    group_by,
    pct_delta,
    ratio,
    run_label,
    run_summary,
    saved_pct,
    slo_from_mapping,
    summarize_by,
    summarize_records,
)
from turboserve.bench.records import RequestRecord, RunResult
from turboserve.engine.core.types import FinishReason
from turboserve.gateway.backends.protocol import TokenEvent

MS = 1_000_000


def build_stream_record(
    *,
    chunks: list[tuple[int, int]],
    send_ns: int = 0,
    usage: dict[str, int] | None = None,
) -> RequestRecord:
    """Feed ``(tokens, t_ns)`` chunks through a builder and return the record.

    Timestamps are supplied explicitly so the assertions are about the arithmetic and not
    about how fast the test machine happens to be.
    """
    builder = RecordBuilder("r1", tenant="t", prompt_tokens=5, backend="mock", lane="canary")
    builder.sent(send_ns)
    for index, (tokens, t_ns) in enumerate(chunks):
        last = index == len(chunks) - 1
        event = (
            TokenEvent.final(
                "r1", FinishReason.LENGTH, token_ids=list(range(tokens)), usage=usage, t_ns=t_ns
            )
            if last
            else TokenEvent.delta("r1", list(range(tokens)), t_ns=t_ns)
        )
        builder.observe(event, t_ns=t_ns)
    return builder.build()


# -- RecordBuilder ------------------------------------------------------------------------


def test_single_token_chunks_give_ttft_and_one_gap_per_token_after_the_first() -> None:
    record = build_stream_record(chunks=[(1, 10 * MS), (1, 15 * MS), (1, 21 * MS), (1, 30 * MS)])
    assert record.ok
    assert record.output_tokens == 4
    assert record.ttft_ms == pytest.approx(10.0)
    assert record.itl_ms == pytest.approx([5.0, 6.0, 9.0])
    assert len(record.itl_ns) == record.output_tokens - 1
    assert record.e2e_ms == pytest.approx(30.0)
    assert record.tpot_ms == pytest.approx((30.0 - 10.0) / 3)
    assert record.tenant == "t"
    assert record.backend == "mock"
    assert record.lane == "canary"


def test_a_multi_token_chunk_splits_its_interval_evenly() -> None:
    """Speculative decoding accepts several tokens per step; the gap belongs to all of them."""
    record = build_stream_record(chunks=[(1, 10 * MS), (4, 30 * MS)])
    assert record.output_tokens == 5
    assert record.itl_ms == pytest.approx([5.0, 5.0, 5.0, 5.0])
    assert sum(record.itl_ns) == pytest.approx(20 * MS, rel=1e-9)


def test_tokens_in_the_first_chunk_contribute_no_gap() -> None:
    """Nothing was observed to elapse for them, so no zero is invented for the series."""
    record = build_stream_record(chunks=[(3, 10 * MS), (1, 14 * MS)])
    assert record.output_tokens == 4
    assert record.itl_ms == pytest.approx([4.0])
    assert record.ttft_ms == pytest.approx(10.0)


def test_a_usage_block_overrides_the_counted_tokens() -> None:
    record = build_stream_record(
        chunks=[(1, 5 * MS), (1, 9 * MS)],
        usage={"prompt_tokens": 128, "completion_tokens": 64},
    )
    assert record.prompt_tokens == 128
    assert record.output_tokens == 64


def test_text_only_chunks_are_counted_one_token_each() -> None:
    builder = RecordBuilder("r1")
    builder.sent(0)
    builder.observe(TokenEvent(request_id="r1", text="hel"), t_ns=2 * MS)
    builder.observe(
        TokenEvent(request_id="r1", text="lo", finished=True, finish_reason=FinishReason.STOP),
        t_ns=4 * MS,
    )
    record = builder.build()
    assert record.output_tokens == 2
    assert record.itl_ms == pytest.approx([2.0])


def test_failure_marks_the_record_and_still_stamps_the_end() -> None:
    builder = RecordBuilder("r1")
    builder.sent(0)
    builder.observe(TokenEvent.delta("r1", [1], t_ns=3 * MS), t_ns=3 * MS)
    builder.failed("backend went away", t_ns=8 * MS)
    record = builder.build()
    assert not record.ok
    assert record.error == "backend went away"
    assert record.e2e_ms == pytest.approx(8.0)
    assert builder.started


def test_build_is_repeatable() -> None:
    builder = RecordBuilder("r1")
    builder.sent(0)
    builder.observe(TokenEvent.final("r1", FinishReason.STOP, token_ids=[1], t_ns=MS), t_ns=MS)
    assert builder.build().to_dict() == builder.build().to_dict()
    assert builder.request_id == "r1"


# -- aggregation --------------------------------------------------------------------------


def make_records(count: int, *, tenant: str = "a", ttft_ms: float = 10.0) -> list[RequestRecord]:
    """Records with known latencies, for the aggregation assertions."""
    return [
        RequestRecord(
            request_id=f"{tenant}-{index}",
            tenant=tenant,
            prompt_tokens=10,
            output_tokens=5,
            t_send_ns=0,
            t_first_ns=int(ttft_ms * MS),
            t_last_ns=int((ttft_ms + 40) * MS),
            itl_ns=[10 * MS] * 4,
            backend="reference",
        )
        for index in range(count)
    ]


def test_summarize_records_matches_the_run_summariser() -> None:
    records = make_records(6)
    direct = summarize_records(records)
    run = RunResult(scenario="s", profile="p", hardware={}, requests=list(records))
    assert direct == run.summarize()
    assert direct["num_requests"] == 6
    assert direct["ttft_ms"]["p50"] == pytest.approx(10.0)


def test_summarize_records_applies_an_slo_and_a_price() -> None:
    records = make_records(4, ttft_ms=10.0) + make_records(4, tenant="b", ttft_ms=100.0)
    summary = summarize_records(records, slo=SLO(ttft_ms=50.0), gpu_price_per_hour=2.49)
    assert summary["goodput"]["num_met"] == 4
    assert summary["goodput"]["ratio"] == pytest.approx(0.5)
    assert summary["cost_per_1m_output_tokens_usd"] is not None


def test_group_by_partitions_in_first_seen_order() -> None:
    records = make_records(2, tenant="b") + make_records(3, tenant="a")
    grouped = group_by(records, "tenant")
    assert list(grouped) == ["b", "a"]
    assert [len(value) for value in grouped.values()] == [2, 3]
    assert list(group_by(records, lambda record: record.backend)) == ["reference"]


def test_summarize_by_produces_one_summary_per_group() -> None:
    records = make_records(2, tenant="a") + make_records(3, tenant="b")
    summaries = summarize_by(records, "tenant")
    assert set(summaries) == {"a", "b"}
    assert summaries["b"]["num_requests"] == 3


# -- ratios and comparisons ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "baseline", "expected"),
    [(6.0, 2.0, 3.0), (1.0, 0.0, None), (None, 2.0, None), (2.0, None, None)],
)
def test_ratio_returns_none_rather_than_infinity(
    candidate: float | None, baseline: float | None, expected: float | None
) -> None:
    assert ratio(candidate, baseline) == expected


def test_pct_delta_and_saved_pct_signs() -> None:
    assert pct_delta(80.0, 100.0) == pytest.approx(-20.0)
    assert pct_delta(130.0, 100.0) == pytest.approx(30.0)
    assert pct_delta(1.0, 0.0) is None
    assert saved_pct(100.0, 10.0) == pytest.approx(90.0)
    assert saved_pct(0.0, 10.0) is None


def test_comparison_reads_ratios_and_deltas_from_two_summaries() -> None:
    baseline = summarize_records(make_records(4, ttft_ms=100.0))
    candidate = summarize_records(make_records(4, ttft_ms=50.0))
    comparison = Comparison.from_summaries("naive", "continuous", baseline, candidate)
    assert comparison.baseline == "naive"
    assert comparison.candidate == "continuous"
    assert comparison.ttft_p50_delta_pct == pytest.approx(-50.0)
    assert comparison.ttft_p95_delta_pct == pytest.approx(-50.0)
    assert comparison.error_rate_delta == pytest.approx(0.0)
    assert comparison.cost_ratio is None  # neither run recorded a GPU price
    assert set(comparison.to_dict()) >= {"baseline", "candidate", "output_tok_s_ratio"}


def make_run(label: str, *, baseline: str | None = None, ttft_ms: float = 10.0) -> RunResult:
    """A finished run carrying the label convention the report renderer reads."""
    config: dict[str, object] = {"label": label}
    if baseline is not None:
        config["baseline_label"] = baseline
    run = RunResult(
        scenario="naive_vs_cb",
        profile="dev-2060",
        hardware={},
        config=config,
        requests=make_records(4, ttft_ms=ttft_ms),
    )
    run.finish()
    return run


def test_run_label_prefers_the_explicit_label_then_the_backend() -> None:
    assert run_label(make_run("continuous batching")) == "continuous batching"
    backend_only = RunResult(scenario="s", profile="p", hardware={}, config={"backend": "vllm"})
    assert run_label(backend_only) == "vllm"
    assert run_label(RunResult(scenario="s", profile="p", hardware={})) == "s/p"


def test_run_summary_computes_when_the_run_was_never_finished() -> None:
    run = RunResult(scenario="s", profile="p", hardware={}, requests=make_records(2))
    assert not run.summary
    assert run_summary(run)["num_requests"] == 2


def test_baseline_label_uses_the_declaration_then_the_first_run() -> None:
    runs = [make_run("cb", baseline="naive"), make_run("naive", baseline="naive")]
    assert baseline_label(runs) == "naive"
    assert baseline_label([make_run("only")]) == "only"
    assert baseline_label([]) is None


def test_compare_runs_omits_the_baseline_row() -> None:
    runs = [
        make_run("naive", baseline="naive", ttft_ms=100.0),
        make_run("cb", baseline="naive", ttft_ms=50.0),
    ]
    comparisons = compare_runs(runs)
    assert [c.candidate for c in comparisons] == ["cb"]
    assert comparisons[0].baseline == "naive"
    assert comparisons[0].ttft_p50_delta_pct == pytest.approx(-50.0)


def test_compare_runs_falls_back_when_the_declared_baseline_is_absent() -> None:
    runs = [make_run("cb", baseline="missing"), make_run("vllm", baseline="missing")]
    comparisons = compare_runs(runs)
    assert [c.baseline for c in comparisons] == ["cb"]
    assert compare_runs([]) == []


# -- SLO parsing --------------------------------------------------------------------------


def test_slo_from_mapping_ignores_unset_and_unknown_keys() -> None:
    slo = slo_from_mapping({"ttft_ms": 250.0, "tpot_ms": None, "unrelated": "x"})
    assert slo == SLO(ttft_ms=250.0)
    assert slo_from_mapping({}) is None
    assert slo_from_mapping(None) is None
    assert slo_from_mapping({"e2e_ms": None}) is None


def test_slo_from_mapping_rejects_a_non_positive_objective() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        slo_from_mapping({"ttft_ms": 0.0})
