"""Unit tests for the benchmark result schema.

The percentile expectations below are hand-computed from the linear-interpolation
definition in ``bench/records.py``; the records are synthetic timestamps, not measurements
of anything, and no number here describes the performance of any system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from turboserve.bench.records import (
    SCHEMA_VERSION,
    SLO,
    Percentiles,
    RequestRecord,
    RunResult,
    default_index_path,
    percentile,
    software_versions,
    utc_now_iso,
)

MS = 1_000_000
S = 1_000_000_000


# --------------------------------------------------------------------------------------
# percentile
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        (0.0, 1.0),
        (50.0, 2.5),  # rank 0.50 * 3 = 1.5 -> 2 + (3 - 2) * 0.5
        (90.0, 3.7),  # rank 0.90 * 3 = 2.7 -> 3 + (4 - 3) * 0.7
        (95.0, 3.85),  # rank 2.85
        (99.0, 3.97),  # rank 2.97
        (100.0, 4.0),
    ],
)
def test_percentile_linear_interpolation_on_four_samples(q: float, expected: float) -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], q) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("q", "expected"),
    [(50.0, 30.0), (90.0, 46.0), (95.0, 48.0), (99.0, 49.6)],
)
def test_percentile_on_five_samples(q: float, expected: float) -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], q) == pytest.approx(expected)


def test_percentile_of_a_single_sample_is_that_sample() -> None:
    assert percentile([7.0], 99.0) == 7.0


def test_percentile_of_an_empty_sample_is_none() -> None:
    assert percentile([], 50.0) is None


@pytest.mark.parametrize("q", [-1.0, 100.1])
def test_percentile_rejects_out_of_range_quantiles(q: float) -> None:
    with pytest.raises(ValueError, match=r"q must be in \[0, 100\]"):
        percentile([1.0], q)


# --------------------------------------------------------------------------------------
# Percentiles
# --------------------------------------------------------------------------------------


def test_percentiles_from_values() -> None:
    summary = Percentiles.from_values([1.0, 2.0, 3.0, 4.0])
    assert summary.count == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.p50 == pytest.approx(2.5)
    assert summary.p95 == pytest.approx(3.85)
    assert summary.min == 1.0
    assert summary.max == 4.0
    assert not summary.is_empty


def test_percentiles_of_nothing_is_empty_not_zero() -> None:
    summary = Percentiles.from_values([])
    assert summary.is_empty
    assert summary.count == 0
    assert summary.p50 is None
    assert summary.mean is None
    assert summary.to_dict()["p99"] is None


def test_percentiles_drops_non_finite_samples() -> None:
    summary = Percentiles.from_values([1.0, float("nan"), float("inf"), 3.0])
    assert summary.count == 2
    assert summary.mean == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# RequestRecord
# --------------------------------------------------------------------------------------


def _record(**overrides: Any) -> RequestRecord:
    payload: dict[str, Any] = {
        "request_id": "r1",
        "tenant": "acme",
        "prompt_tokens": 5,
        "output_tokens": 11,
        "t_send_ns": 0,
        "t_first_ns": 100 * MS,
        "t_last_ns": 1 * S,
        "itl_ns": [90 * MS] * 10,
        "backend": "reference",
    }
    payload.update(overrides)
    return RequestRecord(**payload)


def test_record_latency_properties() -> None:
    record = _record()
    assert record.ttft_ms == pytest.approx(100.0)
    assert record.e2e_ms == pytest.approx(1000.0)
    # (1.0 s - 0.1 s) spread over 10 gaps.
    assert record.tpot_ms == pytest.approx(90.0)
    assert record.itl_ms == [pytest.approx(90.0)] * 10
    assert record.total_tokens == 16


def test_record_without_tokens_has_no_latencies() -> None:
    record = _record(t_first_ns=None, t_last_ns=None, output_tokens=0, itl_ns=[], ok=False)
    assert record.ttft_ms is None
    assert record.e2e_ms is None
    assert record.tpot_ms is None


def test_record_tpot_needs_two_output_tokens() -> None:
    assert _record(output_tokens=1).tpot_ms is None


def test_record_tpot_is_none_when_every_token_arrived_at_once() -> None:
    # What a blocking baseline produces: one event carrying the whole completion, so the
    # client observed no interval between tokens. Zero would read as an instant decode.
    record = _record(t_last_ns=_record().t_first_ns, itl_ns=[])
    assert record.tpot_ms is None
    assert record.e2e_ms == record.ttft_ms


def test_record_defaults_are_a_successful_stable_lane_request() -> None:
    record = RequestRecord(request_id="r")
    assert record.ok
    assert record.lane == "stable"
    assert record.error is None
    assert record.itl_ns == []


def test_record_round_trips_through_a_dict() -> None:
    record = _record()
    assert RequestRecord.from_dict(record.to_dict()) == record


def test_record_from_dict_ignores_unknown_fields() -> None:
    data = _record().to_dict()
    data["some_future_field"] = 1
    assert RequestRecord.from_dict(data).request_id == "r1"


# --------------------------------------------------------------------------------------
# SLO
# --------------------------------------------------------------------------------------


def test_slo_asserts_only_the_fields_it_sets() -> None:
    record = _record()  # ttft 100 ms, tpot 90 ms, e2e 1000 ms
    assert SLO().is_met_by(record)
    assert SLO(ttft_ms=150.0).is_met_by(record)
    assert not SLO(ttft_ms=50.0).is_met_by(record)
    assert not SLO(tpot_ms=10.0).is_met_by(record)
    assert not SLO(e2e_ms=500.0).is_met_by(record)


def test_slo_is_never_met_by_a_failed_request() -> None:
    assert not SLO().is_met_by(_record(ok=False, error="reset"))


def test_slo_is_not_met_when_the_metric_is_missing() -> None:
    record = _record(output_tokens=1)  # no tpot
    assert not SLO(tpot_ms=1000.0).is_met_by(record)


# --------------------------------------------------------------------------------------
# RunResult aggregation
# --------------------------------------------------------------------------------------


def _run() -> RunResult:
    """Two synthetic requests over a 1.5 s window; see the assertions for the arithmetic."""
    run = RunResult(scenario="naive_vs_cb", profile="dev-2060")
    run.add(_record())
    run.add(
        _record(
            request_id="r2",
            output_tokens=6,
            t_send_ns=500 * MS,
            t_first_ns=700 * MS,
            t_last_ns=1500 * MS,
            itl_ns=[160 * MS] * 5,
        )
    )
    return run


def test_wall_seconds_spans_first_send_to_last_token() -> None:
    assert _run().wall_seconds() == pytest.approx(1.5)


def test_summarize_counts_and_rates() -> None:
    summary = _run().summarize()
    assert summary["num_requests"] == 2
    assert summary["num_ok"] == 2
    assert summary["num_failed"] == 0
    assert summary["error_rate"] == 0.0
    assert summary["output_tokens"] == 17
    assert summary["prompt_tokens"] == 10
    assert summary["output_tok_s"] == pytest.approx(17 / 1.5)
    assert summary["total_tok_s"] == pytest.approx(27 / 1.5)
    assert summary["req_s"] == pytest.approx(2 / 1.5)


def test_summarize_latency_distributions() -> None:
    summary = _run().summarize()
    assert summary["ttft_ms"]["count"] == 2
    assert summary["ttft_ms"]["p50"] == pytest.approx(150.0)  # midpoint of 100 and 200
    assert summary["e2e_ms"]["p50"] == pytest.approx(1000.0)
    assert summary["tpot_ms"]["p50"] == pytest.approx(125.0)  # midpoint of 90 and 160
    assert summary["itl_ms"]["count"] == 15  # 10 gaps + 5 gaps


def test_summarize_excludes_failed_requests_from_latency_but_not_from_error_rate() -> None:
    run = _run()
    run.add(_record(request_id="r3", ok=False, error="reset", t_first_ns=None, t_last_ns=None))
    summary = run.summarize()
    assert summary["num_requests"] == 3
    assert summary["error_rate"] == pytest.approx(1 / 3)
    assert summary["ttft_ms"]["count"] == 2


def test_summarize_of_an_empty_run_is_all_zeros_and_nones() -> None:
    summary = RunResult(scenario="s", profile="p").summarize()
    assert summary["num_requests"] == 0
    assert summary["error_rate"] == 0.0
    assert summary["wall_s"] == 0.0
    assert summary["output_tok_s"] == 0.0
    assert summary["ttft_ms"]["p95"] is None


def test_summarize_stores_the_summary_on_the_run() -> None:
    run = _run()
    summary = run.summarize()
    assert run.summary == summary


def test_goodput_counts_only_requests_inside_the_slo() -> None:
    run = _run()
    result = run.goodput(SLO(ttft_ms=150.0))  # r1 (100 ms) passes, r2 (200 ms) does not
    assert result["num_met"] == 1
    assert result["ratio"] == pytest.approx(0.5)
    assert result["req_s"] == pytest.approx(1 / 1.5)
    assert result["slo"] == {"ttft_ms": 150.0, "tpot_ms": None, "e2e_ms": None}


def test_summary_carries_goodput_only_when_an_slo_is_given() -> None:
    assert _run().summarize()["goodput"] is None
    assert _run().summarize(slo=SLO(e2e_ms=10_000.0))["goodput"]["num_met"] == 2


def test_cost_per_million_tokens_needs_a_recorded_price() -> None:
    run = _run()
    assert run.summarize()["cost_per_1m_output_tokens_usd"] is None
    run.gpu_price_per_hour = 3.6  # $0.001 per second
    # 1.5 s of GPU time over 17 output tokens.
    expected = 3.6 * (1.5 / 3600.0) / 17 * 1_000_000
    assert run.summarize()["cost_per_1m_output_tokens_usd"] == pytest.approx(expected)


# --------------------------------------------------------------------------------------
# RunResult provenance and serialisation
# --------------------------------------------------------------------------------------


def test_provenance_defaults_to_measured() -> None:
    assert RunResult(scenario="s", profile="p").provenance == "measured"


def test_projected_provenance_is_allowed() -> None:
    run = RunResult(scenario="s", profile="p", provenance="projected", provenance_note="note")
    assert run.provenance == "projected"
    assert run.provenance_note == "note"


def test_unknown_provenance_is_rejected() -> None:
    with pytest.raises(ValueError, match="provenance must be"):
        RunResult(scenario="s", profile="p", provenance="guessed")  # type: ignore[arg-type]


def test_to_dict_puts_the_schema_version_first() -> None:
    data = _run().to_dict()
    assert next(iter(data)) == "schema_version"
    assert data["schema_version"] == SCHEMA_VERSION


def test_json_round_trip_preserves_every_field() -> None:
    run = _run()
    run.config = {"concurrency": 4}
    run.hardware = {"gpu_name": "synthetic", "git": {"sha": "abc123"}}
    run.software = {"torch": "2.6.0"}
    run.git_sha = "abc123"
    run.started_at = utc_now_iso()
    run.gpu_price_per_hour = 2.49
    run.price_source = "synthetic fixture"
    run.finish(slo=SLO(ttft_ms=1000.0))

    restored = RunResult.from_json(run.to_json())
    assert restored.scenario == run.scenario
    assert restored.profile == run.profile
    assert restored.config == run.config
    assert restored.hardware == run.hardware
    assert restored.software == run.software
    assert restored.git_sha == "abc123"
    assert restored.gpu_price_per_hour == 2.49
    assert restored.price_source == "synthetic fixture"
    assert restored.started_at == run.started_at
    assert restored.finished_at == run.finished_at
    assert restored.summary == run.summary
    assert restored.requests == run.requests


def test_finish_stamps_a_finished_timestamp_and_summarises() -> None:
    run = _run()
    assert run.finished_at is None
    summary = run.finish()
    assert run.finished_at is not None
    assert summary["num_requests"] == 2


def test_from_dict_rejects_an_unreadable_schema_version() -> None:
    data = _run().to_dict()
    data["schema_version"] = "99"
    with pytest.raises(ValueError, match="cannot be read by this build"):
        RunResult.from_dict(data)


def test_start_captures_hardware_software_and_git_sha() -> None:
    run = RunResult.start(
        "prefix_cache",
        "dev-2060",
        config={"shared_prefix_tokens": 128},
        hardware={"gpu_name": "synthetic", "git": {"sha": "deadbeef"}},
        provenance="projected",
        provenance_note="regenerate with make bench-h100",
        gpu_price_per_hour=2.49,
        price_source="synthetic fixture",
    )
    assert run.scenario == "prefix_cache"
    assert run.git_sha == "deadbeef"
    assert run.started_at is not None
    assert run.provenance == "projected"
    assert run.config == {"shared_prefix_tokens": 128}
    assert "torch" in run.software


def test_software_versions_reports_installed_and_missing_packages() -> None:
    versions = software_versions(["turboserve", "definitely-not-installed"])
    assert versions["turboserve"] is not None
    assert versions["definitely-not-installed"] is None


# --------------------------------------------------------------------------------------
# Saving and the results index
# --------------------------------------------------------------------------------------


def test_default_index_path_sits_at_the_results_root(tmp_path: Path) -> None:
    results = tmp_path / "results"
    target = results / "naive_vs_cb" / "20260101-000000.json"
    assert default_index_path(target) == results / "index.json"


def test_default_index_path_falls_back_to_the_sibling_directory(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "run.json"
    assert default_index_path(target) == (tmp_path / "elsewhere" / "index.json").resolve()


def test_save_writes_the_file_and_creates_the_index(tmp_path: Path) -> None:
    run = _run()
    run.finish()
    target = tmp_path / "results" / "naive_vs_cb" / "run-1.json"
    written = run.save(target)

    assert written == target
    reloaded = RunResult.load(target)
    assert reloaded.summary == run.summary

    index = json.loads((tmp_path / "results" / "index.json").read_text(encoding="utf-8"))
    assert len(index) == 1
    assert index[0]["path"] == "naive_vs_cb/run-1.json"
    assert index[0]["scenario"] == "naive_vs_cb"
    assert index[0]["profile"] == "dev-2060"
    assert index[0]["provenance"] == "measured"
    assert index[0]["num_requests"] == 2


def test_save_appends_to_an_existing_index(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for name in ("run-1.json", "run-2.json"):
        _run().save(results / "naive_vs_cb" / name)
    index = json.loads((results / "index.json").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in index] == [
        "naive_vs_cb/run-1.json",
        "naive_vs_cb/run-2.json",
    ]


def test_save_repairs_a_corrupt_index_instead_of_losing_the_run(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "index.json").write_text("{not json", encoding="utf-8")
    _run().save(results / "spec_decode" / "run-1.json")
    index = json.loads((results / "index.json").read_text(encoding="utf-8"))
    assert len(index) == 1


def test_save_can_skip_the_index(tmp_path: Path) -> None:
    target = tmp_path / "results" / "chaos" / "run-1.json"
    _run().save(target, update_index=False)
    assert target.exists()
    assert not (tmp_path / "results" / "index.json").exists()


def test_save_honours_an_explicit_index_path(tmp_path: Path) -> None:
    index = tmp_path / "custom-index.json"
    _run().save(tmp_path / "deep" / "run.json", index_path=index)
    entries = json.loads(index.read_text(encoding="utf-8"))
    assert entries[0]["path"] == "deep/run.json"


def test_save_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _run().save(results / "multi_lora" / "run-1.json")
    leftovers = [path.name for path in results.rglob("*.tmp")]
    assert leftovers == []
