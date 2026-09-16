"""Usage accounting: token counts, the three latencies, prices and per-tenant totals."""

from __future__ import annotations

import pytest

from turboserve.engine.core.types import FinishReason
from turboserve.gateway.backends.protocol import TokenEvent
from turboserve.gateway.metrics import GatewayMetrics
from turboserve.gateway.usage import (
    ModelPrice,
    PriceTable,
    UsageAccumulator,
    UsageRecord,
    UsageTracker,
    estimate_text_tokens,
)


def accumulator(**overrides: object) -> UsageAccumulator:
    """An accumulator whose arrival time is zero, so every timing reads as an offset."""
    data: dict[str, object] = {
        "request_id": "req-1",
        "tenant_id": "acme",
        "model": "m",
        "prompt_tokens": 10,
        "arrival_ts": 0.0,
    }
    data.update(overrides)
    return UsageAccumulator(
        str(data.pop("request_id")),
        str(data.pop("tenant_id")),
        str(data.pop("model")),
        **data,  # type: ignore[arg-type]
    )


# -- timings ------------------------------------------------------------------------------


def test_the_three_latencies_follow_their_definitions() -> None:
    acc = accumulator()
    acc.on_event(TokenEvent.delta("req-1", [1]), now=0.2)
    acc.on_event(TokenEvent.delta("req-1", [2]), now=0.3)
    acc.on_event(TokenEvent.final("req-1", FinishReason.STOP, token_ids=[3]), now=0.5)
    record = acc.finish()
    assert record.ttft_s == pytest.approx(0.2)  # arrival -> first output token
    assert record.e2e_s == pytest.approx(0.5)  # arrival -> finish
    assert record.tpot_s == pytest.approx(0.15)  # (0.5 - 0.2) / (3 - 1)


def test_ttft_is_measured_from_arrival_not_from_routing() -> None:
    # The gateway's own auth, quota and routing time belongs inside TTFT: it is what the
    # client experiences, and excluding it would flatter the engine.
    acc = accumulator(arrival_ts=0.0)
    acc.on_event(TokenEvent.delta("req-1", [1]), now=1.0)
    assert acc.ttft_s == pytest.approx(1.0)


def test_tpot_is_none_below_two_output_tokens() -> None:
    acc = accumulator()
    acc.on_event(TokenEvent.final("req-1", FinishReason.STOP, token_ids=[1]), now=0.4)
    record = acc.finish()
    assert record.tpot_s is None
    assert record.ttft_s == pytest.approx(0.4)


def test_an_unstarted_request_has_no_latencies() -> None:
    acc = accumulator()
    record = acc.finish(status="error", now=0.9)
    assert record.ttft_s is None
    assert record.tpot_s is None
    assert record.e2e_s == pytest.approx(0.9)
    assert record.ok is False


# -- token counting -----------------------------------------------------------------------


def test_token_ids_are_counted_exactly() -> None:
    acc = accumulator()
    acc.on_event(TokenEvent.delta("req-1", [1, 2, 3]), now=0.1)
    acc.on_event(TokenEvent.final("req-1", FinishReason.LENGTH, token_ids=[4]), now=0.2)
    assert acc.completion_tokens == 4


def test_a_text_only_stream_counts_one_token_per_delta() -> None:
    # An HTTP backend that reports no usage streams text, not ids; one delta is one token.
    acc = accumulator()
    acc.on_event(TokenEvent.delta("req-1", [], "Hello"), now=0.1)
    acc.on_event(TokenEvent.delta("req-1", [], " world"), now=0.2)
    acc.on_event(TokenEvent.final("req-1", FinishReason.STOP), now=0.3)
    assert acc.completion_tokens == 2


def test_the_backends_own_usage_wins() -> None:
    acc = accumulator(prompt_tokens=10, estimated=True)
    acc.on_event(TokenEvent.delta("req-1", [], "a"), now=0.1)
    acc.on_event(
        TokenEvent.final(
            "req-1",
            FinishReason.STOP,
            usage={"prompt_tokens": 37, "completion_tokens": 5, "cached_prompt_tokens": 12},
        ),
        now=0.2,
    )
    record = acc.finish()
    assert record.prompt_tokens == 37
    assert record.completion_tokens == 5
    assert record.cached_prompt_tokens == 12
    # An authoritative count replaces the estimate, so the record is no longer flagged.
    assert record.estimated is False


def test_an_estimated_prompt_is_flagged() -> None:
    acc = accumulator(estimated=True)
    acc.on_event(TokenEvent.final("req-1", FinishReason.STOP, token_ids=[1]), now=0.1)
    assert acc.finish().estimated is True


def test_estimate_text_tokens_is_never_zero() -> None:
    assert estimate_text_tokens("") == 1
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("a" * 400) == 100


# -- the OpenAI usage object ---------------------------------------------------------------


def test_openai_usage_totals_and_omits_empty_cache_details() -> None:
    record = UsageRecord(
        request_id="r", tenant_id="acme", model="m", prompt_tokens=7, completion_tokens=3
    )
    assert record.total_tokens == 10
    assert record.to_openai_usage() == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_openai_usage_reports_prefix_cache_hits_where_openai_does() -> None:
    record = UsageRecord(
        request_id="r",
        tenant_id="acme",
        model="m",
        prompt_tokens=100,
        completion_tokens=3,
        cached_prompt_tokens=64,
    )
    assert record.to_openai_usage()["prompt_tokens_details"] == {"cached_tokens": 64}


def test_record_round_trips_through_a_plain_mapping() -> None:
    record = UsageRecord(request_id="r", tenant_id="acme", model="m", completion_tokens=2)
    data = record.to_dict()
    assert data["request_id"] == "r"
    assert data["completion_tokens"] == 2
    assert set(data) >= {"ttft_s", "tpot_s", "e2e_s", "status", "cost_usd", "estimated"}


# -- prices --------------------------------------------------------------------------------


def test_price_is_per_million_tokens() -> None:
    price = ModelPrice(input_per_1m_usd=2.0, output_per_1m_usd=6.0)
    assert price.cost_usd(1_000_000, 0) == pytest.approx(2.0)
    assert price.cost_usd(0, 500_000) == pytest.approx(3.0)
    assert price.cost_usd(1_000, 1_000) == pytest.approx(0.008)


def test_an_unpriced_model_costs_none_not_zero() -> None:
    # "unpriced" and "free" must not look the same on a bill.
    table = PriceTable.from_mapping({"m": {"input_per_1m_usd": 1.0, "output_per_1m_usd": 2.0}})
    assert table.cost_usd("m", 1_000_000, 0) == pytest.approx(1.0)
    assert table.cost_usd("other", 1_000_000, 0) is None
    assert "m" in table and len(table) == 1
    assert table.models() == ["m"]


def test_price_defaults_label_themselves_as_configured() -> None:
    assert ModelPrice(input_per_1m_usd=0.0, output_per_1m_usd=0.0).source == "configured"


def test_negative_prices_are_refused() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ModelPrice(input_per_1m_usd=-1.0, output_per_1m_usd=1.0)


# -- tracker -------------------------------------------------------------------------------


def test_tracker_prices_publishes_and_totals_one_record() -> None:
    metrics = GatewayMetrics()
    tracker = UsageTracker(
        prices=PriceTable.from_mapping({"m": {"input_per_1m_usd": 1.0, "output_per_1m_usd": 10.0}}),
        metrics=metrics,
    )
    acc = tracker.start(
        request_id="r1",
        tenant_id="acme",
        model="m",
        prompt_tokens=1000,
        arrival_ts=0.0,
        backend="b",
        lane="canary",
    )
    acc.on_event(TokenEvent.delta("r1", [1]), now=0.1)
    acc.on_event(TokenEvent.final("r1", FinishReason.STOP, token_ids=[2]), now=0.3)
    record = tracker.complete(acc)

    assert record.cost_usd == pytest.approx(0.001 + 0.00002)
    labels = {"tenant": "acme", "model": "m", "backend": "b", "lane": "canary"}
    assert metrics.sample_value("turboserve_gateway_requests_total", **labels, status="ok") == 1.0
    assert (
        metrics.sample_value("turboserve_gateway_tokens_total", **labels, kind="prompt") == 1000.0
    )
    assert metrics.sample_value("turboserve_gateway_e2e_seconds_count", **labels) == 1.0
    assert metrics.sample_value("turboserve_gateway_cost_usd_total", tenant="acme", model="m") == (
        pytest.approx(record.cost_usd)
    )


def test_tracker_totals_accumulate_per_tenant() -> None:
    tracker = UsageTracker()
    for index, status in enumerate(("ok", "ok", "error")):
        tracker.record(
            UsageRecord(
                request_id=f"r{index}",
                tenant_id="acme",
                model="m",
                prompt_tokens=10,
                completion_tokens=5,
                status=status,
            )
        )
    totals = tracker.totals_for("acme")
    assert totals.requests == 3
    assert totals.failed == 1
    assert totals.total_tokens == 45
    assert totals.to_dict()["prompt_tokens"] == 30
    assert tracker.totals_for("nobody").requests == 0


def test_tracker_reset_drops_totals() -> None:
    tracker = UsageTracker()
    tracker.record(UsageRecord(request_id="r", tenant_id="acme", model="m"))
    assert set(tracker.totals()) == {"acme"}
    tracker.reset()
    assert tracker.totals() == {}


def test_an_already_priced_record_is_not_repriced() -> None:
    tracker = UsageTracker(
        prices=PriceTable.from_mapping({"m": {"input_per_1m_usd": 1.0, "output_per_1m_usd": 1.0}})
    )
    record = tracker.record(
        UsageRecord(request_id="r", tenant_id="acme", model="m", prompt_tokens=10, cost_usd=0.5)
    )
    assert record.cost_usd == 0.5
