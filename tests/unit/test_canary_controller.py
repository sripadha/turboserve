"""Unit tests for the SLO gate.

Every latency in this file is a synthetic fixture value chosen to sit on one side of a
threshold; nothing here is measured, and no number in it describes the behaviour of any
real deployment.

The controller is driven on an explicit clock: every ``now`` is passed in, so these tests
replay rollouts that would take hours of wall-clock in microseconds and are bit-for-bit
reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turboserve.canary.controller import (
    CanaryConfig,
    CanaryController,
    CanaryError,
    CanaryState,
    DecisionKind,
    Lane,
    LaneSummary,
    LaneWindow,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "canary.yaml"

# Synthetic latencies (milliseconds) used as gate inputs.
FAST_TTFT = 100.0
FAST_E2E = 500.0


def fast_config(**overrides: object) -> CanaryConfig:
    """A small, fast policy so a whole rollout fits in a handful of simulated seconds."""
    base: dict[str, object] = {
        "steps": (1, 5, 25, 50, 100),
        "step_hold_s": 10.0,
        "window_s": 600.0,
        "min_requests": 10,
        "max_error_rate": 0.005,
        "max_p95_ratio_vs_stable": 1.25,
        "stall_timeout_s": None,
    }
    base.update(overrides)
    return CanaryConfig(**base)  # type: ignore[arg-type]


def feed(
    controller: CanaryController,
    lane: Lane,
    count: int,
    *,
    now: float,
    ok: bool = True,
    ttft_ms: float = FAST_TTFT,
    e2e_ms: float = FAST_E2E,
) -> None:
    """Observe ``count`` identical requests on ``lane`` at instant ``now``."""
    for _ in range(count):
        controller.observe(lane, ok, ttft_ms, e2e_ms, now=now)


def healthy_step(controller: CanaryController, at: float, count: int = 10) -> None:
    """Fill both lanes with equally healthy traffic."""
    feed(controller, "stable", count, now=at)
    feed(controller, "canary", count, now=at)


# ---------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------


def test_default_config_steps_are_monotone_and_end_at_100() -> None:
    config = CanaryConfig()
    assert config.steps[-1] == 100
    assert list(config.steps) == sorted(set(config.steps))


@pytest.mark.parametrize(
    "steps",
    [(), (5, 5, 100), (50, 25, 100), (1, 5, 50), (0, 100), (1, 101)],
)
def test_invalid_step_ladders_are_rejected(steps: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        CanaryConfig(steps=steps)


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        CanaryConfig.model_validate({"steps": (100,), "max_error_rte": 0.1})


def test_repo_config_file_loads() -> None:
    config = CanaryConfig.from_yaml(CONFIG_PATH)
    assert config.steps[-1] == 100
    assert config.window_s > 0
    assert config.min_requests >= 1


def test_from_yaml_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "canary.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        CanaryConfig.from_yaml(path)


def test_from_yaml_accepts_a_bare_document(tmp_path: Path) -> None:
    path = tmp_path / "canary.yaml"
    path.write_text("steps: [10, 100]\nmin_requests: 3\n", encoding="utf-8")
    config = CanaryConfig.from_yaml(path)
    assert config.steps == (10, 100)
    assert config.min_requests == 3


# ---------------------------------------------------------------------------------------
# Windows and summaries
# ---------------------------------------------------------------------------------------


def test_window_expires_samples_outside_the_window() -> None:
    window = LaneWindow("canary", window_s=10.0)
    for t in (0.0, 5.0, 9.0):
        window.observe(True, ttft_ms=FAST_TTFT, e2e_ms=FAST_E2E, now=t)
    assert len(window) == 3
    summary = window.summary(now=12.0)
    assert summary.requests == 2  # the sample at t=0 fell out of the 10 s window
    assert summary.p95_ttft_ms == pytest.approx(FAST_TTFT)


def test_window_is_bounded_by_max_samples() -> None:
    window = LaneWindow("stable", window_s=1000.0, max_samples=4)
    for t in range(10):
        window.observe(True, now=float(t))
    assert len(window) == 4


def test_window_never_un_expires_an_out_of_order_sample() -> None:
    window = LaneWindow("canary", window_s=10.0)
    window.observe(True, now=100.0)
    window.observe(True, now=50.0)  # arrives late, already outside the window
    assert window.summary(now=100.0).requests == 1


def test_lane_summary_error_rate_and_validation() -> None:
    assert LaneSummary(lane="canary").error_rate == 0.0
    assert LaneSummary(lane="canary", requests=8, errors=2).error_rate == pytest.approx(0.25)
    with pytest.raises(ValueError):
        LaneSummary(lane="canary", requests=1, errors=2)
    with pytest.raises(ValueError):
        LaneSummary(lane="canary", requests=-1)


def test_failed_requests_count_as_errors_but_not_as_latency_samples() -> None:
    window = LaneWindow("canary", window_s=100.0)
    window.observe(True, ttft_ms=FAST_TTFT, e2e_ms=FAST_E2E, now=0.0)
    window.observe(False, ttft_ms=1.0, e2e_ms=1.0, now=1.0)
    summary = window.summary(now=2.0)
    assert summary.requests == 2
    assert summary.errors == 1
    # The failure's 1 ms must not drag the percentile down.
    assert summary.p95_ttft_ms == pytest.approx(FAST_TTFT)


# ---------------------------------------------------------------------------------------
# Clock injection
# ---------------------------------------------------------------------------------------


def test_injected_clock_is_used_when_now_is_omitted() -> None:
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    controller = CanaryController(fast_config(), clock=lambda: next(ticks))
    decision = controller.start("v2")
    assert decision.at == 0.0
    controller.observe("canary", True, FAST_TTFT, FAST_E2E)
    assert controller.window("canary").summary(now=1.0).requests == 1
    assert controller.tick().at == 2.0


# ---------------------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------------------


def test_weights_are_monotone_and_reach_promotion() -> None:
    config = fast_config()
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    assert controller.canary_weight == config.steps[0]
    assert controller.state is CanaryState.CANARY

    now = 0.0
    for _ in range(len(config.steps)):
        healthy_step(controller, at=now + 1.0)
        now += config.step_hold_s
        controller.tick(now)

    weights = [decision.weight for decision in controller.history]
    assert weights == sorted(weights), weights
    assert weights[-1] == 100
    assert controller.state is CanaryState.PROMOTED
    assert controller.canary_weight == 100
    assert controller.stable_weight == 0

    kinds = [decision.kind for decision in controller.history]
    assert kinds[-1] is DecisionKind.PROMOTE
    assert kinds.count(DecisionKind.ADVANCE) == len(config.steps)  # start + 4 advances
    assert controller.history[-1].is_terminal


def test_promotion_only_happens_from_the_final_step() -> None:
    config = fast_config(steps=(50, 100))
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)

    healthy_step(controller, at=1.0)
    first = controller.tick(config.step_hold_s)
    assert first.kind is DecisionKind.ADVANCE
    assert controller.canary_weight == 100
    assert controller.state is CanaryState.CANARY  # 100 % is served, but not yet promoted

    healthy_step(controller, at=config.step_hold_s + 1.0)
    second = controller.tick(2 * config.step_hold_s)
    assert second.kind is DecisionKind.PROMOTE
    assert controller.state is CanaryState.PROMOTED


def test_step_is_held_until_the_hold_time_elapses() -> None:
    config = fast_config()
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    healthy_step(controller, at=0.5)

    decision = controller.tick(config.step_hold_s - 0.001)
    assert decision.kind is DecisionKind.HOLD
    assert "held" in decision.reason
    assert controller.canary_weight == config.steps[0]


# ---------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------


def test_holds_while_min_requests_is_unmet_even_after_the_hold_time() -> None:
    config = fast_config(min_requests=10)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "stable", 50, now=1.0)
    feed(controller, "canary", 9, now=1.0)

    decision = controller.tick(10 * config.step_hold_s)
    assert decision.kind is DecisionKind.HOLD
    assert "9 of 10" in decision.reason
    assert controller.canary_weight == config.steps[0]
    assert controller.state is CanaryState.CANARY


def test_a_few_failures_below_min_requests_do_not_roll_back() -> None:
    """Two bad requests out of three are not evidence; the gate waits for a sample."""
    config = fast_config(min_requests=10)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "canary", 2, now=1.0, ok=False)
    feed(controller, "canary", 1, now=1.0)

    assert controller.tick(config.step_hold_s).kind is DecisionKind.HOLD
    assert controller.state is CanaryState.CANARY


def test_stall_timeout_rolls_back_a_lane_that_never_gets_traffic() -> None:
    config = fast_config(min_requests=10, stall_timeout_s=60.0)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "stable", 100, now=1.0)

    assert controller.tick(59.0).kind is DecisionKind.HOLD
    decision = controller.tick(60.0)
    assert decision.kind is DecisionKind.ROLLBACK
    assert "stalled" in decision.reason
    assert controller.canary_weight == 0


def test_rollback_on_error_rate_breach() -> None:
    config = fast_config(min_requests=10, max_error_rate=0.005)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "stable", 20, now=1.0)
    feed(controller, "canary", 19, now=1.0)
    feed(controller, "canary", 1, now=1.0, ok=False)

    decision = controller.tick(1.0)
    assert decision.kind is DecisionKind.ROLLBACK
    assert "error rate" in decision.reason
    assert controller.state is CanaryState.ROLLED_BACK
    assert controller.canary_weight == 0
    assert controller.stable_weight == 100


def test_error_rate_exactly_at_the_limit_does_not_roll_back() -> None:
    config = fast_config(min_requests=10, max_error_rate=0.1)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "canary", 9, now=1.0)
    feed(controller, "canary", 1, now=1.0, ok=False)

    assert controller.tick(1.0).kind is DecisionKind.HOLD
    assert controller.state is CanaryState.CANARY


def test_rollback_on_p95_ratio_breach_against_stable() -> None:
    config = fast_config(min_requests=10, max_p95_ratio_vs_stable=1.25)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "stable", 20, now=1.0, ttft_ms=100.0, e2e_ms=1000.0)
    feed(controller, "canary", 20, now=1.0, ttft_ms=200.0, e2e_ms=1000.0)

    decision = controller.tick(1.0)
    assert decision.kind is DecisionKind.ROLLBACK
    assert "TTFT" in decision.reason
    assert "2.00x" in decision.reason
    assert controller.state is CanaryState.ROLLED_BACK


def test_end_to_end_p95_ratio_is_gated_too() -> None:
    config = fast_config(min_requests=10, max_p95_ratio_vs_stable=1.25)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "stable", 20, now=1.0, ttft_ms=100.0, e2e_ms=1000.0)
    feed(controller, "canary", 20, now=1.0, ttft_ms=100.0, e2e_ms=2000.0)

    decision = controller.tick(1.0)
    assert decision.kind is DecisionKind.ROLLBACK
    assert "E2E" in decision.reason


def test_both_lanes_slowing_together_does_not_roll_back() -> None:
    """The ratio gate exists so a fleet-wide slowdown does not blame the canary."""
    config = fast_config(min_requests=10, max_p95_ratio_vs_stable=1.25)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "stable", 20, now=1.0, ttft_ms=900.0, e2e_ms=9000.0)
    feed(controller, "canary", 20, now=1.0, ttft_ms=950.0, e2e_ms=9100.0)

    assert controller.tick(1.0).kind is DecisionKind.HOLD
    assert controller.state is CanaryState.CANARY


def test_ratio_gate_waits_for_a_baseline() -> None:
    """With too little stable traffic there is nothing to compare against."""
    config = fast_config(min_requests=10, max_p95_ratio_vs_stable=1.25)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "stable", 3, now=1.0, ttft_ms=100.0, e2e_ms=1000.0)
    feed(controller, "canary", 20, now=1.0, ttft_ms=5000.0, e2e_ms=50000.0)

    assert controller.tick(1.0).kind is DecisionKind.HOLD


def test_absolute_ttft_gate_is_opt_in() -> None:
    lenient = fast_config(min_requests=10, max_p95_ttft_ms=None, max_p95_ratio_vs_stable=100.0)
    controller = CanaryController(lenient, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "canary", 20, now=1.0, ttft_ms=9999.0, e2e_ms=9999.0)
    assert controller.tick(1.0).kind is DecisionKind.HOLD

    strict = fast_config(min_requests=10, max_p95_ttft_ms=500.0, max_p95_ratio_vs_stable=100.0)
    gated = CanaryController(strict, clock=lambda: 0.0)
    gated.start("v2", now=0.0)
    feed(gated, "canary", 20, now=1.0, ttft_ms=9999.0, e2e_ms=9999.0)
    decision = gated.tick(1.0)
    assert decision.kind is DecisionKind.ROLLBACK
    assert "p95 TTFT" in decision.reason


def test_a_breach_at_a_later_step_rolls_all_the_way_back() -> None:
    config = fast_config(min_requests=10)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    healthy_step(controller, at=1.0, count=20)
    assert controller.tick(config.step_hold_s).kind is DecisionKind.ADVANCE
    assert controller.tick(2 * config.step_hold_s).kind is DecisionKind.ADVANCE
    assert controller.canary_weight == 25

    feed(controller, "canary", 20, now=2 * config.step_hold_s + 1.0, ok=False)
    decision = controller.tick(2 * config.step_hold_s + 2.0)
    assert decision.kind is DecisionKind.ROLLBACK
    assert controller.canary_weight == 0


# ---------------------------------------------------------------------------------------
# Overrides, lifecycle and serialisation
# ---------------------------------------------------------------------------------------


def test_tick_accepts_external_lane_summaries() -> None:
    """The Kubernetes path pushes Prometheus-derived summaries through the same gates."""
    config = fast_config(min_requests=10)
    controller = CanaryController(config, clock=lambda: 0.0)
    controller.start("v2", now=0.0)

    decision = controller.tick(
        1.0,
        stable=LaneSummary(lane="stable", requests=100, errors=0, p95_ttft_ms=100.0),
        canary=LaneSummary(lane="canary", requests=100, errors=50, p95_ttft_ms=100.0),
    )
    assert decision.kind is DecisionKind.ROLLBACK
    assert decision.canary is not None
    assert decision.canary.requests == 100


def test_tick_is_a_no_op_outside_a_rollout() -> None:
    controller = CanaryController(fast_config(), clock=lambda: 0.0)
    decision = controller.tick(1.0)
    assert decision.kind is DecisionKind.HOLD
    assert controller.history == ()
    assert controller.canary_weight == 0


def test_starting_twice_is_refused_and_restart_clears_the_windows() -> None:
    controller = CanaryController(fast_config(), clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    with pytest.raises(CanaryError, match="already running"):
        controller.start("v3", now=1.0)

    feed(controller, "canary", 5, now=1.0)
    controller.abort(now=2.0)
    controller.start("v3", now=3.0)
    assert controller.summary("canary", now=3.0).requests == 0
    assert controller.version == "v3"
    assert controller.canary_weight == controller.config.steps[0]


def test_start_requires_a_version() -> None:
    controller = CanaryController(fast_config(), clock=lambda: 0.0)
    with pytest.raises(ValueError):
        controller.start("", now=0.0)


def test_abort_outside_a_rollout_is_an_error() -> None:
    controller = CanaryController(fast_config(), clock=lambda: 0.0)
    with pytest.raises(CanaryError):
        controller.abort(now=0.0)


def test_abort_records_its_reason_and_zeroes_the_weight() -> None:
    controller = CanaryController(fast_config(), clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    decision = controller.abort("operator pulled the rollout", now=5.0)
    assert decision.kind is DecisionKind.ROLLBACK
    assert decision.reason == "operator pulled the rollout"
    assert controller.state is CanaryState.ROLLED_BACK
    assert controller.canary_weight == 0
    assert controller.is_terminal


def test_reset_returns_to_idle() -> None:
    controller = CanaryController(fast_config(), clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "canary", 5, now=1.0)
    controller.reset()
    assert controller.state is CanaryState.IDLE
    assert controller.history == ()
    assert controller.version is None
    assert controller.summary("canary", now=1.0).requests == 0


def test_lane_weights_always_sum_to_100() -> None:
    controller = CanaryController(fast_config(), clock=lambda: 0.0)
    assert controller.lane_weights() == {"stable": 100, "canary": 0}

    controller.start("v2", now=0.0)
    weights = controller.lane_weights()
    assert weights["stable"] + weights["canary"] == 100
    assert weights["canary"] == controller.config.steps[0]
    assert controller.canary_fraction == pytest.approx(weights["canary"] / 100.0)

    controller.abort(now=1.0)
    assert controller.lane_weights() == {"stable": 100, "canary": 0}


def test_unknown_lane_is_rejected() -> None:
    controller = CanaryController(fast_config(), clock=lambda: 0.0)
    with pytest.raises(ValueError):
        controller.observe("shadow", True, now=0.0)  # type: ignore[arg-type]


def test_decisions_and_snapshot_are_json_serialisable() -> None:
    controller = CanaryController(fast_config(min_requests=1), clock=lambda: 0.0)
    controller.start("v2", now=0.0)
    feed(controller, "canary", 2, now=1.0)
    controller.tick(2.0)
    payload = json.dumps(
        {
            "snapshot": controller.snapshot(),
            "history": [decision.to_dict() for decision in controller.history],
        }
    )
    restored = json.loads(payload)
    assert restored["snapshot"]["state"] == "canary"
    assert restored["history"][0]["kind"] == "advance"
    assert restored["history"][-1]["canary"]["error_rate"] == 0.0
