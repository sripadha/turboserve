"""Fault specifications: parsing, validation, and the timeline they expand to.

Pure data, so these tests need no clock, no sockets and no event loop. What they pin down is
the part of a chaos run that has to be reviewable before it is run: that a typo is rejected
rather than ignored, that a schedule string round-trips, and that the same schedule and seed
always produce the same victims at the same moments.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from turboserve.chaos.faults import (
    DEFAULT_GRACE_S,
    DEFAULT_RESTART_DELAY_S,
    Fault,
    FaultAction,
    FaultError,
    FaultSchedule,
    outage_windows,
    parse_duration,
    parse_milliseconds,
    parse_probability,
)

WORKERS = ("replica-0", "replica-1", "replica-2")


# -- scalar parsing ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("10s", 10.0), ("500ms", 0.5), ("2m", 120.0), ("1h", 3600.0), ("1.5", 1.5), ("0", 0.0)],
)
def test_durations_accept_the_units_the_grammar_documents(text: str, expected: float) -> None:
    assert parse_duration(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "   ", "abc", "10x", "-5s", "s"])
def test_unreadable_durations_are_rejected(text: str) -> None:
    with pytest.raises(FaultError):
        parse_duration(text)


@pytest.mark.parametrize(("text", "expected"), [("0.05", 0.05), ("5%", 0.05), ("1", 1.0)])
def test_probabilities_accept_fractions_and_percentages(text: str, expected: float) -> None:
    assert parse_probability(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["1.5", "-0.1", "", "half", "150%"])
def test_probabilities_outside_the_unit_interval_are_rejected(text: str) -> None:
    with pytest.raises(FaultError):
        parse_probability(text)


def test_a_bare_latency_is_milliseconds_but_a_united_one_is_converted() -> None:
    assert parse_milliseconds("500") == pytest.approx(500.0)
    assert parse_milliseconds("500ms") == pytest.approx(500.0)
    assert parse_milliseconds("1.5s") == pytest.approx(1500.0)


# -- fault parsing ----------------------------------------------------------------------


def test_kill_defaults_to_an_ungraceful_death_with_a_restart() -> None:
    fault = Fault.parse("kill:every=10s")
    assert fault.kind == "kill"
    assert fault.every_s == pytest.approx(10.0)
    assert fault.grace_s == DEFAULT_GRACE_S
    assert fault.restart_delay_s == DEFAULT_RESTART_DELAY_S
    assert fault.first_at_s == pytest.approx(10.0)
    assert fault.target is None


def test_kill_accepts_a_drain_a_restart_delay_a_start_offset_and_a_victim() -> None:
    fault = Fault.parse("kill:every=10s,grace=5s,restart=3s,start=1s,target=replica-1")
    assert (fault.grace_s, fault.restart_delay_s, fault.first_at_s) == (5.0, 3.0, 1.0)
    assert fault.target == "replica-1"


def test_latency_and_error_carry_a_probability() -> None:
    latency = Fault.parse("latency:p=0.05,ms=500")
    assert (latency.probability, latency.latency_ms) == (0.05, 500.0)
    assert Fault.parse("error:p=0.01").probability == pytest.approx(0.01)


def test_partition_is_a_window() -> None:
    fault = Fault.parse("partition:at=20s,for=5s")
    assert (fault.at_s, fault.for_s) == (20.0, 5.0)


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "explode:p=1",
        "kill",
        "kill:every=0s",
        "kill:every",
        "kill:every=10s,every=5s",
        "kill:evry=10s",
        "latency:p=0.05",
        "latency:ms=500",
        "latency:p=0,ms=500",
        "error:p=0",
        "error:p=0.1,ms=5",
        "partition:at=20s",
        "partition:for=5s",
        "partition:at=20s,for=0s",
    ],
)
def test_incomplete_or_misspelled_specifications_are_rejected(spec: str) -> None:
    with pytest.raises(FaultError):
        Fault.parse(spec)


def test_a_described_fault_reparses_to_an_equal_one() -> None:
    # `raw` is cleared so describe() has to render the fields rather than echo the input.
    original = replace(Fault.parse("kill:every=10s,grace=500ms,restart=1s"), raw="")
    rebuilt = Fault.parse(original.describe())
    assert (rebuilt.every_s, rebuilt.grace_s, rebuilt.restart_delay_s) == (10.0, 0.5, 1.0)
    assert replace(rebuilt, raw="") == original


def test_a_fault_serialises_the_fields_its_kind_actually_uses() -> None:
    data = Fault.parse("latency:p=0.05,ms=500").to_dict()
    assert data["kind"] == "latency"
    assert data["probability"] == pytest.approx(0.05)
    assert data["latency_ms"] == pytest.approx(500.0)
    assert "every_s" not in data


# -- schedules --------------------------------------------------------------------------


def test_a_schedule_separates_steady_probabilities_from_timed_events() -> None:
    schedule = FaultSchedule.from_string("kill:every=10s; latency:p=0.05,ms=500; error:p=0.01")
    steady = schedule.steady()
    assert steady.latency_probability == pytest.approx(0.05)
    assert steady.latency_ms == pytest.approx(500.0)
    assert steady.error_probability == pytest.approx(0.01)
    assert not steady.is_empty
    assert len(schedule) == 3
    assert not schedule.is_empty


def test_an_empty_schedule_has_nothing_steady_and_no_events() -> None:
    schedule = FaultSchedule()
    assert schedule.is_empty
    assert schedule.steady().is_empty
    assert schedule.events(duration_s=10.0, workers=WORKERS) == ()


def test_two_faults_of_the_same_probabilistic_kind_are_refused() -> None:
    with pytest.raises(FaultError, match="at most one"):
        FaultSchedule.from_string("error:p=0.01; error:p=0.02")


def test_kills_expand_into_drain_kill_restart_and_spread_over_the_fleet() -> None:
    schedule = FaultSchedule.parse(["kill:every=10s,grace=2s,restart=3s"])
    events = schedule.events(duration_s=25.0, workers=WORKERS, seed=0)
    assert [event.action for event in events] == [
        FaultAction.DRAIN,
        FaultAction.KILL,
        FaultAction.RESTART,
        FaultAction.DRAIN,
        FaultAction.KILL,
        FaultAction.RESTART,
    ]
    assert [event.t_s for event in events] == [10.0, 12.0, 15.0, 20.0, 22.0, 25.0]
    victims = [event.target for event in events if event.action is FaultAction.KILL]
    assert len(set(victims)) == 2, "consecutive kills must not fall on the same replica"


def test_an_ungraceful_kill_has_no_drain_phase() -> None:
    events = FaultSchedule.parse(["kill:every=10s"]).events(duration_s=15.0, workers=WORKERS)
    assert [event.action for event in events] == [FaultAction.KILL, FaultAction.RESTART]


def test_a_targeted_kill_always_hits_the_same_replica() -> None:
    schedule = FaultSchedule.parse(["kill:every=5s,target=replica-2"])
    events = schedule.events(duration_s=16.0, workers=WORKERS, seed=1)
    assert {event.target for event in events} == {"replica-2"}


def test_a_kill_aimed_at_an_unknown_replica_is_an_error() -> None:
    schedule = FaultSchedule.parse(["kill:every=5s,target=replica-9"])
    with pytest.raises(FaultError, match="unknown worker"):
        schedule.events(duration_s=10.0, workers=WORKERS)


def test_partitions_expand_into_a_start_and_an_end() -> None:
    events = FaultSchedule.parse(["partition:at=20s,for=5s"]).events(
        duration_s=30.0, workers=WORKERS, seed=3
    )
    assert [event.action for event in events] == [
        FaultAction.PARTITION_START,
        FaultAction.PARTITION_END,
    ]
    assert [event.t_s for event in events] == [20.0, 25.0]
    assert events[0].target == events[1].target


def test_a_partition_scheduled_after_the_run_ends_is_dropped() -> None:
    events = FaultSchedule.parse(["partition:at=90s,for=5s"]).events(
        duration_s=30.0, workers=WORKERS
    )
    assert events == ()


def test_the_same_seed_gives_the_same_timeline_and_a_different_one_may_not() -> None:
    schedule = FaultSchedule.parse(["kill:every=7s"])
    first = schedule.events(duration_s=60.0, workers=WORKERS, seed=1234)
    again = schedule.events(duration_s=60.0, workers=WORKERS, seed=1234)
    assert first == again
    offsets = {
        schedule.events(duration_s=60.0, workers=WORKERS, seed=seed)[0].target for seed in range(12)
    }
    assert len(offsets) > 1, "the victim offset must depend on the seed"


def test_recoveries_are_ordered_before_disruptions_at_the_same_instant() -> None:
    schedule = FaultSchedule.parse(
        ["kill:every=10s,restart=5s,target=replica-0", "kill:every=15s,target=replica-1"]
    )
    events = schedule.events(duration_s=16.0, workers=WORKERS)
    at_fifteen = [event for event in events if event.t_s == 15.0]
    assert [event.action for event in at_fifteen] == [FaultAction.RESTART, FaultAction.KILL]


def test_a_schedule_reports_when_it_leaves_no_replica_serving() -> None:
    schedule = FaultSchedule.parse(
        [
            "kill:every=10s,restart=20s,target=replica-0",
            "kill:every=12s,restart=20s,target=replica-1",
        ]
    )
    events = schedule.events(duration_s=20.0, workers=("replica-0", "replica-1"))
    assert outage_windows(events, ("replica-0", "replica-1"), duration_s=20.0) == ((12.0, 20.0),)


def test_a_schedule_that_keeps_one_replica_alive_reports_no_outage() -> None:
    schedule = FaultSchedule.parse(["kill:every=5s,restart=1s"])
    events = schedule.events(duration_s=20.0, workers=WORKERS, seed=2)
    assert outage_windows(events, WORKERS, duration_s=20.0) == ()


def test_a_schedule_serialises_everything_needed_to_rerun_it() -> None:
    schedule = FaultSchedule.from_string("kill:every=10s; error:p=0.01")
    data = schedule.to_dict()
    assert data["spec"] == "kill:every=10s; error:p=0.01"
    assert [fault["kind"] for fault in data["faults"]] == ["kill", "error"]
    assert data["steady"]["error_probability"] == pytest.approx(0.01)
    assert FaultSchedule.from_string(data["spec"]).describe() == schedule.describe()


def test_expanding_a_schedule_needs_a_positive_duration_and_a_fleet() -> None:
    schedule = FaultSchedule.parse(["kill:every=1s"])
    with pytest.raises(FaultError):
        schedule.events(duration_s=0.0, workers=WORKERS)
    with pytest.raises(FaultError):
        schedule.events(duration_s=10.0, workers=())
