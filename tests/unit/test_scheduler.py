"""Unit tests for :mod:`turboserve.engine.core.scheduler`.

The scheduler is checked as a state machine rather than function by function: most tests
drive a workload to completion through a small step loop that stands in for the runtime,
asserting invariants after every step. The properties that matter are the ones a serving
engine cannot violate even once -- the step budgets, the one-entry-per-sequence rule, the
monotonicity of progress outside preemption, and, under ``tenant_fair``, a bounded wait for
every tenant.
"""

from __future__ import annotations

import random
from itertools import groupby

import pytest

from turboserve.engine.core.scheduler import ScheduledSeq, Scheduler, SchedulerOutput
from turboserve.engine.core.sequence import SeqStatus, Sequence
from turboserve.engine.core.types import FinishReason, SamplingParams, SchedulerConfig

NEXT_TOKEN = 77
EOS = 2


def config(**kwargs: object) -> SchedulerConfig:
    base: dict[str, object] = {
        "num_blocks": 64,
        "block_size": 4,
        "max_num_batched_tokens": 16,
        "max_num_seqs": 4,
    }
    base.update(kwargs)
    return SchedulerConfig(**base)  # type: ignore[arg-type]


def run_step(
    scheduler: Scheduler,
    output: SchedulerOutput,
    *,
    eos_token_id: int | None = None,
) -> None:
    """Stand in for the runtime: feed one sampled token back per sampling sequence."""
    for item in output.sampled():
        scheduler.append_token(item.seq, NEXT_TOKEN, eos_token_id=eos_token_id)


def drive(scheduler: Scheduler, *, max_steps: int = 500) -> list[SchedulerOutput]:
    """Run until everything finishes, checking invariants after every step."""
    steps: list[SchedulerOutput] = []
    for _ in range(max_steps):
        if not scheduler.has_unfinished():
            return steps
        out = scheduler.schedule()
        scheduler.check_invariants()
        steps.append(out)
        run_step(scheduler, out)
    raise AssertionError("workload did not finish within the step limit")


# -- budgets ----------------------------------------------------------------------------------


def test_step_budgets_are_never_exceeded() -> None:
    rng = random.Random(20260916)
    scheduler = Scheduler(config(max_num_batched_tokens=13, max_num_seqs=3))
    for index in range(12):
        scheduler.add_request(
            f"r{index}",
            [rng.randrange(1000) for _ in range(rng.randint(1, 25))],
            SamplingParams(max_tokens=rng.randint(1, 6)),
        )
    for out in drive(scheduler):
        assert out.num_batched_tokens <= 13
        assert out.num_seqs <= 3
        assert len({item.seq_id for item in out.scheduled}) == out.num_seqs
    assert scheduler.stats()["num_finished"] == 12


def test_chunked_prefill_splits_a_long_prompt_at_the_budget() -> None:
    scheduler = Scheduler(config(max_num_batched_tokens=8, max_num_seqs=4, num_blocks=32))
    seq = scheduler.add_request("long", list(range(20)), SamplingParams(max_tokens=1))
    first = scheduler.schedule()
    assert first.num_prefill_tokens == 8
    assert first.scheduled[0].is_chunk is True
    assert first.scheduled[0].samples_token is False
    assert not first.sampled()
    run_step(scheduler, first)
    assert seq.num_computed_tokens == 8

    second = scheduler.schedule()
    assert second.num_prefill_tokens == 8
    assert second.scheduled[0].is_chunk is True
    run_step(scheduler, second)

    third = scheduler.schedule()
    assert third.num_prefill_tokens == 4
    assert third.scheduled[0].is_prefill is True
    assert third.scheduled[0].is_chunk is False
    assert third.scheduled[0].samples_token is True
    run_step(scheduler, third)
    assert seq.is_finished and seq.finish_reason is FinishReason.LENGTH


def test_disabled_chunked_prefill_schedules_whole_prompts_only() -> None:
    scheduler = Scheduler(
        config(max_num_batched_tokens=16, max_num_seqs=4, enable_chunked_prefill=False)
    )
    scheduler.add_request("a", list(range(10)), SamplingParams(max_tokens=2))
    scheduler.add_request("b", list(range(100, 110)), SamplingParams(max_tokens=2))
    first = scheduler.schedule()
    assert [item.num_new_tokens for item in first.scheduled] == [10]
    assert scheduler.num_waiting == 1
    run_step(scheduler, first)
    second = scheduler.schedule()
    assert sorted(item.num_new_tokens for item in second.scheduled) == [1, 10]


def test_a_prompt_that_cannot_ever_be_scheduled_is_rejected_at_arrival() -> None:
    scheduler = Scheduler(config(max_num_batched_tokens=16, enable_chunked_prefill=False))
    with pytest.raises(ValueError, match="chunked prefill is disabled"):
        scheduler.add_request("huge", list(range(17)))
    small_pool = Scheduler(config(num_blocks=2, block_size=4, max_num_batched_tokens=64))
    with pytest.raises(ValueError, match="KV blocks but the pool holds"):
        small_pool.add_request("huge", list(range(40)))
    scheduler.add_request("ok", [1, 2, 3])
    with pytest.raises(ValueError, match="already in flight"):
        scheduler.add_request("ok", [1, 2, 3])


# -- batch layout -----------------------------------------------------------------------------


def test_prefills_come_before_decodes_and_metadata_matches_the_batch() -> None:
    scheduler = Scheduler(config(max_num_batched_tokens=32, max_num_seqs=8, num_blocks=64))
    first = scheduler.add_request("a", list(range(6)), SamplingParams(max_tokens=4))
    run_step(scheduler, scheduler.schedule())
    scheduler.add_request("b", list(range(100, 109)), SamplingParams(max_tokens=4))

    out = scheduler.schedule()
    assert [item.request_id for item in out.scheduled] == ["b", "a"]
    assert out.num_prefill_seqs == 1
    assert out.num_decode_seqs == 1
    assert out.num_prefill_tokens == 9
    assert out.num_decode_tokens == 1
    assert out.num_seqs == len(out) == 2
    assert not out.is_empty

    meta = out.build_attn_metadata(4)
    meta.validate(block_size=4)
    assert meta.query_lens() == [9, 1]
    assert meta.context_lens.tolist() == [9, 7]
    assert meta.num_tokens == 10
    assert out.input_token_ids() == [*range(100, 109), NEXT_TOKEN]
    assert out.positions() == [*range(9), 6]
    assert out.sample_indices() == [8, 9]
    assert [item.seq for item in out.sampled()] == [out.scheduled[0].seq, first]
    assert out.token_lora_ids() == [0] * 10
    assert out.build_lora_context().is_base_only


def test_metadata_rejects_an_empty_step_and_a_short_block_table() -> None:
    scheduler = Scheduler(config())
    empty = scheduler.schedule()
    assert empty.is_empty
    with pytest.raises(ValueError, match="empty step"):
        empty.build_attn_metadata(4)

    scheduler.add_request("a", list(range(6)))
    out = scheduler.schedule()
    out.scheduled[0].block_table = out.scheduled[0].block_table[:1]
    with pytest.raises(ValueError, match="context tokens"):
        out.build_attn_metadata(4)
    with pytest.raises(ValueError, match="block_size must be positive"):
        out.build_attn_metadata(0)


def test_scheduled_seq_validates_its_own_shape() -> None:
    seq = Sequence(seq_id=0, request_id="x", prompt_token_ids=[1, 2])
    with pytest.raises(ValueError, match="at least one token"):
        ScheduledSeq(seq, 0, 0, [], [], [], False, False)
    with pytest.raises(ValueError, match="token ids for"):
        ScheduledSeq(seq, 2, 2, [1], [0, 1], [0], True, False)
    with pytest.raises(ValueError, match="slots for"):
        ScheduledSeq(seq, 2, 2, [1, 2], [0], [0], True, False)
    with pytest.raises(ValueError, match="shorter than"):
        ScheduledSeq(seq, 2, 1, [1, 2], [0, 1], [0], True, False)


# -- determinism ------------------------------------------------------------------------------


def test_scheduling_is_deterministic_for_the_same_queue_state() -> None:
    def trace() -> list[tuple[tuple[str, int], ...]]:
        scheduler = Scheduler(config(num_blocks=12, max_num_batched_tokens=11, max_num_seqs=3))
        for index in range(8):
            scheduler.add_request(
                f"r{index}",
                list(range(index, index + 9)),
                SamplingParams(max_tokens=3),
                arrival=float(index),
            )
        return [
            tuple((item.request_id, item.num_new_tokens) for item in out.scheduled)
            for out in drive(scheduler)
        ]

    assert trace() == trace()


# -- preemption -------------------------------------------------------------------------------


def test_block_exhaustion_preempts_and_every_request_still_finishes() -> None:
    scheduler = Scheduler(
        config(num_blocks=6, block_size=4, max_num_batched_tokens=64, max_num_seqs=8)
    )
    sequences = [
        scheduler.add_request(
            f"r{index}", list(range(index * 50, index * 50 + 8)), SamplingParams(max_tokens=10)
        )
        for index in range(3)
    ]
    progress = {seq.request_id: 0 for seq in sequences}
    preempted_any = False
    for _ in range(500):
        if not scheduler.has_unfinished():
            break
        out = scheduler.schedule()
        scheduler.check_invariants()
        preempted_ids = {seq.request_id for seq in out.preempted}
        preempted_any = preempted_any or bool(preempted_ids)
        for seq in out.preempted:
            assert seq.status is SeqStatus.PREEMPTED
            assert seq.block_table == []
            progress[seq.request_id] = 0
        for item in out.scheduled:
            if item.request_id in preempted_ids:
                continue
            assert item.seq.num_computed_tokens >= progress[item.request_id]
            progress[item.request_id] = item.seq.num_computed_tokens
        run_step(scheduler, out)
    assert preempted_any, "the pool was too large to force a preemption"
    assert all(seq.is_finished for seq in sequences)
    assert all(seq.num_output_tokens == 10 for seq in sequences)
    assert scheduler.stats()["num_preemptions"] >= 1
    pool = scheduler.block_manager
    assert pool.num_free_blocks + pool.allocator.num_retained == 6


def test_the_victim_is_the_lowest_priority_most_recent_sequence() -> None:
    scheduler = Scheduler(
        config(num_blocks=4, block_size=4, max_num_batched_tokens=64, max_num_seqs=8)
    )
    high = scheduler.add_request("high", list(range(4)), SamplingParams(max_tokens=8), priority=5)
    low = scheduler.add_request("low", list(range(100, 104)), SamplingParams(max_tokens=8))
    scheduler.add_request("late", list(range(200, 204)), SamplingParams(max_tokens=8))
    run_step(scheduler, scheduler.schedule())
    assert {seq.request_id for seq in scheduler.running()} == {"high", "low", "late"}
    victims: list[str] = []
    for _ in range(6):
        out = scheduler.schedule()
        victims.extend(seq.request_id for seq in out.preempted)
        run_step(scheduler, out)
        if victims:
            break
    assert victims, "no preemption happened"
    assert victims[0] == "late"
    assert high.status is not SeqStatus.PREEMPTED
    assert low.num_preemptions == 0


def test_a_resumed_sequence_reuses_its_cached_prefix() -> None:
    scheduler = Scheduler(
        config(num_blocks=4, block_size=4, max_num_batched_tokens=64, max_num_seqs=4)
    )
    seq = scheduler.add_request("a", list(range(8)), SamplingParams(max_tokens=4))
    run_step(scheduler, scheduler.schedule())
    scheduler.block_manager.publish_computed_blocks(seq)
    scheduler.block_manager.preempt(seq)
    scheduler._running.remove(seq)  # noqa: SLF001 - emulating a preemption from outside
    scheduler._preempted.append(seq)  # noqa: SLF001
    out = scheduler.schedule()
    assert out.num_cached_tokens == 8
    assert out.scheduled[0].num_new_tokens == 1
    assert out.scheduled[0].num_cached_tokens == 8


# -- prefix caching ---------------------------------------------------------------------------


def test_a_shared_prefix_shortens_the_second_request_prefill() -> None:
    scheduler = Scheduler(config(num_blocks=32, max_num_batched_tokens=64, max_num_seqs=4))
    prompt = list(range(12))
    scheduler.add_request("first", prompt, SamplingParams(max_tokens=1))
    run_step(scheduler, scheduler.schedule())
    scheduler.add_request("second", prompt, SamplingParams(max_tokens=1))
    out = scheduler.schedule()
    assert out.num_cached_tokens == 8
    assert out.num_prefill_tokens == 4
    assert out.scheduled[0].context_len == 12
    meta = out.build_attn_metadata(4)
    meta.validate(block_size=4)
    assert meta.context_lens.tolist() == [12]


def test_prefix_caching_can_be_switched_off() -> None:
    scheduler = Scheduler(config(num_blocks=32, enable_prefix_caching=False))
    assert scheduler.block_manager.prefix_cache is None
    prompt = list(range(8))
    scheduler.add_request("first", prompt, SamplingParams(max_tokens=1))
    run_step(scheduler, scheduler.schedule())
    scheduler.add_request("second", prompt, SamplingParams(max_tokens=1))
    out = scheduler.schedule()
    assert out.num_cached_tokens == 0
    assert out.num_prefill_tokens == 8


# -- lifecycle --------------------------------------------------------------------------------


def test_stop_token_and_eos_retire_a_sequence_and_release_its_blocks() -> None:
    scheduler = Scheduler(config(num_blocks=16, enable_prefix_caching=False))
    seq = scheduler.add_request("a", [1, 2, 3], SamplingParams(max_tokens=99))
    out = scheduler.schedule()
    scheduler.append_token(seq, EOS, eos_token_id=EOS)
    assert seq.is_finished and seq.finish_reason is FinishReason.STOP
    assert seq.stop_token_id == EOS
    assert seq.block_table == []
    assert scheduler.num_running == 0
    assert not scheduler.has_unfinished()
    assert scheduler.block_manager.num_free_blocks == 16
    assert out.scheduled[0].samples_token


def test_abort_works_in_every_state() -> None:
    scheduler = Scheduler(config(num_blocks=16, max_num_seqs=1, max_num_batched_tokens=8))
    waiting = scheduler.add_request("waiting", [1, 2, 3])
    running = scheduler.add_request("running", [4, 5, 6])
    scheduler._running.append(running)  # noqa: SLF001 - place it directly in the running set
    scheduler._waiting.remove(running)  # noqa: SLF001
    running.status = SeqStatus.RUNNING
    assert scheduler.abort("waiting") is True
    assert waiting.finish_reason is FinishReason.ABORT
    assert scheduler.abort("running") is True
    assert scheduler.abort("running") is False
    assert scheduler.num_unfinished == 0
    assert len(scheduler) == 0


def test_reset_drops_everything_and_returns_the_pool() -> None:
    scheduler = Scheduler(config(num_blocks=16))
    scheduler.add_request("a", list(range(8)))
    scheduler.schedule()
    scheduler.reset()
    assert scheduler.num_unfinished == 0
    assert scheduler.num_running == 0
    assert scheduler.block_manager.num_free_blocks == 16
    scheduler.check_invariants()
    assert repr(scheduler).startswith("Scheduler(")


def test_timings_are_stamped_once_and_are_injectable() -> None:
    scheduler = Scheduler(config())
    seq = scheduler.add_request("a", [1, 2, 3], SamplingParams(max_tokens=1), arrival=10.0)
    assert seq.timing.t_arrival == 10.0
    out = scheduler.schedule(now=11.0)
    assert seq.timing.t_first_scheduled == 11.0
    scheduler.schedule(now=12.0)
    assert seq.timing.t_first_scheduled == 11.0
    scheduler.append_token(out.scheduled[0].seq, NEXT_TOKEN, now=13.0)
    assert seq.timing.queue() == pytest.approx(1.0)
    assert seq.timing.ttft() == pytest.approx(3.0)


def test_construction_requires_a_sized_pool() -> None:
    with pytest.raises(ValueError, match="num_blocks must be set"):
        Scheduler(SchedulerConfig(block_size=4))
    from turboserve.engine.core.block_manager import BlockManager

    with pytest.raises(ValueError, match="disagrees with the scheduler"):
        Scheduler(config(block_size=4), BlockManager.create(8, 8))


# -- fairness ---------------------------------------------------------------------------------


def admission_order(scheduler: Scheduler, num_steps: int) -> list[str]:
    """Admit one request per step and record which tenant it belonged to."""
    order: list[str] = []
    for _ in range(num_steps):
        out = scheduler.schedule()
        if out.is_empty:
            break
        assert len(out.scheduled) == 1
        order.append(out.scheduled[0].tenant_id)
        run_step(scheduler, out)
    return order


def fair_scheduler(weights: dict[str, float]) -> Scheduler:
    return Scheduler(
        config(
            num_blocks=64,
            max_num_seqs=1,
            max_num_batched_tokens=16,
            policy="tenant_fair",
            tenant_weights=weights,
        )
    )


def test_tenant_fair_splits_admissions_by_weight() -> None:
    scheduler = fair_scheduler({"a": 1.0, "b": 3.0})
    for index in range(40):
        for tenant in ("a", "b"):
            scheduler.add_request(
                f"{tenant}{index}", [1, 2], SamplingParams(max_tokens=1), tenant_id=tenant
            )
    order = admission_order(scheduler, 40)
    assert len(order) == 40
    assert order.count("b") == 30
    assert order.count("a") == 10
    assert order[:8] == ["a", "b", "b", "b", "a", "b", "b", "b"]


def test_tenant_fair_bounds_how_long_a_tenant_waits() -> None:
    """The starvation bound.

    Weighted fair queueing gives a tenant of weight ``w`` a turn at least every
    ``sum(weights) / w`` admissions. Here that is four, plus at most one step of phase
    offset because the light tenant joins an already-running schedule and its first turn
    lands wherever the current minimum virtual time happens to be. The point of the test is
    that the wait is bounded by a small constant no matter how many requests the heavy
    tenant has queued -- sixty against ten here.
    """
    scheduler = fair_scheduler({"a": 1.0, "b": 3.0})
    for index in range(60):
        scheduler.add_request(f"b{index}", [1, 2], SamplingParams(max_tokens=1), tenant_id="b")
    for index in range(10):
        scheduler.add_request(f"a{index}", [3, 4], SamplingParams(max_tokens=1), tenant_id="a")
    order = admission_order(scheduler, 40)
    positions = [index for index, tenant in enumerate(order) if tenant == "a"]
    assert len(positions) == 10, "every light-tenant request should have been served"
    gaps = [b - a for a, b in zip(positions, positions[1:], strict=False)]
    assert max([positions[0] + 1, *gaps]) <= 5
    assert sum(gaps) / len(gaps) == pytest.approx(4.0, abs=0.5)


def test_a_returning_tenant_cannot_bank_credit_while_idle() -> None:
    scheduler = fair_scheduler({"a": 1.0, "b": 1.0})
    for index in range(6):
        scheduler.add_request(f"b{index}", [1, 2], SamplingParams(max_tokens=1), tenant_id="b")
    assert admission_order(scheduler, 4) == ["b"] * 4
    scheduler.add_request("a0", [3, 4], SamplingParams(max_tokens=1), tenant_id="a")
    scheduler.add_request("a1", [3, 4], SamplingParams(max_tokens=1), tenant_id="a")
    # A newly active tenant starts at the current minimum virtual time: it is served
    # promptly, but it does not get to claim the four turns it "missed".
    assert admission_order(scheduler, 4) == ["a", "b", "a", "b"]


def test_a_tenant_that_was_served_then_went_idle_cannot_bank_credit() -> None:
    """The returning-tenant branch, which the test above does not reach.

    Above, ``a`` has never been served, so it is absent from the virtual-time map and any
    floor at all works. Here ``a`` is served once, goes idle while ``b`` runs up a large
    virtual time, and comes back: its stored virtual time is stale and small, so unless the
    floor is taken from the *other* backlogged tenants it would win every comparison until
    it caught up -- ten consecutive admissions, which is the starvation the docstring
    promises cannot happen.
    """
    scheduler = fair_scheduler({"a": 1.0, "b": 1.0})
    scheduler.add_request("a-first", [1, 2], SamplingParams(max_tokens=1), tenant_id="a")
    assert admission_order(scheduler, 1) == ["a"]
    for index in range(50):
        scheduler.add_request(f"b{index}", [1, 2], SamplingParams(max_tokens=1), tenant_id="b")
    assert admission_order(scheduler, 50) == ["b"] * 50
    for index in range(10):
        for tenant in ("a", "b"):
            scheduler.add_request(
                f"{tenant}-late{index}", [3, 4], SamplingParams(max_tokens=1), tenant_id=tenant
            )
    order = admission_order(scheduler, 20)
    assert order.count("a") == 10
    assert max(len(list(run)) for _, run in groupby(order)) <= 2
    assert order[:4] == ["a", "b", "a", "b"]


def test_fcfs_admits_strictly_in_arrival_order() -> None:
    scheduler = Scheduler(config(num_blocks=64, max_num_seqs=1, max_num_batched_tokens=16))
    for index in range(6):
        tenant = "a" if index % 2 else "b"
        scheduler.add_request(f"r{index}", [1, 2], SamplingParams(max_tokens=1), tenant_id=tenant)
    order = [out.scheduled[0].request_id for out in (scheduler.schedule() for _ in range(6))]
    assert order == [f"r{index}" for index in range(6)]
