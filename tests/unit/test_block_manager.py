"""Unit tests for :mod:`turboserve.engine.core.block_manager`.

These cover the four decisions the manager makes -- admission, adoption, growth and
publication -- and the invariant that ties them together: a block is in use, free, or
retained by the prefix cache, never two of those at once, however the sequences using it
are admitted, grown, preempted and freed.
"""

from __future__ import annotations

import pytest

from turboserve.engine.core.block_manager import BlockManager
from turboserve.engine.core.kv_cache import BlockAllocator, OutOfBlocksError
from turboserve.engine.core.prefix_cache import PrefixCache
from turboserve.engine.core.sequence import SeqStatus, Sequence
from turboserve.engine.core.types import SamplingParams

BLOCK_SIZE = 4


def make_seq(seq_id: int, prompt: list[int], *, lora_id: int = 0) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        request_id=f"req-{seq_id}",
        prompt_token_ids=list(prompt),
        sampling=SamplingParams(max_tokens=16),
        lora_id=lora_id,
    )


def admit(manager: BlockManager, seq: Sequence, num_tokens: int | None = None) -> list[int]:
    """Allocate, then compute the rest of the prompt in one chunk."""
    manager.allocate(seq)
    todo = num_tokens if num_tokens is not None else seq.num_uncomputed_tokens
    slots = manager.append_slots(seq, todo)
    seq.advance_computed(todo)
    return slots


def test_create_wires_the_cache_in_as_the_recycler() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE)
    assert manager.prefix_cache is not None
    assert manager.block_size == BLOCK_SIZE
    assert manager.num_free_blocks == 8
    seq = make_seq(0, list(range(8)))
    admit(manager, seq)
    manager.publish_computed_blocks(seq)
    manager.free(seq)
    # Published blocks are retained by the cache, not returned to the free list.
    assert manager.allocator.num_retained == 2
    assert manager.num_free_blocks == 6
    assert manager.num_allocatable_blocks == 8
    manager.check_invariants()


def test_disabled_prefix_cache_frees_blocks_outright() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE, enable_prefix_caching=False)
    assert manager.prefix_cache is None
    seq = make_seq(0, list(range(8)))
    admit(manager, seq)
    assert manager.publish_computed_blocks(seq) == 0
    manager.free(seq)
    assert manager.num_free_blocks == 8
    assert manager.allocator.num_retained == 0


def test_constructor_rejects_mismatched_block_sizes() -> None:
    allocator = BlockAllocator(4, BLOCK_SIZE)
    with pytest.raises(ValueError, match="disagrees with the allocator"):
        BlockManager(allocator, BLOCK_SIZE * 2)
    with pytest.raises(ValueError, match="prefix cache block_size"):
        BlockManager(allocator, BLOCK_SIZE, PrefixCache(BLOCK_SIZE * 2))


def test_slot_mapping_is_block_major() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE, enable_prefix_caching=False)
    seq = make_seq(0, list(range(10)))
    slots = admit(manager, seq)
    table = seq.block_table
    assert len(table) == 3
    expected = [table[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE for pos in range(10)]
    assert slots == expected
    assert len(set(slots)) == 10
    with pytest.raises(ValueError, match="no block for position"):
        manager.slot_mapping(seq, 0, 100)


def test_growth_allocates_blocks_only_as_they_are_needed() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE, enable_prefix_caching=False)
    seq = make_seq(0, list(range(4)))
    manager.allocate(seq)
    assert seq.block_table == []
    manager.append_slots(seq, 4)
    seq.advance_computed(4)
    assert len(seq.block_table) == 1
    seq.append_token(99)
    manager.append_slots(seq, 1)
    seq.advance_computed(1)
    assert len(seq.block_table) == 2
    assert manager.num_free_blocks == 6


def test_append_slots_validates_its_inputs() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE, enable_prefix_caching=False)
    seq = make_seq(0, list(range(4)))
    with pytest.raises(ValueError, match="never allocated"):
        manager.append_slots(seq, 1)
    manager.allocate(seq)
    with pytest.raises(ValueError, match="cannot compute up to position"):
        manager.append_slots(seq, 5)
    with pytest.raises(ValueError, match="non-negative"):
        manager.append_slots(seq, -1)
    with pytest.raises(ValueError, match="already holds"):
        manager.append_slots(seq, 4)
        manager.allocate(seq)


def test_admission_accounts_for_retained_prefix_blocks_exactly() -> None:
    manager = BlockManager.create(4, BLOCK_SIZE)
    first = make_seq(0, list(range(16)))
    assert manager.can_allocate(first)
    admit(manager, first)
    manager.publish_computed_blocks(first)
    manager.free(first)
    assert manager.allocator.num_retained == 4  # every block of the prompt was full
    # The pool is nominally full of retained blocks, but a sequence that matches them all
    # needs no new block, so it is admissible.
    second = make_seq(1, list(range(16)))
    assert manager.can_allocate(second)
    # A sequence that shares nothing needs four fresh blocks, which is exactly the pool.
    stranger = make_seq(2, list(range(100, 116)))
    assert manager.can_allocate(stranger)
    too_long = make_seq(3, list(range(100, 130)))
    assert not manager.can_allocate(too_long)


def test_prefix_hit_adopts_blocks_and_credits_computed_tokens() -> None:
    manager = BlockManager.create(16, BLOCK_SIZE)
    prompt = list(range(12))
    first = make_seq(0, prompt)
    admit(manager, first)
    manager.publish_computed_blocks(first)
    manager.free(first)
    cached_blocks = list(manager.prefix_cache._by_block)  # noqa: SLF001 - asserting cache content

    second = make_seq(1, prompt)
    cached_tokens = manager.allocate(second)
    assert cached_tokens == 8, "only full blocks hit, and one token must remain to compute"
    assert second.num_computed_tokens == 8
    assert second.timing.num_cached_prompt_tokens == 8
    assert second.block_table == cached_blocks[:2]
    for block_id in second.block_table:
        assert manager.allocator.ref_count(block_id) == 1
        assert not manager.allocator.is_retained(block_id)
    manager.check_invariants()

    # A third sequence shares the same live blocks by increfing rather than adopting.
    third = make_seq(2, prompt)
    assert manager.allocate(third) == 8
    assert manager.allocator.ref_count(third.block_table[0]) == 2
    manager.check_invariants()


def test_prefix_hit_never_consumes_the_whole_sequence() -> None:
    manager = BlockManager.create(16, BLOCK_SIZE)
    prompt = list(range(8))
    first = make_seq(0, prompt)
    admit(manager, first)
    manager.publish_computed_blocks(first)
    manager.free(first)
    second = make_seq(1, prompt)
    assert manager.allocate(second) == BLOCK_SIZE
    assert second.num_uncomputed_tokens == BLOCK_SIZE


def test_a_different_adapter_does_not_share_blocks() -> None:
    manager = BlockManager.create(16, BLOCK_SIZE)
    prompt = list(range(12))
    base = make_seq(0, prompt, lora_id=0)
    admit(manager, base)
    manager.publish_computed_blocks(base)
    manager.free(base)
    adapted = make_seq(1, prompt, lora_id=1)
    assert manager.allocate(adapted) == 0
    assert adapted.block_table == []


def test_publication_only_covers_blocks_whose_kv_is_computed() -> None:
    manager = BlockManager.create(16, BLOCK_SIZE)
    assert manager.prefix_cache is not None
    seq = make_seq(0, list(range(12)))
    manager.allocate(seq)
    manager.append_slots(seq, 6)
    # Slots exist for six tokens, but nothing is computed yet: nothing may be published.
    assert manager.publish_computed_blocks(seq) == 0
    seq.advance_computed(6)
    assert manager.publish_computed_blocks(seq) == 1  # only the first block is full
    assert manager.prefix_cache.num_cached == 1
    manager.append_slots(seq, 6)
    seq.advance_computed(6)
    assert manager.publish_computed_blocks(seq) == 2
    assert manager.publish_computed_blocks(seq) == 0  # idempotent
    manager.check_invariants()


def test_preempt_frees_blocks_publishes_them_and_resets_progress() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE)
    seq = make_seq(0, list(range(8)))
    admit(manager, seq)
    manager.preempt(seq)
    assert seq.status is SeqStatus.PREEMPTED
    assert seq.num_computed_tokens == 0
    assert seq.block_table == []
    assert seq.num_preemptions == 1
    assert manager.num_allocatable_blocks == 8
    # Resuming re-adopts what it just gave up, so the recompute is nearly free.
    assert manager.allocate(seq) == 4
    manager.check_invariants()


def test_release_rolls_an_admission_back_without_marking_a_preemption() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE)
    seq = make_seq(0, list(range(8)))
    manager.allocate(seq)
    manager.release(seq)
    assert seq.status is SeqStatus.WAITING
    assert seq.num_preemptions == 0
    assert seq.num_computed_tokens == 0
    assert manager.num_sequences == 0


def test_exhausting_the_pool_raises_out_of_blocks() -> None:
    manager = BlockManager.create(2, BLOCK_SIZE, enable_prefix_caching=False)
    seq = make_seq(0, list(range(8)))
    admit(manager, seq)
    assert manager.num_free_blocks == 0
    seq.append_token(1)
    assert not manager.can_append(seq, 1)
    with pytest.raises(OutOfBlocksError):
        manager.append_slots(seq, 1)


def test_can_append_rejects_more_tokens_than_the_sequence_holds() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE, enable_prefix_caching=False)
    seq = make_seq(0, list(range(4)))
    manager.allocate(seq)
    assert manager.can_append(seq, 4)
    assert not manager.can_append(seq, 5)


def test_reset_clears_both_the_pool_and_the_cache() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE)
    assert manager.prefix_cache is not None
    seq = make_seq(0, list(range(8)))
    admit(manager, seq)
    manager.publish_computed_blocks(seq)
    manager.reset()
    assert manager.num_free_blocks == 8
    assert manager.prefix_cache.num_cached == 0
    assert manager.num_sequences == 0
    manager.check_invariants()


def test_stats_expose_pool_and_cache_state() -> None:
    manager = BlockManager.create(8, BLOCK_SIZE)
    seq = make_seq(0, list(range(8)))
    admit(manager, seq)
    stats = manager.stats()
    assert stats["num_sequences"] == 1
    assert stats["num_total"] == 8
    assert stats["num_in_use"] == 2
    assert "prefix_hit_rate" in stats
    assert repr(manager).startswith("BlockManager(")
