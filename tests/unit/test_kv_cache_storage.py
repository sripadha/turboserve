"""Unit tests for the block allocator and the paged KV storage.

Everything here runs on CPU with caches of a few kilobytes: the properties under test are
the ownership invariants (a block is free, in use, or retained -- never two of those) and
the slot arithmetic that maps ``block_id * block_size + offset`` onto the storage tensor.
"""

from __future__ import annotations

import pytest
import torch

from turboserve.engine.core.kv_cache import (
    BlockAllocator,
    BlockAllocatorError,
    BlockRecycler,
    InvalidBlockError,
    KVCache,
    OutOfBlocksError,
)


class FifoRecycler:
    """Test double for a prefix cache: retains every freed block, gives back oldest-first."""

    def __init__(self, *, retain: bool = True) -> None:
        self.retain = retain
        self.retained: list[int] = []
        self.reclaim_calls: list[int] = []

    def on_zero_refs(self, block_id: int) -> bool:
        if not self.retain:
            return False
        self.retained.append(block_id)
        return True

    def reclaim(self, num_blocks: int) -> list[int]:
        self.reclaim_calls.append(num_blocks)
        given, self.retained = self.retained[:num_blocks], self.retained[num_blocks:]
        return given


# --------------------------------------------------------------------------------------
# BlockAllocator
# --------------------------------------------------------------------------------------


def test_allocator_starts_fully_free() -> None:
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    assert allocator.num_total == 4
    assert allocator.num_free == 4
    assert allocator.num_in_use == 0
    assert allocator.num_retained == 0
    assert allocator.block_size == 16
    allocator.check_invariants()


def test_allocate_hands_out_distinct_blocks_with_one_reference() -> None:
    allocator = BlockAllocator(num_blocks=3, block_size=2)
    blocks = [allocator.allocate() for _ in range(3)]
    assert sorted(blocks) == [0, 1, 2]
    assert all(allocator.ref_count(block) == 1 for block in blocks)
    assert allocator.num_free == 0
    assert allocator.num_in_use == 3
    allocator.check_invariants()


def test_allocate_raises_when_the_pool_is_exhausted() -> None:
    allocator = BlockAllocator(num_blocks=1, block_size=2)
    allocator.allocate()
    with pytest.raises(OutOfBlocksError, match="no free KV block"):
        allocator.allocate()


def test_free_list_is_fifo_so_reuse_is_least_recently_freed() -> None:
    allocator = BlockAllocator(num_blocks=3, block_size=2)
    first, second, third = (allocator.allocate() for _ in range(3))
    allocator.decref(second)
    allocator.decref(first)
    allocator.decref(third)
    assert [allocator.allocate() for _ in range(3)] == [second, first, third]


def test_incref_and_decref_track_sharing() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=2)
    block = allocator.allocate()
    assert allocator.incref(block) == 2
    assert allocator.decref(block) == 1
    assert allocator.num_free == 1
    assert allocator.decref(block) == 0
    assert allocator.num_free == 2
    assert allocator.is_free(block)
    allocator.check_invariants()


def test_double_free_raises_instead_of_corrupting_the_free_list() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=2)
    block = allocator.allocate()
    allocator.decref(block)
    with pytest.raises(InvalidBlockError, match="double free"):
        allocator.decref(block)
    assert allocator.num_free == 2
    allocator.check_invariants()


def test_incref_of_a_free_block_raises() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=2)
    with pytest.raises(InvalidBlockError, match="no owner"):
        allocator.incref(0)


@pytest.mark.parametrize("bad_id", [-1, 4, 99])
def test_out_of_range_block_ids_raise(bad_id: int) -> None:
    allocator = BlockAllocator(num_blocks=4, block_size=2)
    with pytest.raises(InvalidBlockError, match="out of range"):
        allocator.ref_count(bad_id)


def test_allocate_many_is_all_or_nothing() -> None:
    allocator = BlockAllocator(num_blocks=4, block_size=2)
    blocks = allocator.allocate_many(3)
    assert len(set(blocks)) == 3
    with pytest.raises(OutOfBlocksError, match="cannot allocate 2 blocks"):
        allocator.allocate_many(2)
    # The failed call must not have consumed the one remaining block.
    assert allocator.num_free == 1
    allocator.check_invariants()


def test_allocate_many_zero_is_a_no_op() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=2)
    assert allocator.allocate_many(0) == []
    assert allocator.num_free == 2


def test_decref_many_and_incref_many() -> None:
    allocator = BlockAllocator(num_blocks=4, block_size=2)
    blocks = allocator.allocate_many(4)
    allocator.incref_many(blocks)
    assert all(allocator.ref_count(block) == 2 for block in blocks)
    allocator.decref_many(blocks)
    allocator.decref_many(blocks)
    assert allocator.num_free == 4
    allocator.check_invariants()


def test_reset_returns_every_block() -> None:
    allocator = BlockAllocator(num_blocks=3, block_size=2)
    allocator.allocate_many(3)
    allocator.reset()
    assert allocator.num_free == 3
    assert allocator.num_in_use == 0
    allocator.check_invariants()


def test_stats_reports_utilization() -> None:
    allocator = BlockAllocator(num_blocks=4, block_size=2)
    allocator.allocate_many(2)
    stats = allocator.stats()
    assert stats["num_total"] == 4
    assert stats["num_in_use"] == 2
    assert stats["utilization"] == pytest.approx(0.5)


def test_repr_is_informative() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=8)
    assert "num_total=2" in repr(allocator)
    assert "block_size=8" in repr(allocator)


@pytest.mark.parametrize(("num_blocks", "block_size"), [(-1, 4), (4, 0), (4, -2)])
def test_allocator_rejects_impossible_geometry(num_blocks: int, block_size: int) -> None:
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks=num_blocks, block_size=block_size)


# --------------------------------------------------------------------------------------
# BlockAllocator with a recycler (the prefix-cache hook)
# --------------------------------------------------------------------------------------


def test_fifo_recycler_satisfies_the_protocol() -> None:
    assert isinstance(FifoRecycler(), BlockRecycler)


def test_freed_blocks_are_retained_by_the_recycler_not_returned_to_the_free_list() -> None:
    recycler = FifoRecycler()
    allocator = BlockAllocator(num_blocks=2, block_size=2, recycler=recycler)
    block = allocator.allocate()
    allocator.decref(block)
    assert allocator.num_free == 1
    assert allocator.num_retained == 1
    assert allocator.is_retained(block)
    assert not allocator.is_free(block)
    assert allocator.num_allocatable == 2
    allocator.check_invariants()


def test_retained_blocks_are_reclaimed_only_when_the_free_list_runs_dry() -> None:
    recycler = FifoRecycler()
    allocator = BlockAllocator(num_blocks=2, block_size=2, recycler=recycler)
    first = allocator.allocate()
    allocator.decref(first)
    second = allocator.allocate()  # takes the remaining free block, not the retained one
    assert second != first
    assert recycler.reclaim_calls == []
    third = allocator.allocate()  # free list empty: must reclaim
    assert third == first
    assert recycler.reclaim_calls == [1]
    assert allocator.num_retained == 0
    allocator.check_invariants()


def test_adopt_takes_a_retained_block_for_a_prefix_hit() -> None:
    recycler = FifoRecycler()
    allocator = BlockAllocator(num_blocks=2, block_size=2, recycler=recycler)
    block = allocator.allocate()
    allocator.decref(block)
    assert allocator.adopt(block) == block
    assert allocator.ref_count(block) == 1
    assert allocator.num_retained == 0
    assert allocator.num_in_use == 1
    allocator.check_invariants()


def test_adopt_rejects_a_free_or_live_block() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=2)
    with pytest.raises(InvalidBlockError, match="not retained"):
        allocator.adopt(0)
    block = allocator.allocate()
    with pytest.raises(InvalidBlockError, match="not retained"):
        allocator.adopt(block)


def test_release_retained_frees_only_retained_ids() -> None:
    recycler = FifoRecycler()
    allocator = BlockAllocator(num_blocks=3, block_size=2, recycler=recycler)
    retained = allocator.allocate()
    live = allocator.allocate()
    allocator.decref(retained)
    assert allocator.release_retained([retained, live]) == 1
    assert allocator.is_free(retained)
    assert allocator.ref_count(live) == 1
    allocator.check_invariants()


def test_declining_recycler_behaves_like_no_recycler() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=2, recycler=FifoRecycler(retain=False))
    block = allocator.allocate()
    allocator.decref(block)
    assert allocator.num_free == 2
    assert allocator.num_retained == 0


def test_check_invariants_catches_a_manufactured_double_free() -> None:
    allocator = BlockAllocator(num_blocks=2, block_size=2)
    allocator.allocate()
    allocator._free.append(0)  # simulate the corruption decref() is guarding against
    with pytest.raises(BlockAllocatorError, match="referenced but also reclaimable"):
        allocator.check_invariants()


# --------------------------------------------------------------------------------------
# KVCache: sizing
# --------------------------------------------------------------------------------------

#: 2 layers, 4 tokens per block, 2 KV heads, head_dim 8, fp32:
#: per layer  2 (k and v) * 4 * 2 * 8 * 4 bytes = 512; across 2 layers = 1024 bytes/block.
GEOMETRY = {
    "num_layers": 2,
    "block_size": 4,
    "num_kv_heads": 2,
    "head_dim": 8,
    "dtype": torch.float32,
}


def test_bytes_per_block_matches_hand_computation() -> None:
    cache = KVCache(num_blocks=6, **GEOMETRY)
    assert cache.bytes_per_block_per_layer == 512
    assert cache.bytes_per_block == 1024
    assert cache.total_bytes == 6 * 1024
    assert cache.num_slots == 24
    assert cache.shape == (6, 4, 2, 8)


def test_bytes_per_block_for_matches_the_instance_property() -> None:
    cache = KVCache(num_blocks=3, **GEOMETRY)
    assert KVCache.bytes_per_block_for(**GEOMETRY) == cache.bytes_per_block


def test_half_precision_halves_the_footprint() -> None:
    geometry = {**GEOMETRY, "dtype": torch.float16}
    assert KVCache.bytes_per_block_for(**geometry) == 512


def test_blocks_for_memory_rounds_down() -> None:
    assert KVCache.blocks_for_memory(10_000, **GEOMETRY) == 9
    assert KVCache.blocks_for_memory(1024, **GEOMETRY) == 1
    assert KVCache.blocks_for_memory(1023, **GEOMETRY) == 0


def test_blocks_for_memory_never_negative() -> None:
    assert KVCache.blocks_for_memory(0, **GEOMETRY) == 0
    assert KVCache.blocks_for_memory(-5, **GEOMETRY) == 0


def test_slot_index_is_block_times_size_plus_offset() -> None:
    assert KVCache.slot_index(3, 2, 4) == 14


@pytest.mark.parametrize("field", ["num_layers", "block_size", "num_kv_heads", "head_dim"])
def test_kv_cache_rejects_non_positive_geometry(field: str) -> None:
    geometry = {**GEOMETRY, field: 0}
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        KVCache(num_blocks=2, **geometry)


# --------------------------------------------------------------------------------------
# KVCache: storage
# --------------------------------------------------------------------------------------


def _cache() -> KVCache:
    return KVCache(
        num_layers=2,
        num_blocks=4,
        block_size=2,
        num_kv_heads=2,
        head_dim=3,
        dtype=torch.float32,
        device="cpu",
    )


def test_allocation_is_lazy() -> None:
    cache = _cache()
    assert not cache.is_allocated
    assert cache.bytes_per_block > 0  # sizing questions do not allocate
    assert not cache.is_allocated
    cache.allocate()
    assert cache.is_allocated
    assert len(cache.k) == len(cache.v) == 2
    assert cache.k[0].shape == (4, 2, 2, 3)


def test_accessing_k_allocates_on_demand() -> None:
    cache = _cache()
    assert cache.k[0].abs().sum().item() == 0.0
    assert cache.is_allocated


def test_free_releases_the_tensors_and_allocate_is_idempotent() -> None:
    cache = _cache()
    cache.allocate()
    cache.allocate()
    assert len(cache.k) == 2
    cache.free()
    assert not cache.is_allocated


def test_write_read_roundtrip() -> None:
    cache = _cache()
    torch.manual_seed(0)
    slots = torch.tensor([0, 1, 5], dtype=torch.long)
    k = torch.randn(3, 2, 3)
    v = torch.randn(3, 2, 3)
    cache.write(0, slots, k, v)
    read_k, read_v = cache.read(0, slots)
    torch.testing.assert_close(read_k, k)
    torch.testing.assert_close(read_v, v)


def test_write_lands_in_the_right_block_and_offset() -> None:
    cache = _cache()
    # slot 5 == block 2, offset 1 at block_size 2.
    slots = torch.tensor([5], dtype=torch.long)
    k = torch.full((1, 2, 3), 7.0)
    v = torch.full((1, 2, 3), -7.0)
    cache.write(1, slots, k, v)
    block_k, block_v = cache.block(1, 2)
    assert block_k[0].abs().sum().item() == 0.0
    torch.testing.assert_close(block_k[1], k[0])
    torch.testing.assert_close(block_v[1], v[0])


def test_layers_do_not_alias_each_other() -> None:
    cache = _cache()
    slots = torch.tensor([0], dtype=torch.long)
    cache.write(0, slots, torch.ones(1, 2, 3), torch.ones(1, 2, 3))
    read_k, _ = cache.read(1, slots)
    assert read_k.abs().sum().item() == 0.0


def test_write_overwrites_the_same_slot() -> None:
    cache = _cache()
    slots = torch.tensor([3], dtype=torch.long)
    cache.write(0, slots, torch.ones(1, 2, 3), torch.ones(1, 2, 3))
    cache.write(0, slots, torch.full((1, 2, 3), 2.0), torch.full((1, 2, 3), 2.0))
    read_k, read_v = cache.read(0, slots)
    torch.testing.assert_close(read_k, torch.full((1, 2, 3), 2.0))
    torch.testing.assert_close(read_v, torch.full((1, 2, 3), 2.0))


def test_write_casts_the_incoming_dtype() -> None:
    cache = KVCache(
        num_layers=1, num_blocks=2, block_size=2, num_kv_heads=1, head_dim=2, dtype=torch.float16
    )
    slots = torch.tensor([0], dtype=torch.long)
    cache.write(0, slots, torch.ones(1, 1, 2, dtype=torch.float32), torch.ones(1, 1, 2))
    read_k, _ = cache.read(0, slots)
    assert read_k.dtype == torch.float16


def test_zero_clears_every_layer() -> None:
    cache = _cache()
    slots = torch.tensor([0, 1], dtype=torch.long)
    cache.write(0, slots, torch.ones(2, 2, 3), torch.ones(2, 2, 3))
    cache.zero_()
    read_k, read_v = cache.read(0, slots)
    assert read_k.abs().sum().item() == 0.0
    assert read_v.abs().sum().item() == 0.0


def test_write_rejects_a_mismatched_token_count() -> None:
    cache = _cache()
    slots = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="k must have shape"):
        cache.write(0, slots, torch.ones(3, 2, 3), torch.ones(2, 2, 3))


def test_write_rejects_a_mismatched_head_shape() -> None:
    cache = _cache()
    slots = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="v must have shape"):
        cache.write(0, slots, torch.ones(1, 2, 3), torch.ones(1, 2, 4))


def test_write_rejects_a_non_int64_slot_mapping() -> None:
    cache = _cache()
    with pytest.raises(ValueError, match="slot_mapping must be int64"):
        cache.write(
            0, torch.tensor([0], dtype=torch.int32), torch.ones(1, 2, 3), torch.ones(1, 2, 3)
        )


def test_write_rejects_a_two_dimensional_slot_mapping() -> None:
    cache = _cache()
    with pytest.raises(ValueError, match="slot_mapping must be 1-D"):
        cache.write(
            0, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 2, 3), torch.ones(1, 2, 3)
        )


@pytest.mark.parametrize("layer", [-1, 2, 5])
def test_layer_bounds_are_checked(layer: int) -> None:
    cache = _cache()
    with pytest.raises(ValueError, match="out of range"):
        cache.layer(layer)


def test_block_bounds_are_checked() -> None:
    cache = _cache()
    with pytest.raises(ValueError, match="block id 9 out of range"):
        cache.block(0, 9)


def test_repr_reports_geometry_and_allocation_state() -> None:
    cache = _cache()
    assert "allocated=False" in repr(cache)
    cache.allocate()
    assert "allocated=True" in repr(cache)
