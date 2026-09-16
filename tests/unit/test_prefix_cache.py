"""Unit tests for :mod:`turboserve.engine.core.prefix_cache`.

Three families of property are checked: the hash chain really is a chain (any earlier
difference changes every later hash), the index and the LRU order stay consistent through
insert/acquire/release/evict, and the cache honours the recycler contract the block
allocator depends on -- above all that a cached block is never also on the free list.
"""

from __future__ import annotations

import pytest

from turboserve.engine.core.kv_cache import BlockAllocator, BlockRecycler
from turboserve.engine.core.prefix_cache import (
    ROOT_HASH,
    PrefixCache,
    block_hash,
    block_hash_chain,
)
from turboserve.engine.core.types import NO_LORA

BLOCK_SIZE = 4


def test_hash_chain_depends_on_every_earlier_token() -> None:
    base = list(range(12))
    chain = block_hash_chain(base, BLOCK_SIZE)
    assert len(chain) == 3
    for position in range(12):
        changed = list(base)
        changed[position] = 999
        other = block_hash_chain(changed, BLOCK_SIZE)
        first_affected = position // BLOCK_SIZE
        assert chain[:first_affected] == other[:first_affected]
        for index in range(first_affected, 3):
            assert chain[index] != other[index], f"token {position} did not change block {index}"


def test_hash_chain_is_deterministic_and_length_sensitive() -> None:
    assert block_hash_chain([1, 2, 3, 4], BLOCK_SIZE) == block_hash_chain([1, 2, 3, 4], BLOCK_SIZE)
    # A partial trailing block is never hashed: it will still receive tokens.
    assert block_hash_chain([1, 2, 3], BLOCK_SIZE) == []
    assert len(block_hash_chain(list(range(9)), BLOCK_SIZE)) == 2
    assert len(block_hash_chain(list(range(9)), BLOCK_SIZE, max_blocks=1)) == 1


def test_same_tokens_at_a_different_offset_hash_differently() -> None:
    tail = [9, 9, 9, 9]
    as_first = block_hash(ROOT_HASH, tail)
    as_second = block_hash(block_hash(ROOT_HASH, [1, 1, 1, 1]), tail)
    assert as_first != as_second


def test_lora_id_is_part_of_the_identity() -> None:
    tokens = [1, 2, 3, 4]
    assert block_hash(ROOT_HASH, tokens, NO_LORA) != block_hash(ROOT_HASH, tokens, 1)
    assert block_hash_chain(tokens, BLOCK_SIZE, 1) != block_hash_chain(tokens, BLOCK_SIZE, 2)


def test_lookup_returns_the_longest_unbroken_prefix() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    tokens = list(range(12))
    chain = block_hash_chain(tokens, BLOCK_SIZE)
    assert cache.lookup(tokens) == []
    cache.insert(10, chain[0])
    cache.insert(12, chain[2])  # a gap: block 1 is missing
    assert cache.lookup(tokens) == [10]
    cache.insert(11, chain[1])
    assert cache.lookup(tokens) == [10, 11, 12]
    match = cache.match(tokens)
    assert match.num_blocks == 3
    assert match.num_tokens == 12
    assert match.hashes == tuple(chain)
    assert not match.is_empty


def test_lookup_respects_max_blocks_and_partial_blocks() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    tokens = list(range(10))
    for index, digest in enumerate(block_hash_chain(tokens, BLOCK_SIZE)):
        cache.insert(index, digest)
    assert cache.lookup(tokens) == [0, 1]
    assert cache.lookup(tokens, max_blocks=1) == [0]
    assert cache.lookup(tokens[:3]) == []


def test_statistics_count_queried_blocks() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    tokens = list(range(12))
    chain = block_hash_chain(tokens, BLOCK_SIZE)
    cache.insert(0, chain[0])
    cache.lookup(tokens)
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["tokens_saved"] == BLOCK_SIZE
    assert stats["hit_rate"] == pytest.approx(1 / 3)
    cache.lookup(tokens, record=False)
    assert cache.stats()["hits"] == 1


def test_insert_is_idempotent_and_refuses_reinterpreting_a_block() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    first, second = block_hash_chain([1, 2, 3, 4, 5, 6, 7, 8], BLOCK_SIZE)
    assert cache.insert(3, first) is True
    assert cache.insert(3, first) is False
    # A different block computed the same prefix concurrently: the first one keeps the slot.
    assert cache.insert(4, first) is False
    assert cache.get(first) == 3
    with pytest.raises(ValueError, match="different hash"):
        cache.insert(3, second)


def test_release_makes_a_block_evictable_and_acquire_pins_it_again() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    digest = block_hash(ROOT_HASH, [1, 2, 3, 4])
    cache.insert(5, digest)
    assert cache.num_evictable == 0
    assert cache.release(5) is True
    assert cache.is_evictable(5)
    cache.acquire(5)
    assert not cache.is_evictable(5)
    assert cache.evict(4) == []
    assert cache.release(99) is False  # unknown block: the allocator just frees it


def test_eviction_is_least_recently_released_first() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    chain = block_hash_chain(list(range(12)), BLOCK_SIZE)
    for index, digest in enumerate(chain):
        cache.insert(index, digest)
        cache.release(index)
    cache.acquire(1)
    cache.release(1)  # block 1 becomes the most recently released
    assert cache.evict(2) == [0, 2]
    assert cache.num_cached == 1
    assert cache.get(chain[0]) is None
    assert cache.evict(0) == []
    assert cache.stats()["evictions"] == 2


def test_remove_and_reset_clear_both_indexes() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    chain = block_hash_chain(list(range(8)), BLOCK_SIZE)
    cache.insert(0, chain[0])
    cache.insert(1, chain[1])
    cache.release(1)
    assert cache.remove(1) is True
    assert cache.remove(1) is False
    assert cache.num_evictable == 0
    assert cache.remove_many([0, 7]) == 1
    assert len(cache) == 0
    cache.insert(2, chain[0])
    cache.reset()
    cache.check_invariants()
    assert cache.num_cached == 0
    cache.reset_stats()
    assert cache.stats()["inserts"] == 0


def test_cache_satisfies_the_recycler_protocol() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    assert isinstance(cache, BlockRecycler)
    digest = block_hash(ROOT_HASH, [1, 2, 3, 4])
    cache.insert(0, digest)
    assert cache.on_zero_refs(0) is True
    assert cache.on_zero_refs(1) is False
    assert cache.reclaim(1) == [0]


def test_cached_blocks_are_never_free_in_the_allocator() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    allocator = BlockAllocator(4, BLOCK_SIZE, recycler=cache)
    chain = block_hash_chain(list(range(8)), BLOCK_SIZE)
    first, second = allocator.allocate(), allocator.allocate()
    cache.insert(first, chain[0])
    cache.insert(second, chain[1])
    allocator.decref(first)
    allocator.decref(second)
    assert allocator.num_free == 2
    assert allocator.num_retained == 2
    for block_id in (first, second):
        assert cache.contains_block(block_id)
        assert not allocator.is_free(block_id)
        assert allocator.is_retained(block_id) == cache.is_evictable(block_id)
    # Draining the pool forces the cache to give the retained blocks back.
    allocator.allocate_many(4)
    assert cache.num_cached == 0
    assert allocator.num_retained == 0
    allocator.check_invariants()
    cache.check_invariants()


def test_check_invariants_catches_a_desynchronised_lru() -> None:
    cache = PrefixCache(BLOCK_SIZE)
    cache.insert(0, block_hash(ROOT_HASH, [1, 2, 3, 4]))
    cache.release(0)
    cache.check_invariants()
    cache._by_block.clear()  # simulate a bug elsewhere
    with pytest.raises(ValueError):
        cache.check_invariants()


def test_block_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="block_size"):
        PrefixCache(0)
    with pytest.raises(ValueError, match="block_size"):
        block_hash_chain([1, 2], 0)
