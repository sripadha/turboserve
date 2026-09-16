"""Content-addressed cache of computed KV blocks ("automatic prefix caching").

Two requests that start with the same tokens produce, layer for layer, exactly the same
keys and values for that shared span -- attention is causal, so a prefix's KV does not
depend on anything that comes after it. The second request can therefore adopt the first
request's blocks instead of running prefill over them again. That is the whole idea; this
module is the index that makes it possible to *find* those blocks.

**Hash chain.** A block is identified not by its own tokens but by the entire prefix
ending at it: ``h_i = blake2b(h_{i-1} || lora_id || tokens_i)``, with a fixed root
:data:`ROOT_HASH`. Chaining is what makes the identity sound -- ``["b", "c"]`` as the
second block of ``["a"], ["b", "c"]`` is a different thing from the same tokens at the
start of a sequence, and only the chained hash distinguishes them. The LoRA id is mixed in
for the same reason: an adapter changes the projections, so the same tokens under a
different adapter are different KV and must not collide. Only *full* blocks are hashed; a
partially filled block will still receive tokens, so caching it would publish KV under a
hash that does not describe the block's final contents.

**Ownership.** This cache never allocates or frees memory. It implements
:class:`~turboserve.engine.core.kv_cache.BlockRecycler`, the one hook
:class:`~turboserve.engine.core.kv_cache.BlockAllocator` offers: when a block's reference
count reaches zero the allocator asks :meth:`PrefixCache.on_zero_refs` whether to keep it,
and when the allocator runs out of blocks it asks :meth:`PrefixCache.reclaim` for the
least recently used ones back. So a cached block is never "free": it is either in use by a
sequence or *retained*, and the allocator's three-state invariant holds unchanged.

Collision risk is the usual content-addressing trade-off: a 128-bit blake2b digest is
treated as identity, and two different prefixes that collide would share KV. The
alternative -- storing and comparing every block's token ids -- costs memory proportional
to the cache and a comparison on every lookup; vLLM makes the same trade.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from turboserve.engine.core.types import NO_LORA

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "HASH_DIGEST_SIZE",
    "ROOT_HASH",
    "PrefixCache",
    "PrefixMatch",
    "block_hash",
    "block_hash_chain",
]

#: Digest length in bytes. 16 bytes (128 bits) keeps the index small while leaving the
#: chance of an accidental collision far below the chance of a silent hardware fault.
HASH_DIGEST_SIZE = 8 * 2

#: Root of every hash chain. Domain-separated so that a digest from this cache can never
#: be confused with a digest produced elsewhere in the project.
ROOT_HASH: bytes = hashlib.blake2b(
    b"turboserve/prefix-cache/v1", digest_size=HASH_DIGEST_SIZE
).digest()


def _token_bytes(token_ids: Sequence[int]) -> bytes:
    """Encode token ids as fixed-width little-endian int64, so lengths cannot alias."""
    return b"".join(int(token).to_bytes(8, "little", signed=True) for token in token_ids)


def block_hash(parent_hash: bytes, token_ids: Sequence[int], lora_id: int = NO_LORA) -> bytes:
    """Hash one full block given the hash of the block before it.

    ``parent_hash`` is :data:`ROOT_HASH` for the first block of a sequence. The result
    identifies *the prefix ending at this block*, not the block's tokens in isolation.
    """
    digest = hashlib.blake2b(digest_size=HASH_DIGEST_SIZE)
    digest.update(parent_hash)
    digest.update(int(lora_id).to_bytes(8, "little", signed=True))
    digest.update(_token_bytes(token_ids))
    return digest.digest()


def block_hash_chain(
    token_ids: Sequence[int],
    block_size: int,
    lora_id: int = NO_LORA,
    *,
    max_blocks: int | None = None,
) -> list[bytes]:
    """Hashes of every full block of ``token_ids``, in order.

    A trailing partial block is skipped (see the module docstring). ``max_blocks`` caps
    the chain, which callers use to keep at least one token of the prompt uncomputed.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    num_full = len(token_ids) // block_size
    if max_blocks is not None:
        num_full = min(num_full, max(0, max_blocks))
    hashes: list[bytes] = []
    parent = ROOT_HASH
    for index in range(num_full):
        block = token_ids[index * block_size : (index + 1) * block_size]
        parent = block_hash(parent, block, lora_id)
        hashes.append(parent)
    return hashes


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    """The longest cached prefix found for a token sequence.

    Carries the hashes as well as the block ids because the caller has to continue the
    chain: the first block it computes itself hangs off ``hashes[-1]``, and recomputing
    the chain from the root to find that out would defeat the purpose.
    """

    block_ids: tuple[int, ...]
    hashes: tuple[bytes, ...]
    block_size: int

    @property
    def num_blocks(self) -> int:
        """Number of cached blocks matched."""
        return len(self.block_ids)

    @property
    def num_tokens(self) -> int:
        """Number of prompt tokens whose KV the match makes unnecessary to recompute."""
        return len(self.block_ids) * self.block_size

    @property
    def is_empty(self) -> bool:
        """Whether nothing matched."""
        return not self.block_ids


@dataclass(slots=True)
class _Entry:
    """Index record for one cached block."""

    block_id: int
    block_hash: bytes


class PrefixCache:
    """Hash -> block id index with LRU eviction of unreferenced blocks.

    The cache holds two kinds of block. A block that some sequence is still using is
    *pinned*: it is in the index and can be adopted by another sequence, but it cannot be
    evicted, because evicting it would hand live KV memory to somebody else. A block whose
    last user has gone is *evictable*: still in the index, still holding valid KV, and
    first in line to be handed back to the allocator when the pool runs dry. The LRU order
    covers only evictable blocks, ordered by the moment they became evictable.

    The class satisfies :class:`~turboserve.engine.core.kv_cache.BlockRecycler`
    structurally; the allocator only ever calls :meth:`on_zero_refs` and :meth:`reclaim`.
    """

    __slots__ = (
        "_block_size",
        "_by_block",
        "_by_hash",
        "_evictable",
        "_evictions",
        "_hits",
        "_inserts",
        "_misses",
    )

    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._block_size = block_size
        self._by_hash: dict[bytes, int] = {}
        self._by_block: dict[int, _Entry] = {}
        # Ordered set of evictable block ids, oldest first. OrderedDict rather than a
        # deque so that acquire() can remove a block from the middle in O(1).
        self._evictable: OrderedDict[int, None] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._inserts = 0
        self._evictions = 0

    # -- introspection -------------------------------------------------------------------

    @property
    def block_size(self) -> int:
        """Tokens per block; must equal the allocator's and the scheduler's."""
        return self._block_size

    @property
    def num_cached(self) -> int:
        """Blocks in the index, pinned and evictable together."""
        return len(self._by_block)

    @property
    def num_evictable(self) -> int:
        """Cached blocks no sequence is using, i.e. what :meth:`reclaim` can give back."""
        return len(self._evictable)

    @property
    def hit_rate(self) -> float:
        """Fraction of queried blocks that were found; ``0.0`` before any lookup."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def contains_block(self, block_id: int) -> bool:
        """Whether this block is indexed by the cache."""
        return block_id in self._by_block

    def is_evictable(self, block_id: int) -> bool:
        """Whether this block is cached and currently unreferenced."""
        return block_id in self._evictable

    def get(self, block_hash_value: bytes) -> int | None:
        """Block id for a hash, or ``None``. Does not affect LRU order or statistics."""
        return self._by_hash.get(block_hash_value)

    def stats(self) -> dict[str, int | float]:
        """Snapshot for the engine's ``stats()`` and the scheduler's logs."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "tokens_saved": self._hits * self._block_size,
            "inserts": self._inserts,
            "evictions": self._evictions,
            "num_cached": len(self._by_block),
            "num_evictable": len(self._evictable),
        }

    # -- lookup ---------------------------------------------------------------------------

    def match(
        self,
        token_ids: Sequence[int],
        lora_id: int = NO_LORA,
        *,
        max_blocks: int | None = None,
        record: bool = True,
    ) -> PrefixMatch:
        """Longest run of cached full blocks starting at the beginning of ``token_ids``.

        The walk stops at the first miss, even if a later block happens to be in the index:
        attention needs an unbroken prefix, and a block whose ancestors are gone is
        unreachable by construction. ``record=False`` asks the same question without
        counting it, which is what an admission check does before it decides whether the
        sequence can be admitted at all.
        """
        num_full = len(token_ids) // self._block_size
        if max_blocks is not None:
            num_full = min(num_full, max(0, max_blocks))
        blocks: list[int] = []
        hashes: list[bytes] = []
        parent = ROOT_HASH
        for index in range(num_full):
            parent = block_hash(
                parent,
                token_ids[index * self._block_size : (index + 1) * self._block_size],
                lora_id,
            )
            block_id = self._by_hash.get(parent)
            if block_id is None:
                break
            blocks.append(block_id)
            hashes.append(parent)
        if record:
            self._hits += len(blocks)
            self._misses += num_full - len(blocks)
        return PrefixMatch(tuple(blocks), tuple(hashes), self._block_size)

    def lookup(
        self,
        token_ids: Sequence[int],
        lora_id: int = NO_LORA,
        *,
        max_blocks: int | None = None,
        record: bool = True,
    ) -> list[int]:
        """Block ids of the longest cached prefix. The caller must incref or adopt them."""
        return list(self.match(token_ids, lora_id, max_blocks=max_blocks, record=record).block_ids)

    # -- mutation --------------------------------------------------------------------------

    def insert(self, block_id: int, block_hash_value: bytes) -> bool:
        """Index a freshly computed, full block. Returns whether it was indexed.

        ``False`` means an equivalent block (same hash) is already cached: two sequences
        computed the same prefix concurrently, which the cache cannot prevent because both
        missed before either finished. The caller keeps its own block; it simply stays
        uncached and is freed normally, and later requests share the block that got there
        first.

        Raises :class:`ValueError` if the block id is already indexed under a *different*
        hash, which would mean a block's contents were reinterpreted while it was cached.
        """
        existing = self._by_block.get(block_id)
        if existing is not None:
            if existing.block_hash == block_hash_value:
                return False
            raise ValueError(
                f"block {block_id} is already cached under a different hash; "
                "a cached block must be removed before its contents are reused"
            )
        if block_hash_value in self._by_hash:
            return False
        self._by_hash[block_hash_value] = block_id
        self._by_block[block_id] = _Entry(block_id, block_hash_value)
        self._inserts += 1
        return True

    def acquire(self, block_id: int) -> None:
        """Note that a sequence now references this block, so it must not be evicted.

        Called after the block manager adopts or increfs a cached block. Blocks that are
        not in the index are ignored, so callers do not have to check first.
        """
        self._evictable.pop(block_id, None)

    def release(self, block_id: int) -> bool:
        """Note that the block lost its last reference; keep it if it is cached.

        Returns whether the cache retains the block. This is the decision the allocator
        acts on: ``True`` moves the block to the retained state instead of the free list.
        """
        if block_id not in self._by_block:
            return False
        self._evictable[block_id] = None
        self._evictable.move_to_end(block_id)
        return True

    def on_zero_refs(self, block_id: int) -> bool:
        """:class:`~turboserve.engine.core.kv_cache.BlockRecycler` hook: same as :meth:`release`."""
        return self.release(block_id)

    def evict(self, num_blocks: int) -> list[int]:
        """Drop up to ``num_blocks`` least recently released blocks from the index.

        The returned ids are no longer cached; the allocator puts them back on the free
        list. Pinned blocks are never returned, so eviction can come up short -- that is
        the signal that the pool is genuinely full of live KV and the scheduler must
        preempt instead.
        """
        if num_blocks <= 0:
            return []
        evicted: list[int] = []
        while self._evictable and len(evicted) < num_blocks:
            block_id, _ = self._evictable.popitem(last=False)
            entry = self._by_block.pop(block_id, None)
            if entry is not None:
                self._by_hash.pop(entry.block_hash, None)
            evicted.append(block_id)
        self._evictions += len(evicted)
        if evicted:
            logger.debug("prefix cache evicted %d blocks", len(evicted))
        return evicted

    def reclaim(self, num_blocks: int) -> list[int]:
        """:class:`~turboserve.engine.core.kv_cache.BlockRecycler` hook: same as :meth:`evict`."""
        return self.evict(num_blocks)

    def remove(self, block_id: int) -> bool:
        """Forget a block entirely, whether pinned or evictable. Returns whether it was known."""
        entry = self._by_block.pop(block_id, None)
        if entry is None:
            return False
        self._by_hash.pop(entry.block_hash, None)
        self._evictable.pop(block_id, None)
        return True

    def remove_many(self, block_ids: Iterable[int]) -> int:
        """Forget several blocks; returns how many were known."""
        return sum(1 for block_id in block_ids if self.remove(block_id))

    def reset(self) -> None:
        """Empty the index, keeping the statistics counters."""
        self._by_hash.clear()
        self._by_block.clear()
        self._evictable.clear()

    def reset_stats(self) -> None:
        """Zero the hit/miss/insert/eviction counters without touching the index."""
        self._hits = 0
        self._misses = 0
        self._inserts = 0
        self._evictions = 0

    # -- invariants -------------------------------------------------------------------------

    def check_invariants(self) -> None:
        """Raise :class:`ValueError` if the two indexes or the LRU order disagree.

        Linear in the cache size: a test and debugging tool, not a per-step check.
        """
        if len(self._by_hash) != len(self._by_block):
            raise ValueError(
                f"index sizes disagree: {len(self._by_hash)} hashes, {len(self._by_block)} blocks"
            )
        for block_id, entry in self._by_block.items():
            if entry.block_id != block_id:
                raise ValueError(f"entry for block {block_id} records block {entry.block_id}")
            if self._by_hash.get(entry.block_hash) != block_id:
                raise ValueError(f"block {block_id} is not reachable by its own hash")
        for block_id in self._evictable:
            if block_id not in self._by_block:
                raise ValueError(f"block {block_id} is evictable but not cached")

    def __len__(self) -> int:
        return len(self._by_block)

    def __repr__(self) -> str:
        return (
            f"PrefixCache(block_size={self._block_size}, cached={len(self._by_block)}, "
            f"evictable={len(self._evictable)}, hits={self._hits}, misses={self._misses})"
        )
