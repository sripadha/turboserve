"""Policy layer over the block allocator: who gets which KV blocks, and when.

:class:`~turboserve.engine.core.kv_cache.BlockAllocator` knows how to hand out and count
references on blocks; :class:`~turboserve.engine.core.prefix_cache.PrefixCache` knows which
block holds which prefix. :class:`BlockManager` is the only place that knows about both and
about :class:`~turboserve.engine.core.sequence.Sequence`, so it is where the four decisions
that matter are made:

* **Admission** (:meth:`can_allocate`) -- is there room for this request's whole KV, given
  what its prefix already has cached? Admitting a request the pool cannot hold would
  guarantee a preemption a few steps later, after the prefill work has been spent.
* **Adoption** (:meth:`allocate`) -- take references on the cached blocks of the longest
  matching prefix and credit them as already computed, so prefill starts partway in.
* **Growth** (:meth:`append_slots`) -- extend a sequence one chunk or one token at a time,
  handing back the flat KV slots the model writes into.
* **Publication** (:meth:`publish_computed_blocks`) -- offer newly completed blocks to the
  prefix cache, but only once their KV has definitely been written.

The publication rule is the subtle one. A block becomes *full* during a step, but its KV is
written by the model **after** the scheduler produced the step's metadata. Publishing it
inside that same step would let another sequence in the *same batch* adopt a block whose
contents do not exist yet. So blocks are published at the top of the following step (the
scheduler calls this method before it schedules anything), by which point the forward pass
that filled them has certainly run. The cost is that a block becomes shareable one step
later than theoretically possible; the benefit is that "cached" never means "about to be".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from turboserve.engine.core.kv_cache import BlockAllocator
from turboserve.engine.core.prefix_cache import ROOT_HASH, PrefixCache, block_hash
from turboserve.engine.core.sequence import Sequence

logger = logging.getLogger(__name__)

__all__ = ["BlockManager"]


@dataclass(slots=True)
class _SeqBlocks:
    """Per-sequence hashing bookkeeping.

    ``hashes[i]`` is the chained hash of the sequence's ``i``-th block; the list grows as
    blocks fill so the chain is never recomputed from the root. ``num_published`` is how
    many of those blocks the prefix cache has been told about, which is also where a
    resumed or adopted sequence starts from.
    """

    hashes: list[bytes] = field(default_factory=list)
    num_published: int = 0


class BlockManager:
    """Allocates, shares and reclaims the KV blocks of running sequences."""

    __slots__ = ("_allocator", "_block_size", "_prefix_cache", "_state")

    def __init__(
        self,
        allocator: BlockAllocator,
        block_size: int,
        prefix_cache: PrefixCache | None = None,
    ) -> None:
        if block_size != allocator.block_size:
            raise ValueError(
                f"block_size {block_size} disagrees with the allocator's {allocator.block_size}"
            )
        if prefix_cache is not None and prefix_cache.block_size != block_size:
            raise ValueError(
                f"prefix cache block_size {prefix_cache.block_size} disagrees with "
                f"the manager's {block_size}"
            )
        self._allocator = allocator
        self._block_size = block_size
        self._prefix_cache = prefix_cache
        self._state: dict[int, _SeqBlocks] = {}

    @classmethod
    def create(
        cls,
        num_blocks: int,
        block_size: int,
        *,
        enable_prefix_caching: bool = True,
    ) -> BlockManager:
        """Build an allocator, a prefix cache and a manager already wired together.

        The wiring matters and is easy to get wrong by hand: the cache must be installed as
        the allocator's *recycler* before any block is freed, otherwise freed blocks go
        straight back on the free list and the cache silently never retains anything.
        """
        cache = PrefixCache(block_size) if enable_prefix_caching else None
        allocator = BlockAllocator(num_blocks, block_size, recycler=cache)
        return cls(allocator, block_size, cache)

    # -- introspection -------------------------------------------------------------------

    @property
    def allocator(self) -> BlockAllocator:
        """The underlying reference-counted block pool."""
        return self._allocator

    @property
    def prefix_cache(self) -> PrefixCache | None:
        """The prefix cache, or ``None`` when prefix caching is disabled."""
        return self._prefix_cache

    @property
    def block_size(self) -> int:
        """Tokens per block."""
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        """Blocks available without evicting anything."""
        return self._allocator.num_free

    @property
    def num_allocatable_blocks(self) -> int:
        """Blocks available including those the prefix cache would give back."""
        return self._allocator.num_allocatable

    @property
    def num_sequences(self) -> int:
        """Sequences currently holding blocks."""
        return len(self._state)

    def stats(self) -> dict[str, int | float]:
        """Snapshot of pool occupancy and prefix-cache effectiveness."""
        out: dict[str, int | float] = {"num_sequences": len(self._state)}
        out.update(self._allocator.stats())
        if self._prefix_cache is not None:
            out.update({f"prefix_{k}": v for k, v in self._prefix_cache.stats().items()})
        return out

    # -- admission and adoption ------------------------------------------------------------

    def _max_hit_blocks(self, seq: Sequence) -> int:
        """Cap on prefix-cache hits: at least one token must be left to compute.

        A sequence whose entire content was cached would enter the next step with nothing
        to run through the model, and so with no hidden state to take logits from. The
        engine would have to fall back to recomputing the last block anyway, so the cap is
        applied up front where it is cheap.
        """
        return max(0, (seq.num_tokens - 1) // self._block_size)

    def can_allocate(self, seq: Sequence) -> bool:
        """Whether the pool can hold this sequence's KV, counting prefix hits.

        The arithmetic is less obvious than "needed minus hits": adopting a *retained*
        cached block converts it from allocatable to in-use, so it stops counting towards
        the space left for the blocks that still have to be computed. A hit on a block that
        another sequence is already using costs nothing, because that block was never
        allocatable in the first place. Both cases are handled exactly rather than
        conservatively, because a conservative admission test shows up directly as queueing
        delay for long prompts with warm prefixes.
        """
        needed = seq.num_blocks_needed(self._block_size)
        if self._prefix_cache is None:
            return needed <= self._allocator.num_allocatable
        match = self._prefix_cache.match(
            seq.token_ids, seq.lora_id, max_blocks=self._max_hit_blocks(seq), record=False
        )
        retained_hits = sum(1 for b in match.block_ids if self._allocator.is_retained(b))
        return needed - match.num_blocks <= self._allocator.num_allocatable - retained_hits

    def allocate(self, seq: Sequence) -> int:
        """Give the sequence its prefix-cache hits and return how many tokens they cover.

        Only the *shared* blocks are taken here. The blocks the sequence has to compute
        itself are allocated chunk by chunk in :meth:`append_slots`, so a long prompt under
        chunked prefill does not have to hold its entire KV footprint from the first step.
        """
        if seq.block_table:
            raise ValueError(
                f"sequence {seq.request_id!r} already holds {len(seq.block_table)} blocks; "
                "free() or preempt() it before allocating again"
            )
        state = _SeqBlocks()
        cached_tokens = 0
        cache = self._prefix_cache
        if cache is not None:
            match = cache.match(
                seq.token_ids, seq.lora_id, max_blocks=self._max_hit_blocks(seq), record=True
            )
            for block_id, digest in zip(match.block_ids, match.hashes, strict=True):
                if self._allocator.is_retained(block_id):
                    self._allocator.adopt(block_id)
                else:
                    self._allocator.incref(block_id)
                cache.acquire(block_id)
                seq.block_table.append(block_id)
                state.hashes.append(digest)
            state.num_published = match.num_blocks
            cached_tokens = match.num_tokens
        seq.num_computed_tokens = cached_tokens
        seq.timing.num_cached_prompt_tokens = min(cached_tokens, seq.num_prompt_tokens)
        self._state[seq.seq_id] = state
        if cached_tokens:
            logger.debug(
                "sequence %s adopted %d cached blocks (%d tokens)",
                seq.request_id,
                len(seq.block_table),
                cached_tokens,
            )
        return cached_tokens

    # -- growth ------------------------------------------------------------------------------

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        """Whether ``num_new_tokens`` more tokens fit without preempting anybody."""
        end = seq.num_computed_tokens + num_new_tokens
        if end > seq.num_tokens:
            return False
        needed = -(-end // self._block_size) - len(seq.block_table)
        return needed <= self._allocator.num_allocatable

    def append_slots(self, seq: Sequence, num_new_tokens: int) -> list[int]:
        """Extend the sequence by ``num_new_tokens`` and return their KV slots.

        Raises :class:`~turboserve.engine.core.kv_cache.OutOfBlocksError` when the pool is
        exhausted; the scheduler catches that and preempts, so it is a control-flow signal
        rather than an error. Call :meth:`can_append` first to avoid the exception in the
        common path.
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be non-negative, got {num_new_tokens}")
        if seq.seq_id not in self._state:
            raise ValueError(
                f"sequence {seq.request_id!r} was never allocated; call allocate() first"
            )
        start = seq.num_computed_tokens
        end = start + num_new_tokens
        if end > seq.num_tokens:
            raise ValueError(
                f"sequence {seq.request_id!r} has {seq.num_tokens} tokens; cannot compute "
                f"up to position {end}"
            )
        needed = -(-end // self._block_size)
        while len(seq.block_table) < needed:
            seq.block_table.append(self._allocator.allocate())
        return self.slot_mapping(seq, start, num_new_tokens)

    def slot_mapping(self, seq: Sequence, start_pos: int, num_tokens: int) -> list[int]:
        """Flat KV slots for absolute token positions ``[start_pos, start_pos + num_tokens)``.

        A slot is ``block_id * block_size + offset`` -- the same flattening
        :meth:`~turboserve.engine.core.kv_cache.KVCache.write` indexes with, so the model
        can scatter a whole step's K/V with one ``index_copy_``.
        """
        if start_pos < 0 or num_tokens < 0:
            raise ValueError("start_pos and num_tokens must be non-negative")
        block_size = self._block_size
        table = seq.block_table
        slots: list[int] = []
        for position in range(start_pos, start_pos + num_tokens):
            index = position // block_size
            if index >= len(table):
                raise ValueError(
                    f"sequence {seq.request_id!r} has no block for position {position} "
                    f"({len(table)} blocks allocated)"
                )
            slots.append(table[index] * block_size + position % block_size)
        return slots

    # -- publication ----------------------------------------------------------------------------

    def publish_computed_blocks(self, seq: Sequence) -> int:
        """Offer every newly completed block of this sequence to the prefix cache.

        Returns how many blocks were newly indexed. Safe to call repeatedly: each block is
        considered exactly once, and a block whose hash is already cached (because another
        sequence computed the same prefix first) is skipped without being retried.
        """
        cache = self._prefix_cache
        state = self._state.get(seq.seq_id)
        if cache is None or state is None:
            return 0
        block_size = self._block_size
        num_full = min(seq.num_computed_tokens // block_size, len(seq.block_table))
        published = 0
        while state.num_published < num_full:
            index = state.num_published
            while len(state.hashes) <= index:
                next_index = len(state.hashes)
                parent = state.hashes[next_index - 1] if next_index else ROOT_HASH
                state.hashes.append(
                    block_hash(parent, seq.block_tokens(next_index, block_size), seq.lora_id)
                )
            if cache.insert(seq.block_table[index], state.hashes[index]):
                published += 1
            state.num_published = index + 1
        return published

    # -- release ---------------------------------------------------------------------------------

    def free(self, seq: Sequence) -> None:
        """Drop the sequence's references to its blocks, after publishing what it computed.

        Full blocks with valid KV go to the prefix cache first, so a finished request leaves
        its prefix behind for the next request that shares it; the allocator then retains
        those blocks instead of freeing them, and reclaims them only under pressure.
        """
        self.publish_computed_blocks(seq)
        if seq.block_table:
            self._allocator.decref_many(seq.block_table)
            seq.block_table.clear()
        self._state.pop(seq.seq_id, None)

    def release(self, seq: Sequence) -> None:
        """Free the blocks and reset progress, leaving the sequence's status alone.

        Used when the scheduler admits a sequence, discovers the step budget cannot fit any
        of its tokens, and has to put it back in the queue untouched.
        """
        self.free(seq)
        seq.num_computed_tokens = 0
        seq.timing.num_cached_prompt_tokens = 0

    def preempt(self, seq: Sequence) -> None:
        """Take the sequence's blocks away and mark it for recompute.

        This is the recompute preemption strategy: nothing is copied to host memory, the
        progress counter goes back to zero and the sequence re-enters the queue. What makes
        it affordable is that :meth:`free` published its blocks on the way out, so a resumed
        sequence usually re-adopts most of them through the prefix cache instead of
        recomputing them.
        """
        self.free(seq)
        seq.reset_for_recompute()

    def reset(self) -> None:
        """Return every block to the pool and empty the prefix cache.

        Both halves are required: emptying only one leaves the cache pointing at blocks the
        allocator considers free, which is exactly the aliasing the three-state invariant
        exists to prevent.
        """
        self._state.clear()
        self._allocator.reset()
        if self._prefix_cache is not None:
            self._prefix_cache.reset()

    # -- invariants ------------------------------------------------------------------------

    def check_invariants(self) -> None:
        """Verify the allocator, the cache, and the relationship between them.

        Linear in the pool size; used by tests and by the engine's debug paths.
        """
        self._allocator.check_invariants()
        cache = self._prefix_cache
        if cache is None:
            return
        cache.check_invariants()
        for block_id in range(self._allocator.num_total):
            if not cache.contains_block(block_id):
                continue
            if self._allocator.is_free(block_id):
                raise ValueError(f"block {block_id} is cached but on the free list")
            if cache.is_evictable(block_id) != self._allocator.is_retained(block_id):
                raise ValueError(
                    f"block {block_id} is evictable in the cache but not retained by the "
                    "allocator (or the other way round)"
                )

    def __repr__(self) -> str:
        return (
            f"BlockManager(block_size={self._block_size}, sequences={len(self._state)}, "
            f"free={self._allocator.num_free}, retained={self._allocator.num_retained}, "
            f"prefix_cache={'on' if self._prefix_cache is not None else 'off'})"
        )
