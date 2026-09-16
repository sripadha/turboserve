"""Paged KV-cache storage and the reference-counted block allocator underneath it.

The engine never allocates KV memory per request. It allocates a fixed pool of equally
sized blocks once, at startup, and hands out block ids; a sequence's KV is therefore a
list of block ids (its *block table*) rather than a contiguous tensor. That is what makes
continuous batching possible without either reserving ``max_model_len`` per slot or
copying KV around when a sequence grows, and it is what lets two requests that share a
prompt prefix share the blocks holding that prefix's KV.

This module owns the two lowest layers of that scheme:

* :class:`BlockAllocator` -- who owns which block, with reference counts so a shared
  prefix block is only reclaimed once its last user is gone.
* :class:`KVCache` -- the tensors themselves, addressed by flat *slot* index
  ``block_id * block_size + offset``.

The policy layers on top (which blocks a sequence gets, how prefix hashes map to blocks,
when to preempt) live in ``prefix_cache.py`` and the block manager, and reach this module
only through the small protocol :class:`BlockRecycler`.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from turboserve.engine.core.types import resolve_dtype

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "BlockAllocator",
    "BlockAllocatorError",
    "BlockRecycler",
    "InvalidBlockError",
    "KVCache",
    "OutOfBlocksError",
]


class BlockAllocatorError(RuntimeError):
    """Base class for every allocator failure, so callers can catch one type."""


class OutOfBlocksError(BlockAllocatorError):
    """The KV pool is exhausted: no free block and nothing reclaimable.

    This is an expected condition, not a bug: the scheduler catches it and preempts a
    running sequence (recompute mode) rather than failing the request.
    """


class InvalidBlockError(BlockAllocatorError):
    """A block id is out of range, or a block was released more times than it was taken.

    Raised eagerly because the alternative -- a block silently appearing twice on the free
    list -- hands the same KV memory to two tenants, which is a correctness *and* an
    isolation bug that would surface far from its cause.
    """


@runtime_checkable
class BlockRecycler(Protocol):
    """Hook that lets a prefix cache keep freed blocks alive for a while.

    Without a recycler a block whose reference count reaches zero goes straight back on
    the free list and its contents are forfeit. The prefix cache wants the opposite: a
    block holding a hashed, fully computed prefix is *more* valuable once nobody is using
    it, because the next request with the same prefix can adopt it and skip the prefill.

    So the allocator offers the block to the recycler first. A recycler that answers
    ``True`` to :meth:`on_zero_refs` takes custody of the block: it stays out of the free
    list, counted as *retained*, until either the cache hands it back through
    :meth:`reclaim` (because the allocator ran dry) or a new user adopts it via
    :meth:`BlockAllocator.adopt`.
    """

    def on_zero_refs(self, block_id: int) -> bool:
        """Offer a just-freed block to the cache. Return ``True`` to retain it."""
        ...

    def reclaim(self, num_blocks: int) -> Sequence[int]:
        """Give up to ``num_blocks`` retained blocks back, least valuable first."""
        ...


class BlockAllocator:
    """Reference-counted free list over a fixed pool of KV blocks.

    Every operation is O(1) (amortised, and excluding the recycler's own policy): the free
    list is a deque of ids and the reference counts are a flat list indexed by block id.
    A block is in exactly one of three states at all times, which :meth:`check_invariants`
    verifies:

    ``free``
        reference count 0, on the free list, contents meaningless.
    ``in use``
        reference count > 0, off the free list, owned by one or more sequences.
    ``retained``
        reference count 0, off the free list, held by the :class:`BlockRecycler` because
        its contents are a cached prefix.

    The free list is FIFO rather than LIFO on purpose: when there is no recycler, reusing
    the least recently freed block gives a cached-prefix layer above the longest possible
    window in which to adopt a block before it is overwritten.
    """

    __slots__ = ("_block_size", "_free", "_num_blocks", "_recycler", "_refs", "_retained")

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        recycler: BlockRecycler | None = None,
    ) -> None:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._num_blocks = num_blocks
        self._block_size = block_size
        self._recycler = recycler
        self._refs: list[int] = [0] * num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._retained: set[int] = set()

    # -- introspection ---------------------------------------------------------------

    @property
    def num_total(self) -> int:
        """Size of the pool in blocks."""
        return self._num_blocks

    @property
    def num_free(self) -> int:
        """Blocks that can be handed out without reclaiming anything."""
        return len(self._free)

    @property
    def num_retained(self) -> int:
        """Blocks held by the recycler: free of users, but still holding cached KV."""
        return len(self._retained)

    @property
    def num_allocatable(self) -> int:
        """Blocks :meth:`allocate` could serve right now, retained ones included."""
        return len(self._free) + len(self._retained)

    @property
    def num_in_use(self) -> int:
        """Blocks with at least one reference."""
        return self._num_blocks - len(self._free) - len(self._retained)

    @property
    def block_size(self) -> int:
        """Tokens per block."""
        return self._block_size

    def ref_count(self, block_id: int) -> int:
        """Current reference count of ``block_id``."""
        self._check_id(block_id)
        return self._refs[block_id]

    def is_free(self, block_id: int) -> bool:
        """Whether the block is on the free list (contents may be overwritten)."""
        self._check_id(block_id)
        return self._refs[block_id] == 0 and block_id not in self._retained

    def is_retained(self, block_id: int) -> bool:
        """Whether the block is unused but held by the recycler."""
        self._check_id(block_id)
        return block_id in self._retained

    def stats(self) -> dict[str, int | float]:
        """Snapshot for logging and the engine's ``stats()`` endpoint."""
        used = self.num_in_use
        return {
            "num_total": self._num_blocks,
            "num_free": len(self._free),
            "num_retained": len(self._retained),
            "num_in_use": used,
            "utilization": used / self._num_blocks if self._num_blocks else 0.0,
        }

    # -- allocation ------------------------------------------------------------------

    def allocate(self) -> int:
        """Take one block, with a reference count of 1.

        Raises :class:`OutOfBlocksError` when the pool is exhausted and the recycler has
        nothing left to give back.
        """
        if not self._free:
            self._reclaim(1)
        if not self._free:
            raise OutOfBlocksError(
                f"no free KV block: {self.num_in_use}/{self._num_blocks} in use, "
                f"{len(self._retained)} retained by the prefix cache"
            )
        block_id = self._free.popleft()
        self._refs[block_id] = 1
        return block_id

    def allocate_many(self, count: int) -> list[int]:
        """Take ``count`` blocks, or none at all.

        All-or-nothing because a partial allocation would leave the caller holding blocks
        for a sequence it cannot admit; the scheduler would then have to unwind by hand.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > len(self._free):
            self._reclaim(count - len(self._free))
        if count > len(self._free):
            raise OutOfBlocksError(
                f"cannot allocate {count} blocks: only {len(self._free)} free after "
                f"reclaiming ({len(self._retained)} still retained, "
                f"{self.num_in_use}/{self._num_blocks} in use)"
            )
        return [self.allocate() for _ in range(count)]

    def adopt(self, block_id: int) -> int:
        """Take a reference on a retained (cached) block, pulling it out of the recycler.

        This is how a prefix-cache hit is honoured: the cache looked up a block by content
        hash, and the new sequence adopts it instead of recomputing its KV. Raises
        :class:`InvalidBlockError` if the block is free (its contents are already gone) or
        already in use (:meth:`incref` is the right call then).
        """
        self._check_id(block_id)
        if block_id not in self._retained:
            raise InvalidBlockError(
                f"block {block_id} is not retained (refs={self._refs[block_id]}); "
                "only a cached, unreferenced block can be adopted"
            )
        self._retained.discard(block_id)
        self._refs[block_id] = 1
        return block_id

    def incref(self, block_id: int) -> int:
        """Add a reference to an in-use block and return the new count.

        Used when a second sequence starts sharing an already-live prefix block.
        """
        self._check_id(block_id)
        if self._refs[block_id] == 0:
            raise InvalidBlockError(
                f"cannot incref block {block_id}: it has no owner "
                f"({'retained' if block_id in self._retained else 'free'}); "
                "use adopt() for a retained block"
            )
        self._refs[block_id] += 1
        return self._refs[block_id]

    def incref_many(self, block_ids: Iterable[int]) -> None:
        """Add a reference to each of ``block_ids``."""
        for block_id in block_ids:
            self.incref(block_id)

    def decref(self, block_id: int) -> int:
        """Drop a reference and return the new count, freeing the block at zero.

        At zero the block is offered to the recycler first (see :class:`BlockRecycler`);
        if the recycler declines, or there is none, the block goes back on the free list.
        Dropping a reference that was never taken raises :class:`InvalidBlockError`.
        """
        self._check_id(block_id)
        count = self._refs[block_id]
        if count == 0:
            raise InvalidBlockError(
                f"double free of block {block_id}: it is already "
                f"{'retained' if block_id in self._retained else 'free'}"
            )
        count -= 1
        self._refs[block_id] = count
        if count == 0:
            self._release(block_id)
        return count

    def decref_many(self, block_ids: Iterable[int]) -> None:
        """Drop a reference on each of ``block_ids``."""
        for block_id in block_ids:
            self.decref(block_id)

    def release_retained(self, block_ids: Iterable[int]) -> int:
        """Return retained blocks to the free list; used by the cache when it evicts.

        Returns the number of blocks actually released. Ids that are not retained are
        ignored, so an eviction that races with an adoption is a no-op rather than an
        error.
        """
        released = 0
        for block_id in block_ids:
            self._check_id(block_id)
            if block_id in self._retained:
                self._retained.discard(block_id)
                self._free.append(block_id)
                released += 1
        return released

    def reset(self) -> None:
        """Drop every reference and return the whole pool to the free list."""
        self._refs = [0] * self._num_blocks
        self._free = deque(range(self._num_blocks))
        self._retained.clear()

    # -- invariants ------------------------------------------------------------------

    def check_invariants(self) -> None:
        """Raise :class:`BlockAllocatorError` if the three-state invariant is broken.

        Linear in the pool size, so it is a test and debugging tool rather than something
        the scheduler calls per step.
        """
        free_set = set(self._free)
        if len(free_set) != len(self._free):
            raise BlockAllocatorError("a block appears twice on the free list")
        overlap = free_set & self._retained
        if overlap:
            raise BlockAllocatorError(f"blocks {sorted(overlap)} are both free and retained")
        for block_id, count in enumerate(self._refs):
            if count < 0:
                raise BlockAllocatorError(f"block {block_id} has a negative reference count")
            if count == 0 and block_id not in free_set and block_id not in self._retained:
                raise BlockAllocatorError(f"block {block_id} is unreferenced but leaked")
            if count > 0 and (block_id in free_set or block_id in self._retained):
                raise BlockAllocatorError(f"block {block_id} is referenced but also reclaimable")

    # -- internals -------------------------------------------------------------------

    def _check_id(self, block_id: int) -> None:
        if not 0 <= block_id < self._num_blocks:
            raise InvalidBlockError(
                f"block id {block_id} out of range for a pool of {self._num_blocks} blocks"
            )

    def _release(self, block_id: int) -> None:
        """Offer a now-unreferenced block to the recycler, else free it."""
        if self._recycler is not None and self._recycler.on_zero_refs(block_id):
            self._retained.add(block_id)
            return
        self._free.append(block_id)

    def _reclaim(self, num_blocks: int) -> None:
        """Ask the recycler for ``num_blocks`` retained blocks and free them."""
        if self._recycler is None or not self._retained:
            return
        wanted = min(num_blocks, len(self._retained))
        given = list(self._recycler.reclaim(wanted))
        released = self.release_retained(given)
        if released != len(given):
            logger.debug(
                "recycler returned %d blocks of which %d were still retained",
                len(given),
                released,
            )

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(num_total={self._num_blocks}, block_size={self._block_size}, "
            f"num_free={len(self._free)}, num_retained={len(self._retained)})"
        )


class KVCache:
    """The paged key/value tensors, one pair per transformer layer.

    Layout per layer is ``(num_blocks, block_size, num_kv_heads, head_dim)``. Writes
    address the cache by *slot*, the flat index ``block_id * block_size + offset``, which
    is exactly the last dimension collapse of that shape -- so a write is a single
    ``index_copy_`` into a flattened view with no gather, no per-sequence Python loop and
    no synchronisation.

    Allocation is lazy: constructing a ``KVCache`` is cheap and side-effect free, and the
    (potentially many gigabytes of) tensors appear on first use or on an explicit
    :meth:`allocate`. That matters because the engine sizes the pool by first profiling
    free device memory with a dummy forward pass, and because tests construct caches to
    ask about :attr:`bytes_per_block` without ever touching a device.
    """

    __slots__ = (
        "_block_size",
        "_device",
        "_dtype",
        "_head_dim",
        "_k",
        "_k_flat",
        "_num_blocks",
        "_num_kv_heads",
        "_num_layers",
        "_v",
        "_v_flat",
    )

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype | str = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("num_blocks", num_blocks),
            ("block_size", block_size),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        self._num_layers = num_layers
        self._num_blocks = num_blocks
        self._block_size = block_size
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype = resolve_dtype(dtype)
        self._device = torch.device(device)
        self._k: list[torch.Tensor] = []
        self._v: list[torch.Tensor] = []
        self._k_flat: list[torch.Tensor] = []
        self._v_flat: list[torch.Tensor] = []

    # -- shape and size ----------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        """Number of transformer layers this cache serves."""
        return self._num_layers

    @property
    def num_blocks(self) -> int:
        """Blocks in the pool (the same pool is shared by every layer)."""
        return self._num_blocks

    @property
    def block_size(self) -> int:
        """Tokens per block."""
        return self._block_size

    @property
    def num_kv_heads(self) -> int:
        """Key/value heads per layer (fewer than query heads under GQA)."""
        return self._num_kv_heads

    @property
    def head_dim(self) -> int:
        """Size of one attention head."""
        return self._head_dim

    @property
    def dtype(self) -> torch.dtype:
        """Element type of the cached K/V."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device the tensors are (or will be) allocated on."""
        return self._device

    @property
    def num_slots(self) -> int:
        """Total addressable token slots, ``num_blocks * block_size``."""
        return self._num_blocks * self._block_size

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Per-layer tensor shape, ``(num_blocks, block_size, num_kv_heads, head_dim)``."""
        return (self._num_blocks, self._block_size, self._num_kv_heads, self._head_dim)

    @property
    def bytes_per_block_per_layer(self) -> int:
        """Bytes one block occupies in one layer, keys and values together."""
        return 2 * self._block_size * self._num_kv_heads * self._head_dim * self._dtype.itemsize

    @property
    def bytes_per_block(self) -> int:
        """Bytes one block costs across *all* layers.

        Block ids are shared by every layer -- a sequence has one block table, not one per
        layer -- so the unit of capacity planning is a block's total footprint. This is
        the number the engine divides free memory by when sizing the pool.
        """
        return self._num_layers * self.bytes_per_block_per_layer

    @property
    def total_bytes(self) -> int:
        """Bytes the whole cache occupies once allocated."""
        return self._num_blocks * self.bytes_per_block

    @staticmethod
    def bytes_per_block_for(
        *,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype | str,
    ) -> int:
        """Footprint of one block, without constructing a cache."""
        itemsize = resolve_dtype(dtype).itemsize
        return num_layers * 2 * block_size * num_kv_heads * head_dim * itemsize

    @classmethod
    def blocks_for_memory(
        cls,
        available_bytes: int | float,
        *,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype | str,
    ) -> int:
        """How many blocks fit in ``available_bytes`` (floor, never negative).

        The engine calls this after profiling: ``available = total * gpu_memory_utilization
        - bytes_already_used_by_weights_and_activations``. Rounding down matters -- one
        block too many is an out-of-memory error at the worst possible moment, under load.
        """
        per_block = cls.bytes_per_block_for(
            num_layers=num_layers,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        if available_bytes <= 0:
            return 0
        return int(available_bytes) // per_block

    @staticmethod
    def slot_index(block_id: int, offset: int, block_size: int) -> int:
        """Flat slot of the ``offset``-th token of ``block_id``."""
        return block_id * block_size + offset

    # -- storage -----------------------------------------------------------------------

    @property
    def is_allocated(self) -> bool:
        """Whether the tensors exist yet."""
        return bool(self._k)

    def allocate(self) -> None:
        """Materialise the tensors; a no-op if they already exist."""
        if self._k:
            return
        for _ in range(self._num_layers):
            k = torch.zeros(self.shape, dtype=self._dtype, device=self._device)
            v = torch.zeros(self.shape, dtype=self._dtype, device=self._device)
            self._k.append(k)
            self._v.append(v)
            # Flat views over the same storage: (num_slots, num_kv_heads, head_dim). Kept
            # so that write() does not rebuild a view on every layer of every step.
            self._k_flat.append(k.view(self.num_slots, self._num_kv_heads, self._head_dim))
            self._v_flat.append(v.view(self.num_slots, self._num_kv_heads, self._head_dim))
        logger.debug(
            "allocated KV cache: %d layers x %s %s on %s (%d bytes)",
            self._num_layers,
            self.shape,
            self._dtype,
            self._device,
            self.total_bytes,
        )

    def free(self) -> None:
        """Drop the tensors, releasing the memory once no other reference remains."""
        self._k.clear()
        self._v.clear()
        self._k_flat.clear()
        self._v_flat.clear()

    @property
    def k(self) -> list[torch.Tensor]:
        """Per-layer key tensors, allocating on first access."""
        self.allocate()
        return self._k

    @property
    def v(self) -> list[torch.Tensor]:
        """Per-layer value tensors, allocating on first access."""
        self.allocate()
        return self._v

    def layer(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(k, v)`` for one layer, both shaped :attr:`shape`."""
        self._check_layer(layer)
        self.allocate()
        return self._k[layer], self._v[layer]

    def write(
        self,
        layer: int,
        slot_mapping: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Scatter one step's new K/V into the cache.

        ``slot_mapping`` is ``[num_tokens]`` int64 flat slots, ``k``/``v`` are
        ``[num_tokens, num_kv_heads, head_dim]`` in the same token order -- exactly the
        layout the attention layer already has after its projections, so no transpose is
        needed. Every slot must be valid: the scheduler assigns a slot to every token it
        schedules, and a ``-1`` here would mean the batch and the block tables disagree.
        """
        self._check_layer(layer)
        if slot_mapping.dtype != torch.long:
            raise ValueError(f"slot_mapping must be int64, got {slot_mapping.dtype}")
        if slot_mapping.dim() != 1:
            raise ValueError(f"slot_mapping must be 1-D, got shape {tuple(slot_mapping.shape)}")
        expected = (slot_mapping.shape[0], self._num_kv_heads, self._head_dim)
        for name, tensor in (("k", k), ("v", v)):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected} to match slot_mapping, "
                    f"got {tuple(tensor.shape)}"
                )
        self.allocate()
        slots = slot_mapping.to(self._device, non_blocking=True)
        self._k_flat[layer].index_copy_(0, slots, k.to(device=self._device, dtype=self._dtype))
        self._v_flat[layer].index_copy_(0, slots, v.to(device=self._device, dtype=self._dtype))

    def read(self, layer: int, slot_mapping: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V back out of the cache for the given slots.

        The inverse of :meth:`write`, used by the reference attention path, by the
        speculative-decoding rollback checks and by the round-trip tests.
        """
        self._check_layer(layer)
        self.allocate()
        slots = slot_mapping.to(self._device)
        return self._k_flat[layer].index_select(0, slots), self._v_flat[layer].index_select(
            0, slots
        )

    def block(self, layer: int, block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(k, v)`` views of one whole block in one layer."""
        self._check_layer(layer)
        if not 0 <= block_id < self._num_blocks:
            raise ValueError(f"block id {block_id} out of range for {self._num_blocks} blocks")
        self.allocate()
        return self._k[layer][block_id], self._v[layer][block_id]

    def zero_(self) -> None:
        """Zero every layer's K/V; used by tests to make stale data visible."""
        self.allocate()
        for k, v in zip(self._k, self._v, strict=True):
            k.zero_()
            v.zero_()

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self._num_layers:
            raise ValueError(f"layer {layer} out of range for {self._num_layers} layers")

    def __repr__(self) -> str:
        return (
            f"KVCache(num_layers={self._num_layers}, num_blocks={self._num_blocks}, "
            f"block_size={self._block_size}, num_kv_heads={self._num_kv_heads}, "
            f"head_dim={self._head_dim}, dtype={self._dtype}, device={self._device}, "
            f"allocated={self.is_allocated})"
        )
