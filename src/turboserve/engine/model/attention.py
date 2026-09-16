"""Paged attention: write this step's K/V into the cache, attend over the block tables.

A packed varlen batch has no ``[batch, seq]`` axis and each sequence's keys and values
live in a scattered list of fixed-size blocks, so the usual dense
``scaled_dot_product_attention(q, k, v, is_causal=True)`` cannot be used directly. This
module supplies the two pieces that bridge the gap:

* :func:`paged_attention_reference` -- the readable, always-correct implementation. It
  walks the batch one sequence at a time, gathers that sequence's blocks into a
  contiguous ``[context_len, num_kv_heads, head_dim]`` view, expands KV heads for GQA,
  builds the causal mask *offset by the tokens already in the cache*, and calls SDPA. It
  supports any per-sequence query length, which is what makes chunked prefill,
  prefix-cache hits and speculative verification (``k+1`` query tokens) all one code path.
* :class:`PagedAttention` -- the dispatcher. It writes K/V to the cache and then picks a
  backend: the Triton flash-decoding kernel when every sequence contributes exactly one
  query token and the tensors are on CUDA, otherwise the reference.

The causal-offset rule is the subtle part and is worth stating once. ``context_lens[i]``
is the sequence's length *after* this step, and its ``q_len`` query tokens occupy
positions ``[ctx - q_len, ctx)``. Query ``j`` may therefore attend to key indices
``0 .. ctx - q_len + j`` inclusive. With ``q_len == ctx`` (an uncached prefill) this is the
ordinary lower-triangular mask; with ``q_len == 1`` (decode) it is "attend to everything";
with ``0 < q_len < ctx`` (a prefill chunk, or a prefix-cache hit) it is the rectangle plus
a triangle, which is exactly why the same kernel serves all three.

References: Kwon et al., *Efficient Memory Management for Large Language Model Serving with
PagedAttention* (SOSP 2023); Dao et al., *FlashAttention-2* (2023) for the online-softmax
formulation used by the Triton backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

from turboserve.engine.core.types import PAD_BLOCK

if TYPE_CHECKING:
    from turboserve.engine.core.kv_cache import KVCache
    from turboserve.engine.core.types import AttnMetadata

logger = logging.getLogger(__name__)

#: Backend names :meth:`PagedAttention.select_backend` may return.
BackendName = Literal["reference", "triton"]


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Host-side view of one batch's shape, read off the device exactly once per step.

    Every field of :class:`~turboserve.engine.core.types.AttnMetadata` that the reference
    attention path needs is a device tensor, and reading one costs a synchronisation. Doing
    that inside the attention layer would cost ``num_layers`` synchronisations per step --
    on a 28-layer model that is 28 stalls per decoded token. The model builds one
    :class:`BatchPlan` before the first layer and threads it down, so the cost is paid once.

    Block tables are trimmed here to the number of blocks each sequence actually needs,
    and the trim is validated: a :data:`~turboserve.engine.core.types.PAD_BLOCK` inside the
    needed range means the scheduler and the batch disagree, which must fail loudly rather
    than read another sequence's KV.
    """

    query_offsets: tuple[int, ...]
    """``num_seqs + 1`` cumulative token offsets (``query_start_loc`` on the host)."""

    context_lens: tuple[int, ...]
    """Per-sequence attended length after this step."""

    block_ids: tuple[tuple[int, ...], ...]
    """Per-sequence block table, trimmed to ``ceil(context_len / block_size)`` entries."""

    block_size: int

    @property
    def num_seqs(self) -> int:
        """Sequences described by this plan."""
        return len(self.context_lens)

    @property
    def query_lens(self) -> tuple[int, ...]:
        """Per-sequence number of query tokens contributed to this step."""
        return tuple(
            self.query_offsets[i + 1] - self.query_offsets[i] for i in range(self.num_seqs)
        )

    @classmethod
    def from_metadata(cls, meta: AttnMetadata, *, block_size: int) -> BatchPlan:
        """Read ``meta`` onto the host and validate the block tables against it."""
        offsets = tuple(int(x) for x in meta.query_start_loc.tolist())
        contexts = tuple(int(x) for x in meta.context_lens.tolist())
        rows = meta.block_tables.tolist()
        if len(offsets) != len(contexts) + 1:
            raise ValueError(
                f"query_start_loc has {len(offsets)} entries for {len(contexts)} sequences"
            )
        if len(rows) != len(contexts):
            raise ValueError(f"block_tables has {len(rows)} rows for {len(contexts)} sequences")
        block_ids: list[tuple[int, ...]] = []
        for seq, ctx in enumerate(contexts):
            needed = -(-ctx // block_size)
            row = [int(b) for b in rows[seq][:needed]]
            if len(row) != needed or any(b == PAD_BLOCK for b in row):
                raise ValueError(
                    f"sequence {seq} needs {needed} blocks for {ctx} tokens but its block "
                    f"table supplies {[b for b in row if b != PAD_BLOCK]}"
                )
            block_ids.append(tuple(row))
        return cls(
            query_offsets=offsets,
            context_lens=contexts,
            block_ids=tuple(block_ids),
            block_size=block_size,
        )


def gather_sequence_kv(
    cache: torch.Tensor, block_ids: tuple[int, ...], context_len: int
) -> torch.Tensor:
    """Gather one sequence's ``[context_len, num_kv_heads, head_dim]`` KV from its blocks.

    ``cache`` is one layer's ``[num_blocks, block_size, num_kv_heads, head_dim]`` tensor.
    The gather is a single ``index_select`` over the block axis followed by a reshape, so
    the copy is one contiguous kernel per sequence rather than one per block.
    """
    if not block_ids:
        return cache.new_empty((0, cache.shape[2], cache.shape[3]))
    index = torch.tensor(block_ids, dtype=torch.long, device=cache.device)
    gathered = cache.index_select(0, index)
    flat = gathered.reshape(-1, cache.shape[2], cache.shape[3])
    return flat[:context_len]


def causal_block_mask(query_len: int, context_len: int, *, device: torch.device) -> torch.Tensor:
    """``[query_len, context_len]`` boolean mask; ``True`` where attention is allowed.

    Query ``j`` sits at absolute position ``context_len - query_len + j`` and may attend to
    every key up to and including its own position. See the module docstring for why this
    single expression covers prefill, chunked prefill, prefix-cache hits and decode.
    """
    queries = torch.arange(query_len, device=device).unsqueeze(1)
    keys = torch.arange(context_len, device=device).unsqueeze(0)
    return keys <= (context_len - query_len + queries)


def paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: AttnMetadata,
    *,
    scale: float,
    plan: BatchPlan | None = None,
) -> torch.Tensor:
    """Attention over the paged cache for a packed varlen batch.

    Args:
        q: ``[num_tokens, num_heads, head_dim]`` queries, already rotated, in batch order.
        k_cache: one layer's ``[num_blocks, block_size, num_kv_heads, head_dim]`` keys,
            with this step's keys already written.
        v_cache: the matching values.
        meta: the batch description; ``context_lens`` must already include this step.
        scale: softmax scale, normally ``head_dim ** -0.5``.
        plan: a precomputed :class:`BatchPlan` to avoid re-reading ``meta`` from the device.

    Returns:
        ``[num_tokens, num_heads, head_dim]`` attention output in the same token order.
    """
    block_size = int(k_cache.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    num_heads = int(q.shape[1])
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"{num_heads} query heads is not a multiple of {num_kv_heads} KV heads")
    group = num_heads // num_kv_heads
    if plan is None:
        plan = BatchPlan.from_metadata(meta, block_size=block_size)

    out = torch.empty_like(q)
    for seq in range(plan.num_seqs):
        start = plan.query_offsets[seq]
        end = plan.query_offsets[seq + 1]
        q_len = end - start
        if q_len == 0:
            continue
        ctx = plan.context_lens[seq]
        if ctx < q_len:
            raise ValueError(
                f"sequence {seq} attends to {ctx} tokens but contributes {q_len} queries"
            )
        keys = gather_sequence_kv(k_cache, plan.block_ids[seq], ctx)
        values = gather_sequence_kv(v_cache, plan.block_ids[seq], ctx)
        if group > 1:
            keys = keys.repeat_interleave(group, dim=1)
            values = values.repeat_interleave(group, dim=1)
        mask = causal_block_mask(q_len, ctx, device=q.device)
        # SDPA wants [batch, heads, seq, dim]; the batch axis is the singleton sequence.
        attended = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1).unsqueeze(0),
            keys.transpose(0, 1).unsqueeze(0).to(q.dtype),
            values.transpose(0, 1).unsqueeze(0).to(q.dtype),
            attn_mask=mask.unsqueeze(0).unsqueeze(0),
            scale=scale,
        )
        out[start:end] = attended.squeeze(0).transpose(0, 1)
    return out


class PagedAttention(nn.Module):
    """Cache write plus backend dispatch for one attention layer.

    Stateless apart from its shape parameters: the KV lives in the engine-owned
    :class:`~turboserve.engine.core.kv_cache.KVCache` that is passed in, so a model can be
    run against different cache pools (the target model's and a speculative draft's)
    without rebuilding the module.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float | None = None,
        prefer_triton: bool = True,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be a multiple of num_kv_heads={num_kv_heads}"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = float(head_dim) ** -0.5 if scale is None else float(scale)
        self.prefer_triton = prefer_triton

    def select_backend(
        self, q: torch.Tensor, meta: AttnMetadata, *, block_size: int | None = None
    ) -> BackendName:
        """Choose the kernel for this batch.

        The Triton kernel is a *decode* kernel: it assumes one query token per sequence and
        therefore no intra-batch causal masking. Anything else -- a prefill, a chunked
        prefill, a speculative verify step -- goes to the reference path, and so does every
        CPU batch, because Triton has no CPU backend here.
        """
        if not self.prefer_triton or q.device.type != "cuda":
            return "reference"
        from turboserve.engine.model.triton_attention import can_use_triton_decode

        usable = can_use_triton_decode(
            q, meta, num_kv_groups=self.num_kv_groups, block_size=block_size
        )
        return "triton" if usable else "reference"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: KVCache,
        layer_idx: int,
        meta: AttnMetadata,
        *,
        plan: BatchPlan | None = None,
    ) -> torch.Tensor:
        """Write ``k``/``v`` into the cache at ``meta.slot_mapping``, then attend.

        The write happens before the read on purpose: every query token must be able to
        attend to its own key, and doing it through the cache (rather than concatenating
        the new K/V onto the gathered history) means prefill and decode read their context
        through exactly the same indirection, so a block-table bug cannot hide in one path.
        """
        kv_cache.write(layer_idx, meta.slot_mapping, k, v)
        k_cache, v_cache = kv_cache.layer(layer_idx)
        backend = self.select_backend(q, meta, block_size=int(k_cache.shape[1]))
        if backend == "triton":
            from turboserve.engine.model.triton_attention import paged_attention_decode_triton

            return paged_attention_decode_triton(
                q, k_cache, v_cache, meta, scale=self.scale, num_kv_groups=self.num_kv_groups
            )
        return paged_attention_reference(q, k_cache, v_cache, meta, scale=self.scale, plan=plan)

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, scale={self.scale:.6f}"
        )
