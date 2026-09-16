"""Triton flash-decoding kernel for the paged KV cache (one query token per sequence).

Decode attention is *bandwidth* bound, not compute bound: every step reads the whole KV
context and multiplies it against a single query vector per head, an arithmetic intensity
of roughly one FLOP per byte. The thing that matters is therefore reading each block
exactly once, in coalesced order, and never materialising a ``[num_heads, context_len]``
score matrix in memory. That is what this kernel does, using the FlashAttention-2 online
softmax (Dao, 2023) applied per block, in the flash-decoding arrangement (Dao et al., 2023)
of one program per ``(sequence, kv_head)`` pair looping over that sequence's blocks.

Deliberate non-use of tensor cores: the scores and the value accumulation are computed
with broadcast multiply-and-reduce rather than ``tl.dot``. A GQA group is 4 to 8 query
heads wide on the models this engine targets, far below the 16x16 minimum tile ``tl.dot``
accepts, so the matrix path would require padding the group dimension to 16 -- more than
double the work -- to use hardware that a bandwidth-bound kernel cannot exploit anyway.
Keeping one code path also keeps the kernel that ships identical to the kernel the GPU test
exercises.

The kernel is a *fast path*, never a requirement: :func:`can_use_triton_decode` is
conservative, and :class:`~turboserve.engine.model.attention.PagedAttention` falls back to
:func:`~turboserve.engine.model.attention.paged_attention_reference` whenever it says no --
on CPU, on any batch containing a prefill or a speculative verify step, on an unsupported
dtype, and on shapes whose per-program tile would spill registers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from turboserve.engine.core.types import AttnMetadata

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whichever branch the machine supports
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only installs have no triton
    TRITON_AVAILABLE = False
    triton = None
    tl = None

#: Element types the kernel accepts. Everything is accumulated in fp32 regardless.
SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16, torch.float32)

#: Upper bound on ``group_pow2 * block_size * head_dim_pow2``, the size of the largest
#: per-program intermediate. Above it the kernel spills registers and the reference path
#: (which lets cuBLAS tile the problem) is the better choice; multi-query models with a
#: 32-wide GQA group land here.
MAX_TILE_ELEMENTS = 16384


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, (value - 1)).bit_length()


if TRITON_AVAILABLE:  # pragma: no branch - a plain import-time guard

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=1, num_stages=1),
            triton.Config({}, num_warps=2, num_stages=2),
            triton.Config({}, num_warps=4, num_stages=2),
        ],
        key=["GROUP_POW2", "HEAD_DIM_POW2", "BLOCK_SIZE"],
    )
    @triton.jit
    def _paged_decode_kernel(  # noqa: PLR0913 - a kernel signature is flat by necessity
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        block_tables_ptr,
        context_lens_ptr,
        scale,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_ot,
        stride_oh,
        stride_od,
        stride_bt_seq,
        stride_bt_blk,
        GROUP: tl.constexpr,
        GROUP_POW2: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_DIM_POW2: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program attends one sequence's single query token for one KV head.

        The program holds the whole GQA group (``GROUP`` query heads) in registers, so the
        block of keys and values it loads is reused ``GROUP`` times before being evicted --
        the point of grouping the heads this way rather than launching one program per
        query head.
        """
        seq = tl.program_id(0)
        kv_head = tl.program_id(1)
        ctx = tl.load(context_lens_ptr + seq)

        head_offs = tl.arange(0, GROUP_POW2)
        head_mask = head_offs < GROUP
        dim_offs = tl.arange(0, HEAD_DIM_POW2)
        dim_mask = dim_offs < HEAD_DIM
        q_heads = kv_head * GROUP + head_offs

        q = tl.load(
            q_ptr + seq * stride_qt + q_heads[:, None] * stride_qh + dim_offs[None, :] * stride_qd,
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q = q * scale

        m_i = tl.full((GROUP_POW2,), float("-inf"), tl.float32)
        l_i = tl.zeros((GROUP_POW2,), tl.float32)
        acc = tl.zeros((GROUP_POW2, HEAD_DIM_POW2), tl.float32)

        slot_offs = tl.arange(0, BLOCK_SIZE)
        num_blocks = tl.cdiv(ctx, BLOCK_SIZE)
        for block in range(0, num_blocks):
            block_id = tl.load(block_tables_ptr + seq * stride_bt_seq + block * stride_bt_blk)
            positions = block * BLOCK_SIZE + slot_offs
            valid = positions < ctx
            k = tl.load(
                k_ptr
                + block_id * stride_kb
                + slot_offs[:, None] * stride_ks
                + kv_head * stride_kh
                + dim_offs[None, :] * stride_kd,
                mask=valid[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr
                + block_id * stride_vb
                + slot_offs[:, None] * stride_vs
                + kv_head * stride_vh
                + dim_offs[None, :] * stride_vd,
                mask=valid[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
            scores = tl.where(valid[None, :], scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
            m_i = m_new

        out = acc / l_i[:, None]
        tl.store(
            out_ptr
            + seq * stride_ot
            + q_heads[:, None] * stride_oh
            + dim_offs[None, :] * stride_od,
            out,
            mask=head_mask[:, None] & dim_mask[None, :],
        )


def triton_decode_tile_elements(*, num_kv_groups: int, head_dim: int, block_size: int) -> int:
    """Size of the largest per-program intermediate, used by the dispatch guard."""
    return _next_power_of_two(num_kv_groups) * block_size * _next_power_of_two(head_dim)


def can_use_triton_decode(
    q: torch.Tensor,
    meta: AttnMetadata,
    *,
    num_kv_groups: int = 1,
    block_size: int | None = None,
) -> bool:
    """Whether :func:`paged_attention_decode_triton` may serve this batch.

    Conservative by design. Every ``False`` here costs a little throughput; a wrong
    ``True`` produces silently incorrect attention, so each condition is checked rather
    than assumed:

    * Triton is importable and the tensors are on CUDA.
    * Every sequence contributes exactly one query token (``meta.is_decode_only``) *and*
      the token count matches the sequence count, so there is no intra-batch causal mask.
    * Every sequence has at least one cached token -- a zero-length context would divide
      by a zero softmax denominator.
    * The dtype is one the kernel loads, and the per-program tile fits
      :data:`MAX_TILE_ELEMENTS`.
    """
    if not TRITON_AVAILABLE or q.device.type != "cuda":
        return False
    if q.dtype not in SUPPORTED_DTYPES:
        return False
    if not meta.is_decode_only or meta.num_tokens != meta.num_seqs:
        return False
    if meta.num_seqs == 0 or meta.max_context_len < 1:
        return False
    if block_size is not None:
        tile = triton_decode_tile_elements(
            num_kv_groups=num_kv_groups, head_dim=int(q.shape[-1]), block_size=block_size
        )
        if tile > MAX_TILE_ELEMENTS:
            return False
    return True


def paged_attention_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: AttnMetadata,
    *,
    scale: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """Run the decode kernel and return ``[num_seqs, num_heads, head_dim]`` output.

    Args:
        q: ``[num_seqs, num_heads, head_dim]`` queries, one token per sequence.
        k_cache: one layer's ``[num_blocks, block_size, num_kv_heads, head_dim]`` keys.
        v_cache: the matching values.
        meta: batch description; ``context_lens`` must already count this step's token.
        scale: softmax scale, normally ``head_dim ** -0.5``.
        num_kv_groups: query heads per KV head.

    Raises:
        RuntimeError: if Triton is unavailable. Callers dispatch through
            :func:`can_use_triton_decode`, so reaching this means the guard was skipped.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "triton is not importable; use paged_attention_reference on this platform"
        )
    num_seqs, num_heads, head_dim = (int(x) for x in q.shape)
    block_size = int(k_cache.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    if num_heads != num_kv_heads * num_kv_groups:
        raise ValueError(
            f"q has {num_heads} heads but the cache has {num_kv_heads} KV heads "
            f"x {num_kv_groups} groups"
        )
    if meta.num_seqs != num_seqs:
        raise ValueError(f"metadata describes {meta.num_seqs} sequences, q has {num_seqs}")

    out = torch.empty_like(q)
    context_lens = meta.context_lens.to(q.device, dtype=torch.int32)
    block_tables = meta.block_tables.to(q.device, dtype=torch.int32)
    grid = (num_seqs, num_kv_heads)
    kernel: Any = _paged_decode_kernel
    kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        block_tables,
        context_lens,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        GROUP=num_kv_groups,
        GROUP_POW2=_next_power_of_two(num_kv_groups),
        HEAD_DIM=head_dim,
        HEAD_DIM_POW2=_next_power_of_two(head_dim),
        BLOCK_SIZE=block_size,
    )
    return out
