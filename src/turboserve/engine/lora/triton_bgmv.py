"""Triton BGMV kernels: one adapter matrix-vector product per token, gathered by slot.

During decode every sequence contributes exactly one token, so the LoRA delta for a
projection is ``B[slot] @ (A[slot] @ x)`` -- a pair of matrix-*vector* products whose
operands are chosen per token. Punica (Chen et al., 2023,
*Punica: Multi-Tenant LoRA Serving*, https://arxiv.org/abs/2310.18547) calls this pattern
BGMV, Batched Gather Matrix-Vector multiplication, and the reason it deserves a kernel is
memory traffic rather than arithmetic: the grouped (SGMV) path of
:mod:`turboserve.engine.lora.layers` launches one pair of GEMMs per *active adapter*, and
with one token per adapter those GEMMs are all shape ``[1, k]`` -- dozens of kernel
launches, each reading a rank-sized slice, for a few thousand multiply-adds. BGMV does the
whole batch in two launches and each program reads only the slot its token needs.

The two kernels mirror the two halves of the low-rank product:

``shrink``
    ``tmp[t, :r] = A[slot_t] @ x[t]`` -- one program per token, accumulating in fp32 over
    tiles of the input dimension. fp32 accumulation is not optional: ``in_features`` is
    thousands of elements and an fp16 running sum loses the small contributions that make
    up most of a LoRA delta.
``expand``
    ``y[t, n] += scaling[slot_t] * B[slot_t][n, :r] @ tmp[t, :r]`` -- one program per
    (token, output tile), reading the existing base output and writing the sum, so no
    atomics are needed: every ``(t, n)`` pair belongs to exactly one program.

Tokens whose slot is :data:`~turboserve.engine.core.types.NO_LORA` exit immediately, which
is what lets a batch mixing base-model and adapter traffic go through one launch.

Rank is padded to a power of two (``BLOCK_R``) and masked, so an adapter of rank 16 and one
of rank 12 share a kernel. The slot rows are indexed ``slot - 1`` because slot 0 is the
base model and never has storage (see :class:`~turboserve.engine.lora.layers.LoRALinear`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_BGMV_RANK",
    "TRITON_AVAILABLE",
    "bgmv_delta",
    "can_use_bgmv",
    "next_power_of_two",
]

try:  # pragma: no cover - import-time capability probe
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only install
    triton = None
    tl = None
    TRITON_AVAILABLE = False

#: Largest rank the kernels tile in one pass. Ranks above this fall back to the grouped
#: path; a BGMV program holding a [BLOCK_N, BLOCK_R] tile of ``B`` runs out of registers
#: long before rank 256 is useful for an adapter.
MAX_BGMV_RANK = 128

#: Dtypes the kernels are compiled for. fp32 is included because CPU-developed code paths
#: are exercised in fp32 on the one GPU available for smoke tests.
_SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16, torch.float32)


def next_power_of_two(value: int) -> int:
    """Smallest power of two ``>= value`` (``1`` for non-positive input)."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def can_use_bgmv(
    x: torch.Tensor,
    *,
    rank: int,
    max_tokens: int,
) -> bool:
    """Whether the BGMV kernels can and should serve this call.

    The "should" half matters as much as the "can": BGMV is a matrix-vector kernel, so it
    is the right choice when the batch is a handful of tokens per adapter (decode, and
    speculative verification) and the wrong one for a long prefill chunk, where the grouped
    GEMM path has real work to amortise a launch over. ``max_tokens`` is that cutoff and is
    a property of the caller, not of the kernel.
    """
    if not TRITON_AVAILABLE:
        return False
    if x.device.type != "cuda":
        return False
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not 1 <= rank <= MAX_BGMV_RANK:
        return False
    if x.dim() != 2 or x.shape[0] > max_tokens:
        return False
    return x.is_contiguous()


if TRITON_AVAILABLE:  # pragma: no cover - requires a CUDA device to execute

    @triton.jit
    def _bgmv_shrink_kernel(  # noqa: PLR0913 - a kernel signature is a flat argument list
        x_ptr,
        a_ptr,
        slot_ptr,
        out_ptr,
        x_stride_t,
        a_stride_slot,
        a_stride_r,
        out_stride_t,
        in_features,
        rank,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        """``out[t, :rank] = A[slot_t - 1] @ x[t]`` in fp32, one program per token."""
        token = tl.program_id(0)
        slot = tl.load(slot_ptr + token)
        if slot <= 0:
            return
        row = slot - 1

        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < rank
        acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

        for k_start in range(0, in_features, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < in_features
            x_vals = tl.load(x_ptr + token * x_stride_t + offs_k, mask=mask_k, other=0.0)
            a_vals = tl.load(
                a_ptr + row * a_stride_slot + offs_r[:, None] * a_stride_r + offs_k[None, :],
                mask=mask_r[:, None] & mask_k[None, :],
                other=0.0,
            )
            acc += tl.sum(a_vals.to(tl.float32) * x_vals.to(tl.float32)[None, :], axis=1)

        tl.store(out_ptr + token * out_stride_t + offs_r, acc, mask=mask_r)

    @triton.jit
    def _bgmv_expand_kernel(  # noqa: PLR0913 - a kernel signature is a flat argument list
        tmp_ptr,
        b_ptr,
        slot_ptr,
        scaling_ptr,
        y_ptr,
        tmp_stride_t,
        b_stride_slot,
        b_stride_n,
        y_stride_t,
        out_features,
        rank,
        BLOCK_R: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> None:
        """``y[t, n] += scaling[slot] * B[slot - 1][n, :rank] @ tmp[t, :rank]``."""
        token = tl.program_id(0)
        slot = tl.load(slot_ptr + token)
        if slot <= 0:
            return
        row = slot - 1

        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < out_features
        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < rank

        tmp_vals = tl.load(tmp_ptr + token * tmp_stride_t + offs_r, mask=mask_r, other=0.0)
        b_vals = tl.load(
            b_ptr + row * b_stride_slot + offs_n[:, None] * b_stride_n + offs_r[None, :],
            mask=mask_n[:, None] & mask_r[None, :],
            other=0.0,
        )
        scale = tl.load(scaling_ptr + row)
        acc = tl.sum(b_vals.to(tl.float32) * tmp_vals[None, :], axis=1) * scale

        y_ptrs = y_ptr + token * y_stride_t + offs_n
        previous = tl.load(y_ptrs, mask=mask_n, other=0.0)
        tl.store(y_ptrs, previous + acc.to(previous.dtype), mask=mask_n)


def bgmv_delta(
    x: torch.Tensor,
    y: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: torch.Tensor,
    token_slot: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    """Add every token's LoRA delta into ``y`` in place, gathering by ``token_slot``.

    Args:
        x: ``[T, in_features]``, contiguous, the projection's input.
        y: ``[T, out_features]``, the base projection's output; updated in place.
        lora_a: ``[num_slots, max_rank, in_features]``.
        lora_b: ``[num_slots, out_features, max_rank]``.
        scaling: ``[num_slots]`` fp32.
        token_slot: ``[T]`` int64, slot per token; ``0`` means the base model.
        rank: how many of ``max_rank`` columns carry data for this call.

    Returns ``y`` so the caller can chain. Raises :class:`RuntimeError` when Triton is not
    importable -- callers gate on :func:`can_use_bgmv` and never reach that.
    """
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by can_use_bgmv
        raise RuntimeError("Triton is not available; use the grouped LoRA path")
    num_tokens, in_features = x.shape
    num_slots, max_rank, a_in = lora_a.shape
    if a_in != in_features:
        raise ValueError(f"lora_a expects in_features {a_in}, got {in_features}")
    if lora_b.shape[0] != num_slots or lora_b.shape[2] != max_rank:
        raise ValueError(
            f"lora_b {tuple(lora_b.shape)} does not match lora_a {tuple(lora_a.shape)}"
        )
    if y.shape[0] != num_tokens or y.shape[1] != lora_b.shape[1]:
        raise ValueError(f"y {tuple(y.shape)} does not match the expansion output")
    if token_slot.shape[0] != num_tokens:
        raise ValueError(f"token_slot has {token_slot.shape[0]} entries for {num_tokens} tokens")
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank {rank} outside the stacked capacity {max_rank}")

    block_r = next_power_of_two(rank)
    tmp = torch.zeros((num_tokens, block_r), dtype=torch.float32, device=x.device)
    slots = token_slot.to(torch.int32)

    shrink: Any = _bgmv_shrink_kernel
    expand: Any = _bgmv_expand_kernel
    shrink[(num_tokens,)](
        x,
        lora_a,
        slots,
        tmp,
        x.stride(0),
        lora_a.stride(0),
        lora_a.stride(1),
        tmp.stride(0),
        in_features,
        rank,
        BLOCK_R=block_r,
        BLOCK_K=min(next_power_of_two(in_features), 256),
        num_warps=4,
    )
    out_features = int(lora_b.shape[1])
    block_n = min(next_power_of_two(out_features), 128)
    expand[(num_tokens, triton.cdiv(out_features, block_n))](
        tmp,
        lora_b,
        slots,
        scaling,
        y,
        tmp.stride(0),
        lora_b.stride(0),
        lora_b.stride(1),
        y.stride(0),
        out_features,
        rank,
        BLOCK_R=block_r,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return y
