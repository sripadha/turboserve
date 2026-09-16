"""The Triton BGMV kernels agree with the grouped path, on a real device.

``gpu``-marked and deliberately tiny (a few kilobytes of tensors, one projection, a handful
of tokens): its job is to prove that the decode kernel compiles and computes the same thing
as the portable path, not to say anything about speed. The reference is computed per token
in fp32, so both implementations are compared against the definition rather than against
each other -- an error shared by both would otherwise pass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from turboserve.engine.lora.layers import LoRABatch, LoRALinear
from turboserve.engine.lora.triton_bgmv import (
    MAX_BGMV_RANK,
    TRITON_AVAILABLE,
    bgmv_delta,
    can_use_bgmv,
    next_power_of_two,
)
from turboserve.engine.model.layers import LinearBase

pytestmark = pytest.mark.gpu

IN_FEATURES = 64
OUT_FEATURES = 48
NUM_SLOTS = 3
MAX_RANK = 8


def build_layer(dtype: torch.dtype, *, prefer_triton: bool) -> LoRALinear:
    """A wrapped projection on CUDA with every slot loaded from a fixed seed."""
    generator = torch.Generator(device="cpu").manual_seed(1234)
    base = LinearBase(IN_FEATURES, OUT_FEATURES, dtype=dtype, device="cuda", name="q_proj")
    with torch.no_grad():
        base.weight.copy_(
            (torch.randn(OUT_FEATURES, IN_FEATURES, generator=generator) * 0.05).to(dtype)
        )
    layer = LoRALinear(base, num_slots=NUM_SLOTS, max_rank=MAX_RANK, prefer_triton=prefer_triton)
    for slot in range(1, NUM_SLOTS + 1):
        rank = MAX_RANK if slot % 2 else MAX_RANK // 2
        a = (torch.randn(rank, IN_FEATURES, generator=generator) * 0.1).to(dtype).cuda()
        b = (torch.randn(OUT_FEATURES, rank, generator=generator) * 0.1).to(dtype).cuda()
        layer.load_slot(slot, a, b, 2.0)
    return layer


def reference_output(layer: LoRALinear, x: torch.Tensor, slots: list[int]) -> torch.Tensor:
    """Base projection plus the per-token delta, all in fp32."""
    weight = layer.weight.float()
    out = F.linear(x.float(), weight)
    for index, slot in enumerate(slots):
        if slot == 0:
            continue
        row = slot - 1
        rank = layer.slot_rank(slot)
        a = layer.lora_a[row, :rank].float()
        b = layer.lora_b[row, :, :rank].float()
        out[index] += float(layer.lora_scaling[row]) * (b @ (a @ x[index].float()))
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_bgmv_matches_the_definition_for_a_mixed_decode_batch(dtype: torch.dtype) -> None:
    slots = [1, 0, 2, 3, 2, 0]
    x = torch.randn(len(slots), IN_FEATURES, generator=torch.Generator().manual_seed(7)) * 0.5
    x = x.to(dtype).cuda()
    context = LoRABatch.from_token_slots(slots, device="cuda")

    triton_layer = build_layer(dtype, prefer_triton=True)
    grouped_layer = build_layer(dtype, prefer_triton=False)
    expected = reference_output(triton_layer, x, slots)

    from_triton = triton_layer(x, context)
    from_grouped = grouped_layer(x, context)

    tolerance = 3e-2 if dtype is torch.float16 else 1e-4
    torch.testing.assert_close(from_triton.float(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(from_grouped.float(), expected, atol=tolerance, rtol=tolerance)
    del triton_layer, grouped_layer, x, context, expected, from_triton, from_grouped
    torch.cuda.empty_cache()


def test_base_only_tokens_are_untouched_by_the_kernel() -> None:
    slots = [0, 1, 0]
    x = torch.randn(3, IN_FEATURES, device="cuda", dtype=torch.float16)
    layer = build_layer(torch.float16, prefer_triton=True)
    out = layer(x, LoRABatch.from_token_slots(slots, device="cuda"))
    base = F.linear(x, layer.weight)
    torch.testing.assert_close(out[0], base[0])
    torch.testing.assert_close(out[2], base[2])
    assert float((out[1] - base[1]).abs().max()) > 0.0
    del layer, x, out, base
    torch.cuda.empty_cache()


def test_can_use_bgmv_gates_on_device_rank_and_batch_size() -> None:
    cuda_x = torch.zeros(4, IN_FEATURES, device="cuda", dtype=torch.float16)
    assert can_use_bgmv(cuda_x, rank=8, max_tokens=256) is TRITON_AVAILABLE
    assert can_use_bgmv(cuda_x, rank=8, max_tokens=2) is False
    assert can_use_bgmv(cuda_x, rank=MAX_BGMV_RANK + 1, max_tokens=256) is False
    assert can_use_bgmv(torch.zeros(4, IN_FEATURES), rank=8, max_tokens=256) is False
    assert can_use_bgmv(cuda_x.to(torch.float64), rank=8, max_tokens=256) is False
    del cuda_x
    torch.cuda.empty_cache()


def test_bgmv_delta_validates_its_shapes() -> None:
    x = torch.zeros(2, IN_FEATURES, device="cuda", dtype=torch.float16)
    y = torch.zeros(2, OUT_FEATURES, device="cuda", dtype=torch.float16)
    a = torch.zeros(NUM_SLOTS, MAX_RANK, IN_FEATURES, device="cuda", dtype=torch.float16)
    b = torch.zeros(NUM_SLOTS, OUT_FEATURES, MAX_RANK, device="cuda", dtype=torch.float16)
    scaling = torch.zeros(NUM_SLOTS, device="cuda")
    slots = torch.zeros(2, dtype=torch.long, device="cuda")

    with pytest.raises(ValueError, match="rank"):
        bgmv_delta(x, y, a, b, scaling, slots, rank=MAX_RANK + 1)
    with pytest.raises(ValueError, match="in_features"):
        bgmv_delta(
            torch.zeros(2, 8, device="cuda", dtype=torch.float16), y, a, b, scaling, slots, rank=4
        )
    with pytest.raises(ValueError, match="token_slot"):
        bgmv_delta(x, y, a, b, scaling, torch.zeros(5, dtype=torch.long, device="cuda"), rank=4)
    del x, y, a, b, scaling, slots
    torch.cuda.empty_cache()


def test_next_power_of_two() -> None:
    assert [next_power_of_two(value) for value in (0, 1, 2, 3, 8, 9)] == [1, 1, 2, 4, 8, 16]
