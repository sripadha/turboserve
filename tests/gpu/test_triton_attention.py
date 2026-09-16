"""The Triton decode kernel agrees with the reference paged attention on CUDA.

These tests exist to prove the kernel *compiles and is numerically right*, not that it is
fast: shapes are deliberately tiny (a few kilobytes of KV) so the file runs in seconds on
any CUDA device, including a small consumer GPU. Performance is measured separately, on
the H100 profile, by ``turboserve bench``.

The tolerance is 1e-2 absolute in fp16. Both implementations accumulate in fp32, but they
sum the context in a different order -- the reference sums the whole context at once inside
SDPA, the kernel accumulates block by block with a rescaled running maximum -- so the last
bits of an fp16 output legitimately differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from turboserve.engine.core.kv_cache import KVCache
from turboserve.engine.core.types import AttnMetadata, build_query_start_loc, pad_block_tables
from turboserve.engine.model.attention import PagedAttention, paged_attention_reference
from turboserve.engine.model.triton_attention import (
    MAX_TILE_ELEMENTS,
    TRITON_AVAILABLE,
    can_use_triton_decode,
    paged_attention_decode_triton,
    triton_decode_tile_elements,
)

pytestmark = pytest.mark.gpu

BLOCK_SIZE = 16
HEAD_DIM = 32


@dataclass(slots=True)
class DecodeSeq:
    """One decoding sequence: where its KV blocks are and how long its context is."""

    blocks: list[int]
    context_len: int


def _decode_metadata(seqs: list[DecodeSeq], device: str) -> AttnMetadata:
    """Metadata for a pure decode step: one new token per sequence."""
    slots = [
        seq.blocks[(seq.context_len - 1) // BLOCK_SIZE] * BLOCK_SIZE
        + (seq.context_len - 1) % BLOCK_SIZE
        for seq in seqs
    ]
    contexts = [seq.context_len for seq in seqs]
    meta = AttnMetadata(
        slot_mapping=torch.tensor(slots, dtype=torch.long),
        block_tables=pad_block_tables([seq.blocks for seq in seqs]),
        context_lens=torch.tensor(contexts, dtype=torch.long),
        query_start_loc=build_query_start_loc([1] * len(seqs)),
        max_query_len=1,
        max_context_len=max(contexts),
        num_decode_seqs=len(seqs),
    )
    meta.validate(block_size=BLOCK_SIZE)
    return meta.to(device)


def _fill_random_cache(
    cache: KVCache, seqs: list[DecodeSeq], *, generator: torch.Generator
) -> None:
    """Write pseudo-random K/V into exactly the slots the sequences own."""
    for seq in seqs:
        slots = torch.tensor(
            [
                seq.blocks[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
                for pos in range(seq.context_len)
            ],
            dtype=torch.long,
        )
        shape = (seq.context_len, cache.num_kv_heads, cache.head_dim)
        k = torch.randn(shape, generator=generator).to(cache.dtype)
        v = torch.randn(shape, generator=generator).to(cache.dtype)
        cache.write(0, slots, k, v)


def _compare(
    seqs: list[DecodeSeq], *, num_heads: int, num_kv_heads: int, dtype: torch.dtype, seed: int
) -> None:
    """Run both backends on the same inputs and assert they agree."""
    device = "cuda"
    generator = torch.Generator().manual_seed(seed)
    num_blocks = max(b for seq in seqs for b in seq.blocks) + 1
    cache = KVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    try:
        _fill_random_cache(cache, seqs, generator=generator)
        meta = _decode_metadata(seqs, device)
        q = torch.randn((len(seqs), num_heads, HEAD_DIM), generator=generator).to(
            device=device, dtype=dtype
        )
        k_cache, v_cache = cache.layer(0)
        scale = HEAD_DIM**-0.5

        expected = paged_attention_reference(q, k_cache, v_cache, meta, scale=scale)
        actual = paged_attention_decode_triton(
            q, k_cache, v_cache, meta, scale=scale, num_kv_groups=num_heads // num_kv_heads
        )

        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    finally:
        cache.free()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_matches_reference_for_multi_head_attention(dtype: torch.dtype) -> None:
    """No GQA: one query head per KV head, contexts that do and do not fill a block."""
    seqs = [
        DecodeSeq(blocks=[2, 0], context_len=17),
        DecodeSeq(blocks=[3], context_len=16),
        DecodeSeq(blocks=[1, 4, 5], context_len=33),
    ]
    _compare(seqs, num_heads=4, num_kv_heads=4, dtype=dtype, seed=11)


def test_matches_reference_for_grouped_query_attention() -> None:
    """GQA: the program holds the whole four-head group while streaming one KV head."""
    seqs = [
        DecodeSeq(blocks=[5, 1], context_len=20),
        DecodeSeq(blocks=[0], context_len=3),
    ]
    _compare(seqs, num_heads=8, num_kv_heads=2, dtype=torch.float16, seed=23)


def test_matches_reference_for_a_single_cached_token() -> None:
    """A one-token context is the degenerate softmax; it must not divide by zero."""
    _compare(
        [DecodeSeq(blocks=[1], context_len=1)],
        num_heads=2,
        num_kv_heads=1,
        dtype=torch.float16,
        seed=5,
    )


def test_matches_reference_for_a_long_scrambled_block_table() -> None:
    """Many blocks in a jumbled order: the kernel must follow the table, not arithmetic."""
    _compare(
        [DecodeSeq(blocks=[7, 2, 9, 0, 4, 6], context_len=90)],
        num_heads=4,
        num_kv_heads=2,
        dtype=torch.float16,
        seed=31,
    )


def test_paged_attention_module_dispatches_to_triton_on_a_decode_step() -> None:
    """The layer picks Triton for CUDA decode and the reference for everything else."""
    assert TRITON_AVAILABLE
    attn = PagedAttention(num_heads=4, num_kv_heads=2, head_dim=HEAD_DIM)
    decode = _decode_metadata([DecodeSeq(blocks=[0], context_len=4)], "cuda")
    q = torch.randn(1, 4, HEAD_DIM, device="cuda", dtype=torch.float16)
    assert attn.select_backend(q, decode, block_size=BLOCK_SIZE) == "triton"
    assert attn.select_backend(q.cpu(), decode.to("cpu"), block_size=BLOCK_SIZE) == "reference"

    prefill = AttnMetadata(
        slot_mapping=torch.tensor([0, 1], dtype=torch.long, device="cuda"),
        block_tables=torch.tensor([[0]], dtype=torch.long, device="cuda"),
        context_lens=torch.tensor([2], dtype=torch.long, device="cuda"),
        query_start_loc=build_query_start_loc([2], device="cuda"),
        max_query_len=2,
        max_context_len=2,
        num_prefill_seqs=1,
    )
    assert attn.select_backend(q, prefill, block_size=BLOCK_SIZE) == "reference"


def test_dispatch_declines_oversized_tiles() -> None:
    """Very wide GQA groups would spill registers, so the reference path takes them."""
    decode = _decode_metadata([DecodeSeq(blocks=[0], context_len=4)], "cuda")
    q = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    assert triton_decode_tile_elements(num_kv_groups=32, head_dim=128, block_size=BLOCK_SIZE) > (
        MAX_TILE_ELEMENTS
    )
    assert not can_use_triton_decode(q, decode, num_kv_groups=32, block_size=BLOCK_SIZE)
    assert can_use_triton_decode(q, decode, num_kv_groups=4, block_size=BLOCK_SIZE)
    del q
    torch.cuda.empty_cache()
