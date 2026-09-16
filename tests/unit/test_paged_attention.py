"""Paged attention equals dense attention, for every batch shape the scheduler emits.

The reference implementation reads its keys and values through two levels of indirection
(a block table, then a slot offset inside each block) and reconstructs the causal mask from
``context_lens - query_len``. Both are easy to get subtly wrong in ways that still produce
finite, plausible numbers, so every test here computes the same attention a second way --
densely, from tensors that were never paged -- and compares.

Block ids are deliberately scrambled and interleaved between sequences: a bug that assumes
blocks are contiguous, or in ascending order, or that sequence ``i`` owns blocks starting
at ``i * n``, passes a test that allocates them in order and fails here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from turboserve.engine.core.kv_cache import KVCache
from turboserve.engine.core.types import (
    PAD_BLOCK,
    AttnMetadata,
    build_query_start_loc,
    pad_block_tables,
)
from turboserve.engine.model.attention import (
    BatchPlan,
    PagedAttention,
    causal_block_mask,
    gather_sequence_kv,
    paged_attention_reference,
)

BLOCK_SIZE = 4
HEAD_DIM = 8


@dataclass(slots=True)
class SeqSpec:
    """One sequence in a synthetic batch: its blocks, cached length and new query length."""

    blocks: list[int]
    num_computed: int
    query_len: int

    @property
    def context_len(self) -> int:
        """Attended length after this step."""
        return self.num_computed + self.query_len


def make_metadata(specs: list[SeqSpec], block_size: int = BLOCK_SIZE) -> AttnMetadata:
    """Build the packed-batch description for ``specs`` (prefill seqs first is not assumed)."""
    slots: list[int] = []
    for spec in specs:
        for pos in range(spec.num_computed, spec.context_len):
            slots.append(spec.blocks[pos // block_size] * block_size + pos % block_size)
    query_lens = [spec.query_len for spec in specs]
    contexts = [spec.context_len for spec in specs]
    return AttnMetadata(
        slot_mapping=torch.tensor(slots, dtype=torch.long),
        block_tables=pad_block_tables([spec.blocks for spec in specs]),
        context_lens=torch.tensor(contexts, dtype=torch.long),
        query_start_loc=build_query_start_loc(query_lens),
        max_query_len=max(query_lens),
        max_context_len=max(contexts),
        num_prefill_seqs=sum(1 for q in query_lens if q > 1),
        num_decode_seqs=sum(1 for q in query_lens if q == 1),
    )


def fill_cache(
    cache: KVCache, specs: list[SeqSpec], keys: list[torch.Tensor], values: list[torch.Tensor]
) -> None:
    """Scatter each sequence's full ``[context_len, kv_heads, head_dim]`` K/V into its blocks."""
    block_size = cache.block_size
    for spec, k, v in zip(specs, keys, values, strict=True):
        slots = torch.tensor(
            [
                spec.blocks[pos // block_size] * block_size + pos % block_size
                for pos in range(spec.context_len)
            ],
            dtype=torch.long,
        )
        for layer in range(cache.num_layers):
            cache.write(layer, slots, k, v)


def dense_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float, num_kv_groups: int
) -> torch.Tensor:
    """Attention for one sequence computed without any paging, as the ground truth.

    ``q`` is ``[q_len, heads, dim]``, ``k``/``v`` are ``[ctx, kv_heads, dim]``; the last
    ``q_len`` positions of the context are the queries.
    """
    ctx = k.shape[0]
    q_len = q.shape[0]
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)
    mask = causal_block_mask(q_len, ctx, device=q.device)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        attn_mask=mask.unsqueeze(0).unsqueeze(0),
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def run_case(
    specs: list[SeqSpec], *, num_heads: int, num_kv_heads: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(paged_output, dense_output)`` for one synthetic batch."""
    torch.manual_seed(seed)
    num_blocks = max(b for spec in specs for b in spec.blocks) + 1
    cache = KVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
    )
    keys = [torch.randn(spec.context_len, num_kv_heads, HEAD_DIM) for spec in specs]
    values = [torch.randn(spec.context_len, num_kv_heads, HEAD_DIM) for spec in specs]
    fill_cache(cache, specs, keys, values)

    total_q = sum(spec.query_len for spec in specs)
    q = torch.randn(total_q, num_heads, HEAD_DIM)
    meta = make_metadata(specs)
    meta.validate(block_size=BLOCK_SIZE)

    k_cache, v_cache = cache.layer(0)
    scale = HEAD_DIM**-0.5
    paged = paged_attention_reference(q, k_cache, v_cache, meta, scale=scale)

    groups = num_heads // num_kv_heads
    dense = torch.empty_like(q)
    offset = 0
    for spec, k, v in zip(specs, keys, values, strict=True):
        dense[offset : offset + spec.query_len] = dense_attention(
            q[offset : offset + spec.query_len], k, v, scale=scale, num_kv_groups=groups
        )
        offset += spec.query_len
    return paged, dense


@pytest.mark.parametrize("query_len", [1, 3, 7])
def test_single_sequence_matches_dense(query_len: int) -> None:
    """A whole prompt in one step (``num_computed == 0``) is ordinary causal attention."""
    specs = [SeqSpec(blocks=[3, 1, 0], num_computed=0, query_len=query_len)]
    paged, dense = run_case(specs, num_heads=4, num_kv_heads=4, seed=query_len)
    torch.testing.assert_close(paged, dense, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("query_len", [1, 3, 7])
def test_with_cached_prefix_matches_dense(query_len: int) -> None:
    """With ``num_computed > 0`` the queries sit at the end of a longer context.

    This is simultaneously the chunked-prefill case, the prefix-cache-hit case and (for
    ``query_len == 1``) the decode case: the engine has exactly one attention path.
    """
    specs = [SeqSpec(blocks=[5, 2, 7, 0], num_computed=9, query_len=query_len)]
    paged, dense = run_case(specs, num_heads=4, num_kv_heads=4, seed=100 + query_len)
    torch.testing.assert_close(paged, dense, rtol=1e-5, atol=1e-6)


def test_mixed_batch_of_prefill_and_decode() -> None:
    """One step holding a prefill chunk, a fresh prompt and two decodes, blocks interleaved."""
    specs = [
        SeqSpec(blocks=[6, 0, 4], num_computed=5, query_len=7),
        SeqSpec(blocks=[2, 5], num_computed=0, query_len=3),
        SeqSpec(blocks=[1, 7, 3], num_computed=10, query_len=1),
        SeqSpec(blocks=[8], num_computed=2, query_len=1),
    ]
    paged, dense = run_case(specs, num_heads=4, num_kv_heads=4, seed=7)
    torch.testing.assert_close(paged, dense, rtol=1e-5, atol=1e-6)


def test_grouped_query_attention_matches_dense() -> None:
    """GQA: four query heads share each KV head, mixed query lengths in one batch."""
    specs = [
        SeqSpec(blocks=[4, 1], num_computed=0, query_len=6),
        SeqSpec(blocks=[0, 3, 2], num_computed=8, query_len=1),
    ]
    paged, dense = run_case(specs, num_heads=8, num_kv_heads=2, seed=21)
    torch.testing.assert_close(paged, dense, rtol=1e-5, atol=1e-6)


def test_paged_attention_module_writes_cache_then_attends() -> None:
    """:class:`PagedAttention` writes this step's K/V before reading, so a query sees itself.

    The single-token, empty-cache case is the sharpest version of that invariant: if the
    write happened after the read the softmax denominator would be zero.
    """
    torch.manual_seed(3)
    cache = KVCache(1, 4, BLOCK_SIZE, num_kv_heads=2, head_dim=HEAD_DIM, dtype=torch.float32)
    specs = [SeqSpec(blocks=[2], num_computed=0, query_len=1)]
    meta = make_metadata(specs)
    attn = PagedAttention(num_heads=2, num_kv_heads=2, head_dim=HEAD_DIM)
    q = torch.randn(1, 2, HEAD_DIM)
    k = torch.randn(1, 2, HEAD_DIM)
    v = torch.randn(1, 2, HEAD_DIM)

    out = attn(q, k, v, cache, 0, meta)

    assert attn.select_backend(q, meta, block_size=BLOCK_SIZE) == "reference"
    # A single key means attention is the identity on the value vector.
    torch.testing.assert_close(out, v, rtol=1e-5, atol=1e-6)
    cached_k, cached_v = cache.read(0, meta.slot_mapping)
    torch.testing.assert_close(cached_k, k)
    torch.testing.assert_close(cached_v, v)


def test_causal_block_mask_offsets_by_cached_tokens() -> None:
    """Query ``j`` of a chunk attends to everything up to its own absolute position."""
    mask = causal_block_mask(3, 5, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_causal_block_mask_for_decode_is_all_true() -> None:
    """A decode step's single query attends to the whole context including itself."""
    mask = causal_block_mask(1, 6, device=torch.device("cpu"))
    assert bool(mask.all())


def test_gather_sequence_kv_follows_the_block_table() -> None:
    """Gathering reassembles tokens in sequence order regardless of block placement."""
    cache = torch.arange(4 * BLOCK_SIZE, dtype=torch.float32).reshape(4, BLOCK_SIZE, 1, 1)
    gathered = gather_sequence_kv(cache, (2, 0), 6)
    assert gathered.shape == (6, 1, 1)
    assert gathered.flatten().tolist() == [8.0, 9.0, 10.0, 11.0, 0.0, 1.0]


def test_gather_sequence_kv_empty_context() -> None:
    """A sequence with no cached tokens gathers an empty, correctly-shaped tensor."""
    cache = torch.zeros(2, BLOCK_SIZE, 3, HEAD_DIM)
    gathered = gather_sequence_kv(cache, (), 0)
    assert gathered.shape == (0, 3, HEAD_DIM)


def test_batch_plan_trims_block_tables_to_the_needed_blocks() -> None:
    """The plan keeps only ``ceil(context_len / block_size)`` entries per sequence."""
    specs = [
        SeqSpec(blocks=[3, 1, 0], num_computed=0, query_len=5),
        SeqSpec(blocks=[2], num_computed=1, query_len=1),
    ]
    plan = BatchPlan.from_metadata(make_metadata(specs), block_size=BLOCK_SIZE)
    assert plan.num_seqs == 2
    assert plan.query_lens == (5, 1)
    assert plan.context_lens == (5, 2)
    assert plan.block_ids == ((3, 1), (2,))


def test_batch_plan_rejects_a_block_table_that_is_too_short() -> None:
    """A padded entry inside the needed range means the batch and the allocator disagree."""
    meta = AttnMetadata(
        slot_mapping=torch.tensor([0, 1], dtype=torch.long),
        block_tables=torch.tensor([[0, PAD_BLOCK]], dtype=torch.long),
        context_lens=torch.tensor([6], dtype=torch.long),
        query_start_loc=build_query_start_loc([2]),
        max_query_len=2,
        max_context_len=6,
        num_prefill_seqs=1,
    )
    with pytest.raises(ValueError, match="needs 2 blocks"):
        BatchPlan.from_metadata(meta, block_size=BLOCK_SIZE)


def test_reference_rejects_more_queries_than_context() -> None:
    """``context_lens`` counts this step's tokens, so it can never be below ``query_len``."""
    cache = KVCache(1, 2, BLOCK_SIZE, num_kv_heads=1, head_dim=HEAD_DIM, dtype=torch.float32)
    meta = AttnMetadata(
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.long),
        block_tables=torch.tensor([[0]], dtype=torch.long),
        context_lens=torch.tensor([2], dtype=torch.long),
        query_start_loc=build_query_start_loc([3]),
        max_query_len=3,
        max_context_len=2,
        num_prefill_seqs=1,
    )
    k_cache, v_cache = cache.layer(0)
    plan = BatchPlan(
        query_offsets=(0, 3), context_lens=(2,), block_ids=((0,),), block_size=BLOCK_SIZE
    )
    with pytest.raises(ValueError, match="contributes 3 queries"):
        paged_attention_reference(
            torch.randn(3, 1, HEAD_DIM), k_cache, v_cache, meta, scale=1.0, plan=plan
        )


def test_paged_attention_rejects_head_count_mismatch() -> None:
    """Query heads must be a whole multiple of KV heads for GQA expansion to be defined."""
    with pytest.raises(ValueError, match="multiple of"):
        PagedAttention(num_heads=6, num_kv_heads=4, head_dim=HEAD_DIM)
