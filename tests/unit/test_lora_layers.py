"""The batched LoRA projection: correct per token, and identical to a merged model.

The headline test is the last one: a single packed batch holding one sequence per adapter
plus one base-model sequence must produce, for every sequence, exactly what PEFT's own
``merge_and_unload()`` model produces for that sequence alone. That is the property the
whole design rests on -- if it holds, "serve N tenants from one copy of the weights" is a
memory optimisation and not a change of behaviour.

Everything else here exists to make a failure of that test diagnosable: the grouping, the
slot storage, and the no-adapter fast path are each checked on their own first.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from turboserve.engine.core.kv_cache import KVCache
from turboserve.engine.core.types import (
    NO_LORA,
    AttnMetadata,
    LoRAContext,
    build_query_start_loc,
    pad_block_tables,
)
from turboserve.engine.lora.layers import (
    LoRABatch,
    LoRALinear,
    install_lora,
    iter_lora_linears,
    lora_linear_factory,
    lora_segments,
    uninstall_lora,
)
from turboserve.engine.lora.registry import LoRARegistry
from turboserve.engine.model import CausalLM
from turboserve.engine.model.layers import LinearBase, use_linear_factory

BLOCK_SIZE = 8
POOL_BLOCKS = 32


def load_adapter_script() -> Any:
    """Return the synthetic-adapter trainer module.

    The trainer used to be a bare script under ``scripts/``; it now lives in the package as
    :mod:`turboserve.engine.lora.make_adapters` and is exposed as
    ``turboserve lora make-adapters``, so loading it is a plain import. The helper is kept
    so the fixtures below read the same as they did while it was a script.
    """
    return importlib.import_module("turboserve.engine.lora.make_adapters")


@pytest.fixture(scope="session")
def adapters_dir(tmp_path_factory: pytest.TempPathFactory, tiny_qwen2_path: Path) -> Path:
    """Two real PEFT adapters, trained for two steps with a large alpha.

    The alpha is deliberately large: after two optimiser steps the ``lora_B`` factors are
    small, and an adapter whose effect on the logits is 1e-3 would let a wrong-slot bug
    pass a 1e-4 comparison. With ``alpha=64`` over rank 4 the two adapters move the tiny
    model's logits by order 1, so "the batch used the other tenant's adapter" is a failure
    thousands of times larger than the tolerance.
    """
    module = load_adapter_script()
    out = tmp_path_factory.mktemp("adapters")
    module.make_adapters(
        str(tiny_qwen2_path),
        out,
        count=2,
        rank=4,
        alpha=64,
        steps=2,
        batch_size=2,
        learning_rate=0.05,
        max_length=32,
        device="cpu",
        local_files_only=True,
    )
    return out


def make_linear(in_features: int, out_features: int, *, seed: int = 0) -> LinearBase:
    """A base projection with reproducible weights."""
    generator = torch.Generator().manual_seed(seed)
    linear = LinearBase(in_features, out_features, name="test.q_proj")
    with torch.no_grad():
        linear.weight.copy_(torch.randn(out_features, in_features, generator=generator) * 0.1)
    return linear


def fill_slot(layer: LoRALinear, slot: int, *, rank: int, seed: int, scaling: float = 2.0) -> None:
    """Load reproducible random factors of ``rank`` into ``slot``."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(rank, layer.in_features, generator=generator) * 0.1
    b = torch.randn(layer.out_features, rank, generator=generator) * 0.1
    layer.load_slot(slot, a, b, scaling)


def reference_delta(layer: LoRALinear, x: torch.Tensor, slots: list[int]) -> torch.Tensor:
    """Per-token LoRA delta computed one token at a time -- the obvious, slow definition."""
    out = torch.zeros(x.shape[0], layer.out_features, dtype=x.dtype)
    for index, slot in enumerate(slots):
        if slot == NO_LORA:
            continue
        row = slot - 1
        rank = layer.slot_rank(slot)
        a = layer.lora_a[row, :rank]
        b = layer.lora_b[row, :, :rank]
        scaling = float(layer.lora_scaling[row])
        out[index] = scaling * (b @ (a @ x[index]))
    return out


# -- the grouping ------------------------------------------------------------------------


def test_from_token_slots_groups_by_slot_and_drops_base_tokens() -> None:
    batch = LoRABatch.from_token_slots([0, 2, 1, 2, 0, 1])
    assert batch.active_slots == [1, 2]
    assert batch.segments == ((1, 0, 2), (2, 2, 4))
    assert batch.order.tolist() == [2, 5, 1, 3]
    batch.validate()


def test_a_base_only_batch_is_empty_but_valid() -> None:
    batch = LoRABatch.from_token_slots([0, 0, 0])
    assert batch.is_base_only
    assert batch.segments == ()
    assert batch.order.numel() == 0
    batch.validate()


def test_validate_rejects_a_grouping_that_lost_a_token() -> None:
    batch = LoRABatch.from_token_slots([1, 1, 2])
    broken = LoRABatch(
        token_lora_slot=batch.token_lora_slot,
        active_slots=batch.active_slots,
        order=batch.order[:1],
        segments=batch.segments,
    )
    with pytest.raises(ValueError, match="segments cover"):
        broken.validate()


def test_segments_are_derived_for_a_plain_context() -> None:
    """A caller that hands over a bare LoRAContext still gets the right grouping."""
    plain = LoRAContext.from_slots([0, 3, 1, 3])
    order, segments = lora_segments(plain)
    assert segments == ((1, 0, 1), (3, 1, 3))
    assert order.tolist() == [2, 1, 3]


def test_to_moves_the_grouping_with_the_slots() -> None:
    batch = LoRABatch.from_token_slots([1, 0, 1])
    assert batch.to("cpu") is batch
    moved = batch.to(torch.device("cpu"))
    assert moved.order.device.type == "cpu"


# -- the layer ---------------------------------------------------------------------------


def test_without_a_context_the_layer_is_the_base_projection() -> None:
    base = make_linear(6, 4)
    layer = LoRALinear(base, num_slots=2, max_rank=4)
    fill_slot(layer, 1, rank=4, seed=1)
    x = torch.randn(3, 6)
    torch.testing.assert_close(layer(x), F.linear(x, base.weight))
    torch.testing.assert_close(
        layer(x, LoRABatch.from_token_slots([0, 0, 0])), F.linear(x, base.weight)
    )


def test_the_wrapper_shares_the_base_parameter_object() -> None:
    """Wrapping must not rename or copy the checkpoint parameter."""
    base = make_linear(6, 4)
    layer = LoRALinear(base, num_slots=1, max_rank=2)
    assert layer.weight is base.weight
    assert "weight" in dict(layer.named_parameters())
    assert "lora_a" not in layer.state_dict(), "slot storage must not enter the checkpoint"


def test_mixed_slots_match_the_per_token_definition() -> None:
    base = make_linear(8, 5)
    layer = LoRALinear(base, num_slots=3, max_rank=4)
    fill_slot(layer, 1, rank=4, seed=11)
    fill_slot(layer, 2, rank=2, seed=12, scaling=0.5)
    fill_slot(layer, 3, rank=4, seed=13)
    slots = [0, 2, 1, 3, 1, 0, 2]
    x = torch.randn(len(slots), 8)
    got = layer(x, LoRABatch.from_token_slots(slots))
    expected = F.linear(x, base.weight) + reference_delta(layer, x, slots)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_an_empty_slot_contributes_nothing() -> None:
    base = make_linear(4, 4)
    layer = LoRALinear(base, num_slots=2, max_rank=2)
    fill_slot(layer, 1, rank=2, seed=5)
    x = torch.randn(2, 4)
    # Slot 2 was never loaded, so the second token must come out as the base model.
    got = layer(x, LoRABatch.from_token_slots([1, 2]))
    torch.testing.assert_close(got[1], F.linear(x[1], base.weight))


def test_clear_slot_erases_an_evicted_tenant(tmp_path: Path) -> None:
    base = make_linear(4, 3)
    layer = LoRALinear(base, num_slots=1, max_rank=4)
    fill_slot(layer, 1, rank=4, seed=7)
    layer.clear_slot(1)
    assert layer.slot_rank(1) == 0
    x = torch.randn(2, 4)
    torch.testing.assert_close(
        layer(x, LoRABatch.from_token_slots([1, 1])), F.linear(x, base.weight)
    )


def test_a_narrower_adapter_zeroes_the_padding_it_does_not_use() -> None:
    """A rank-4 tenant evicted by a rank-2 one must leave nothing behind."""
    base = make_linear(4, 3)
    layer = LoRALinear(base, num_slots=1, max_rank=4)
    fill_slot(layer, 1, rank=4, seed=21)
    fill_slot(layer, 1, rank=2, seed=22)
    assert layer.slot_rank(1) == 2
    assert float(layer.lora_a[0, 2:].abs().max()) == 0.0
    assert float(layer.lora_b[0, :, 2:].abs().max()) == 0.0


def test_slot_and_shape_errors_are_explicit() -> None:
    base = make_linear(4, 3)
    layer = LoRALinear(base, num_slots=2, max_rank=4)
    with pytest.raises(ValueError, match="outside 1..2"):
        layer.load_slot(0, torch.zeros(2, 4), torch.zeros(3, 2), 1.0)
    with pytest.raises(ValueError, match="outside 1..2"):
        layer.load_slot(3, torch.zeros(2, 4), torch.zeros(3, 2), 1.0)
    with pytest.raises(ValueError, match="projection is"):
        layer.load_slot(1, torch.zeros(2, 9), torch.zeros(3, 2), 1.0)
    with pytest.raises(ValueError, match="exceeds the pool"):
        layer.load_slot(1, torch.zeros(9, 4), torch.zeros(3, 9), 1.0)
    with pytest.raises(ValueError, match="pool has 2 slots"):
        layer(torch.randn(1, 4), LoRABatch.from_token_slots([5]))
    with pytest.raises(ValueError, match="describes 2 tokens"):
        layer(torch.randn(3, 4), LoRABatch.from_token_slots([1, 1]))


def test_pool_geometry_must_be_positive() -> None:
    base = make_linear(4, 3)
    with pytest.raises(ValueError, match="num_slots"):
        LoRALinear(base, num_slots=0, max_rank=4)
    with pytest.raises(ValueError, match="max_rank"):
        LoRALinear(base, num_slots=1, max_rank=0)


def test_bytes_per_slot_is_the_two_factors() -> None:
    base = make_linear(64, 32)
    layer = LoRALinear(base, num_slots=4, max_rank=8)
    assert layer.bytes_per_slot == (8 * 64 + 32 * 8) * 4
    assert layer.bytes_reserved == 4 * layer.bytes_per_slot + 4 * 4


# -- installation ------------------------------------------------------------------------


def test_install_wraps_the_seven_projections_and_uninstall_restores_them(
    tiny_qwen2_path: Path,
) -> None:
    model = CausalLM.from_pretrained(
        tiny_qwen2_path, dtype=torch.float32, device="cpu", local_files_only=True
    )
    weight_before = model.model.layers[0].self_attn.q_proj.weight
    registry = LoRARegistry(2, max_lora_rank=4)
    wrapped = install_lora(model, registry)
    assert len(wrapped) == 2 * 7
    assert "lm_head" not in wrapped
    assert dict(iter_lora_linears(model)).keys() == set(wrapped)
    assert model.model.layers[0].self_attn.q_proj.weight is weight_before

    restored = uninstall_lora(model)
    assert restored == wrapped
    assert not dict(iter_lora_linears(model))
    assert model.model.layers[0].self_attn.q_proj.weight is weight_before


def test_install_is_idempotent(tiny_qwen2_path: Path) -> None:
    model = CausalLM.from_pretrained(
        tiny_qwen2_path, dtype=torch.float32, device="cpu", local_files_only=True
    )
    registry = LoRARegistry(2, max_lora_rank=4)
    first = install_lora(model, registry)
    layer = model.model.layers[0].self_attn.q_proj
    second = install_lora(model, registry)
    assert first == second
    assert model.model.layers[0].self_attn.q_proj is layer


def test_install_refuses_a_target_list_that_matches_nothing(tiny_qwen2_path: Path) -> None:
    model = CausalLM.from_pretrained(
        tiny_qwen2_path, dtype=torch.float32, device="cpu", local_files_only=True
    )
    registry = LoRARegistry(1, max_lora_rank=4)
    with pytest.raises(ValueError, match="matches target_modules"):
        install_lora(model, registry, target_modules=["no_such_proj"])


def test_the_factory_installs_during_construction(tiny_qwen2_path: Path) -> None:
    """The construction-time path: the wrapper exists before the checkpoint is loaded."""
    from turboserve.engine.model.model_config import ModelConfig

    config = ModelConfig.from_hf(tiny_qwen2_path, local_files_only=True)
    seen: dict[str, LoRALinear] = {}
    factory = lora_linear_factory(
        num_slots=2, max_rank=4, on_create=lambda name, layer: seen.__setitem__(name, layer)
    )
    with use_linear_factory(factory):
        model = CausalLM(config, dtype=torch.float32, device="cpu")
    assert len(seen) == 2 * 7
    assert isinstance(model.model.layers[0].mlp.gate_proj, LoRALinear)
    assert not isinstance(model.lm_head, LoRALinear)
    report = model.load_checkpoint(Path(tiny_qwen2_path))
    assert report.ok, report.summary()


# -- parity with a merged model ------------------------------------------------------------


def pack_batch(prompts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, AttnMetadata]:
    """Pack whole-prompt prefills of several sequences into one varlen step."""
    query_lens = [len(prompt) for prompt in prompts]
    block_tables: list[list[int]] = []
    slot_mapping: list[int] = []
    next_block = 0
    for prompt in prompts:
        needed = -(-len(prompt) // BLOCK_SIZE)
        blocks = list(range(next_block, next_block + needed))
        next_block += needed
        assert next_block <= POOL_BLOCKS
        block_tables.append(blocks)
        slot_mapping.extend(
            blocks[index // BLOCK_SIZE] * BLOCK_SIZE + index % BLOCK_SIZE
            for index in range(len(prompt))
        )
    meta = AttnMetadata(
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.long),
        block_tables=pad_block_tables(block_tables),
        context_lens=torch.tensor(query_lens, dtype=torch.long),
        query_start_loc=build_query_start_loc(query_lens),
        max_query_len=max(query_lens),
        max_context_len=max(query_lens),
        num_prefill_seqs=len(prompts),
        num_decode_seqs=0,
    )
    meta.validate(block_size=BLOCK_SIZE)
    input_ids = torch.tensor([token for prompt in prompts for token in prompt], dtype=torch.long)
    positions = torch.tensor(
        [index for prompt in prompts for index in range(len(prompt))], dtype=torch.long
    )
    return input_ids, positions, meta


def merged_reference_logits(
    base_path: Path, adapter_dir: Path | None, prompt: list[int]
) -> torch.Tensor:
    """Logits from PEFT's own merged model for one prompt on its own."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.float32, local_files_only=True
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor([prompt], dtype=torch.long)).logits[0].float()


def test_mixed_adapters_in_one_batch_equal_the_merged_models(
    tiny_qwen2_path: Path, adapters_dir: Path
) -> None:
    """One packed step, three sequences, two adapters and the base model.

    Each sequence's logits must match what a separately merged model produces for that
    sequence alone. This is the test the spec names: mixed adapters in one batch within
    1e-4 of PEFT.
    """
    model = CausalLM.from_pretrained(
        tiny_qwen2_path, dtype=torch.float32, device="cpu", local_files_only=True
    )
    registry = LoRARegistry(4, max_lora_rank=8)
    install_lora(model, registry)
    first = registry.register("tenant-0", adapters_dir / "tenant-0")
    second = registry.register("tenant-1", adapters_dir / "tenant-1")
    placement = registry.activate([first, second])

    prompts = [[5, 7, 9, 11, 13, 17, 19], [23, 29, 31, 37, 41], [43, 47, 53, 59, 61, 67]]
    assignment = [first, second, NO_LORA]
    token_slots = [
        placement.get(adapter, NO_LORA)
        for prompt, adapter in zip(prompts, assignment, strict=True)
        for _ in prompt
    ]
    context = LoRABatch.from_token_slots(token_slots)
    context.validate()

    input_ids, positions, meta = pack_batch(prompts)
    cache = KVCache(
        num_layers=model.config.num_hidden_layers,
        num_blocks=POOL_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=torch.float32,
        device="cpu",
    )
    with torch.no_grad():
        hidden = model(input_ids, positions, cache, meta, context)
        logits = model.compute_logits(hidden)

    directories = [adapters_dir / "tenant-0", adapters_dir / "tenant-1", None]
    offset = 0
    base_logits = merged_reference_logits(tiny_qwen2_path, None, prompts[0])
    for prompt, directory in zip(prompts, directories, strict=True):
        expected = merged_reference_logits(tiny_qwen2_path, directory, prompt)
        got = logits[offset : offset + len(prompt)]
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
        offset += len(prompt)

    # The adapters must actually be doing something, or the comparison above is vacuous.
    adapted = logits[: len(prompts[0])]
    assert float((adapted - base_logits).abs().max()) > 1e-2
