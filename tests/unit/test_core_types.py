"""Unit tests for the shared engine contracts in ``engine/core/types.py``.

These are pure-CPU tests on tiny tensors: the point is the invariants every other module
relies on (validation ranges, packed-batch layout, -1 padding, timing arithmetic), not
model behaviour.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from turboserve.config import Settings
from turboserve.engine.core.types import (
    NO_LORA,
    PAD_BLOCK,
    AttnMetadata,
    EngineConfig,
    FinishReason,
    LoRAContext,
    RequestOutput,
    RequestTiming,
    SamplingParams,
    SchedulerConfig,
    build_query_start_loc,
    pad_block_tables,
    resolve_dtype,
)

# --------------------------------------------------------------------------------------
# SamplingParams
# --------------------------------------------------------------------------------------


def test_sampling_params_defaults_are_unconstrained_sampling() -> None:
    params = SamplingParams()
    assert params.max_tokens == 128
    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k == 0
    assert params.stop_token_ids == []
    assert params.stop == []
    assert not params.is_greedy


def test_sampling_params_mutable_defaults_are_not_shared() -> None:
    first = SamplingParams()
    first.stop_token_ids.append(7)
    assert SamplingParams().stop_token_ids == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_k": -1},
        {"max_tokens": 0},
        {"repetition_penalty": 0.0},
        {"nucleus": 0.9},  # unknown field
    ],
)
def test_sampling_params_rejects_invalid(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SamplingParams(**kwargs)


@pytest.mark.parametrize("kwargs", [{"temperature": 0.0}, {"top_p": 1.0}, {"top_k": 0}])
def test_sampling_params_accepts_boundary_values(kwargs: dict[str, float]) -> None:
    assert SamplingParams(**kwargs) is not None


def test_sampling_params_greedy_at_zero_temperature() -> None:
    assert SamplingParams(temperature=0.0).is_greedy


def test_sampling_params_validates_on_assignment() -> None:
    params = SamplingParams()
    with pytest.raises(ValidationError):
        params.top_p = 2.0


# --------------------------------------------------------------------------------------
# FinishReason
# --------------------------------------------------------------------------------------


def test_finish_reason_serialises_as_plain_strings() -> None:
    assert [reason.value for reason in FinishReason] == ["stop", "length", "abort"]
    assert f"{FinishReason.STOP}" == "stop"
    assert FinishReason("length") is FinishReason.LENGTH


# --------------------------------------------------------------------------------------
# RequestTiming
# --------------------------------------------------------------------------------------


def test_request_timing_derived_metrics() -> None:
    timing = RequestTiming(
        t_arrival=1.0,
        t_first_scheduled=1.25,
        t_first_token=1.5,
        t_finish=3.5,
        num_cached_prompt_tokens=32,
    )
    assert timing.ttft() == pytest.approx(0.5)
    assert timing.queue() == pytest.approx(0.25)
    assert timing.e2e() == pytest.approx(2.5)
    # 5 output tokens => 4 gaps over 2.0 s.
    assert timing.tpot(5) == pytest.approx(0.5)
    assert timing.num_cached_prompt_tokens == 32


def test_request_timing_is_none_until_the_event_happened() -> None:
    timing = RequestTiming(t_arrival=1.0)
    assert timing.ttft() is None
    assert timing.e2e() is None
    assert timing.queue() is None
    assert timing.tpot(10) is None


def test_request_timing_tpot_needs_two_tokens() -> None:
    timing = RequestTiming(t_arrival=0.0, t_first_token=1.0, t_finish=1.0)
    assert timing.tpot(1) is None
    assert timing.tpot(0) is None


# --------------------------------------------------------------------------------------
# Packed batch helpers
# --------------------------------------------------------------------------------------


def test_build_query_start_loc_is_cumulative() -> None:
    loc = build_query_start_loc([3, 1, 4])
    assert loc.dtype == torch.long
    assert loc.tolist() == [0, 3, 4, 8]


def test_build_query_start_loc_empty_batch() -> None:
    assert build_query_start_loc([]).tolist() == [0]


def test_pad_block_tables_pads_with_minus_one() -> None:
    table = pad_block_tables([[1, 2, 3], [4], []])
    assert table.dtype == torch.long
    assert table.tolist() == [
        [1, 2, 3],
        [4, PAD_BLOCK, PAD_BLOCK],
        [PAD_BLOCK, PAD_BLOCK, PAD_BLOCK],
    ]


def test_pad_block_tables_honours_wider_max_blocks() -> None:
    table = pad_block_tables([[1]], max_blocks=4)
    assert table.shape == (1, 4)
    assert table.tolist() == [[1, PAD_BLOCK, PAD_BLOCK, PAD_BLOCK]]


def test_pad_block_tables_rejects_narrower_max_blocks() -> None:
    with pytest.raises(ValueError, match="narrower"):
        pad_block_tables([[1, 2, 3]], max_blocks=2)


# --------------------------------------------------------------------------------------
# AttnMetadata
# --------------------------------------------------------------------------------------


def _mixed_batch() -> AttnMetadata:
    """One chunked-prefill sequence of 4 new tokens plus two decoding sequences.

    Block size 2. Sequence 0 attends to 4 tokens (2 blocks), sequences 1 and 2 attend to
    3 tokens each (2 blocks, the second half full).
    """
    return AttnMetadata(
        slot_mapping=torch.tensor([0, 1, 2, 3, 9, 13], dtype=torch.long),
        block_tables=pad_block_tables([[0, 1], [4, 5], [6, 7]]),
        context_lens=torch.tensor([4, 4, 4], dtype=torch.long),
        query_start_loc=build_query_start_loc([4, 1, 1]),
        max_query_len=4,
        max_context_len=4,
        num_prefill_seqs=1,
        num_decode_seqs=2,
    )


def test_attn_metadata_shape_properties() -> None:
    meta = _mixed_batch()
    assert meta.num_seqs == 3
    assert meta.num_tokens == 6
    assert meta.query_lens() == [4, 1, 1]
    assert not meta.is_prefill_only
    assert not meta.is_decode_only
    assert meta.device.type == "cpu"


def test_attn_metadata_validate_accepts_a_consistent_batch() -> None:
    _mixed_batch().validate(block_size=2)


def test_attn_metadata_decode_only_batch() -> None:
    meta = AttnMetadata(
        slot_mapping=torch.tensor([5, 9], dtype=torch.long),
        block_tables=pad_block_tables([[2], [4]]),
        context_lens=torch.tensor([2, 2], dtype=torch.long),
        query_start_loc=build_query_start_loc([1, 1]),
        max_query_len=1,
        max_context_len=2,
        num_decode_seqs=2,
    )
    meta.validate(block_size=2)
    assert meta.is_decode_only
    assert not meta.is_prefill_only


def test_attn_metadata_prefill_only_batch() -> None:
    meta = AttnMetadata(
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.long),
        block_tables=pad_block_tables([[0, 1]]),
        context_lens=torch.tensor([3], dtype=torch.long),
        query_start_loc=build_query_start_loc([3]),
        max_query_len=3,
        max_context_len=3,
        num_prefill_seqs=1,
    )
    meta.validate(block_size=2)
    assert meta.is_prefill_only


def test_attn_metadata_validate_rejects_token_count_mismatch() -> None:
    meta = _mixed_batch()
    meta.slot_mapping = torch.tensor([0, 1, 2], dtype=torch.long)
    with pytest.raises(ValueError, match="query_start_loc ends at"):
        meta.validate()


def test_attn_metadata_validate_rejects_wrong_row_count() -> None:
    meta = _mixed_batch()
    meta.block_tables = pad_block_tables([[0, 1]])
    with pytest.raises(ValueError, match="block_tables has 1 rows"):
        meta.validate()


def test_attn_metadata_validate_rejects_wrong_dtype() -> None:
    meta = _mixed_batch()
    meta.context_lens = meta.context_lens.to(torch.int32)
    with pytest.raises(ValueError, match="context_lens must be int64"):
        meta.validate()


def test_attn_metadata_validate_rejects_stale_max_query_len() -> None:
    meta = _mixed_batch()
    meta.max_query_len = 8
    with pytest.raises(ValueError, match="max_query_len"):
        meta.validate()


def test_attn_metadata_validate_rejects_short_block_table() -> None:
    meta = _mixed_batch()
    # 4 tokens of context need 2 blocks at block_size 2; give sequence 1 only one.
    meta.block_tables = pad_block_tables([[0, 1], [4], [6, 7]])
    with pytest.raises(ValueError, match="needs 2 blocks"):
        meta.validate(block_size=2)


def test_attn_metadata_validate_rejects_context_shorter_than_query() -> None:
    meta = _mixed_batch()
    meta.context_lens = torch.tensor([2, 4, 4], dtype=torch.long)
    with pytest.raises(ValueError, match="contributes 4 queries"):
        meta.validate()


def test_attn_metadata_to_same_device_is_identity() -> None:
    meta = _mixed_batch()
    assert meta.to("cpu") is meta


# --------------------------------------------------------------------------------------
# LoRAContext
# --------------------------------------------------------------------------------------


def test_lora_context_base_only() -> None:
    ctx = LoRAContext.base_only(5)
    assert ctx.num_tokens == 5
    assert ctx.is_base_only
    assert ctx.active_slots == []
    assert ctx.token_lora_slot.tolist() == [NO_LORA] * 5
    ctx.validate()


def test_lora_context_from_slots_derives_sorted_unique_active_slots() -> None:
    ctx = LoRAContext.from_slots([3, 0, 1, 3, 0])
    assert ctx.active_slots == [1, 3]
    assert not ctx.is_base_only
    ctx.validate()


def test_lora_context_validate_rejects_unlisted_slot() -> None:
    ctx = LoRAContext(token_lora_slot=torch.tensor([0, 2], dtype=torch.long), active_slots=[1])
    with pytest.raises(ValueError, match=r"slots \[2\]"):
        ctx.validate()


def test_lora_context_validate_rejects_base_slot_in_active_slots() -> None:
    ctx = LoRAContext(token_lora_slot=torch.tensor([0], dtype=torch.long), active_slots=[NO_LORA])
    with pytest.raises(ValueError, match="active_slots must all be"):
        ctx.validate()


def test_lora_context_to_same_device_is_identity() -> None:
    ctx = LoRAContext.base_only(2)
    assert ctx.to("cpu") is ctx


# --------------------------------------------------------------------------------------
# Config models
# --------------------------------------------------------------------------------------


def test_scheduler_config_defaults() -> None:
    config = SchedulerConfig()
    assert (config.max_num_seqs, config.max_num_batched_tokens, config.block_size) == (
        64,
        2048,
        16,
    )
    assert config.num_blocks is None
    assert config.enable_chunked_prefill and config.enable_prefix_caching
    assert config.policy == "fcfs"


def test_scheduler_config_rejects_non_power_of_two_block_size() -> None:
    with pytest.raises(ValidationError, match="power of two"):
        SchedulerConfig(block_size=12)


def test_scheduler_config_rejects_budget_smaller_than_a_block() -> None:
    with pytest.raises(ValidationError, match="at least"):
        SchedulerConfig(block_size=16, max_num_batched_tokens=8)


def test_scheduler_config_rejects_non_positive_tenant_weight() -> None:
    with pytest.raises(ValidationError, match="weights must be positive"):
        SchedulerConfig(policy="tenant_fair", tenant_weights={"a": 0.0})


def test_scheduler_config_rejects_unknown_policy() -> None:
    with pytest.raises(ValidationError):
        SchedulerConfig(policy="round_robin")


def test_engine_config_delegates_block_fields_to_the_scheduler() -> None:
    config = EngineConfig(model="tiny", scheduler=SchedulerConfig(block_size=8, num_blocks=64))
    assert config.block_size == 8
    assert config.num_blocks == 64


def test_engine_config_explicit_dtype_and_device() -> None:
    config = EngineConfig(model="tiny", dtype="float32", device="cpu")
    assert config.resolved_device() == "cpu"
    assert config.resolved_dtype() is torch.float32


def test_engine_config_auto_dtype_on_cpu_is_float32() -> None:
    config = EngineConfig(model="tiny", device="cpu")
    assert config.resolved_dtype() is torch.float32


def test_engine_config_keeps_subsystem_options_opaque() -> None:
    config = EngineConfig(model="tiny", speculative={"k": 4}, lora={"max_loras": 8})
    assert config.speculative == {"k": 4}
    assert config.lora == {"max_loras": 8}


def test_engine_config_from_settings_copies_every_scheduler_knob() -> None:
    settings = Settings(
        model="tiny-model",
        device="cpu",
        dtype="float32",
        block_size=8,
        num_blocks=32,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        scheduler_policy="tenant_fair",
        gpu_memory_utilization=0.5,
    )
    config = EngineConfig.from_settings(settings)
    assert config.model == "tiny-model"
    assert config.gpu_memory_utilization == 0.5
    assert config.scheduler.block_size == 8
    assert config.scheduler.num_blocks == 32
    assert config.scheduler.max_num_seqs == 4
    assert config.scheduler.max_num_batched_tokens == 128
    assert not config.scheduler.enable_chunked_prefill
    assert not config.scheduler.enable_prefix_caching
    assert config.scheduler.policy == "tenant_fair"
    assert config.resolved_dtype() is torch.float32


def test_engine_config_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        EngineConfig(model="tiny", quantization="awq")


# --------------------------------------------------------------------------------------
# dtype resolution and RequestOutput
# --------------------------------------------------------------------------------------


def test_resolve_dtype_accepts_names_and_torch_dtypes() -> None:
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype(torch.float32) is torch.float32


@pytest.mark.parametrize("name", ["auto", "int8", "fp16"])
def test_resolve_dtype_rejects_unsupported(name: str) -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        resolve_dtype(name)


def test_request_output_usage_accounting() -> None:
    output = RequestOutput(
        request_id="r1",
        new_token_ids=[5, 6],
        text_delta="hi",
        finished=True,
        finish_reason=FinishReason.STOP,
        prompt_tokens=10,
        output_tokens=4,
        cached_prompt_tokens=8,
    )
    assert output.total_tokens == 14
    assert output.usage() == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "cached_prompt_tokens": 8,
    }


def test_request_output_defaults_are_an_empty_unfinished_delta() -> None:
    output = RequestOutput(request_id="r1")
    assert output.new_token_ids == []
    assert output.text_delta == ""
    assert not output.finished
    assert output.finish_reason is None
    assert output.timing.ttft() is None
    assert RequestOutput(request_id="r2").new_token_ids == []
