"""Architecture parsing and safetensors loading.

``ModelConfig`` is where "what does this checkpoint actually compute" is decided, and every
mistake it can make is silent: a wrong activation, a bias that should not exist, an
unhandled RoPE variant. So the tests here are mostly about *rejection* -- that the loader
refuses what it cannot reproduce faithfully instead of approximating it -- plus the
architecture differences between Qwen2 and Llama that are not expressible as config fields.

The weight-loading tests cover the same theme from the other side: a checkpoint that does
not match the model must raise, never leave a randomly initialised layer in the middle of
the stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from turboserve.engine.core.kv_cache import KVCache
from turboserve.engine.model.model_config import (
    SUPPORTED_ARCHITECTURES,
    ModelConfig,
    UnsupportedModelError,
)
from turboserve.engine.model.weights import (
    SAFETENSORS_INDEX,
    LoadReport,
    WeightLoadError,
    layer_index_of,
    load_weights,
    map_weight_name,
    resolve_model_path,
    safetensors_files,
)

QWEN2_RAW: dict[str, object] = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 512,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "eos_token_id": 151645,
}

LLAMA_RAW: dict[str, object] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 256,
    "max_position_embeddings": 2048,
    "attention_bias": False,
    "rope_theta": 10000.0,
}


def test_qwen2_always_has_qkv_bias_and_never_an_output_bias() -> None:
    """Qwen2's bias placement is architectural, not configurable, and asymmetric."""
    config = ModelConfig.from_dict(QWEN2_RAW)
    assert config.qkv_bias is True
    assert config.o_proj_bias is False
    assert config.mlp_bias is False


def test_llama_bias_follows_attention_bias() -> None:
    """Llama drives both attention biases from one flag; ``mlp_bias`` is separate."""
    biased = ModelConfig.from_dict({**LLAMA_RAW, "attention_bias": True, "mlp_bias": True})
    assert (biased.qkv_bias, biased.o_proj_bias, biased.mlp_bias) == (True, True, True)
    plain = ModelConfig.from_dict(LLAMA_RAW)
    assert (plain.qkv_bias, plain.o_proj_bias, plain.mlp_bias) == (False, False, False)


def test_head_dim_defaults_to_hidden_over_heads_but_is_honoured_when_present() -> None:
    """Newer checkpoints set ``head_dim`` independently of ``hidden_size``."""
    assert ModelConfig.from_dict(QWEN2_RAW).head_dim == 8
    assert ModelConfig.from_dict({**QWEN2_RAW, "head_dim": 128}).head_dim == 128
    assert ModelConfig.from_dict({**QWEN2_RAW, "head_dim": 128}).q_proj_size == 8 * 128


def test_derived_gqa_shapes() -> None:
    """``num_kv_groups`` and the projection widths follow from the head counts."""
    config = ModelConfig.from_dict(QWEN2_RAW)
    assert config.num_kv_groups == 4
    assert config.is_gqa is True
    assert config.q_proj_size == 64
    assert config.kv_proj_size == 16
    assert config.attn_scale == pytest.approx(8**-0.5)
    assert ModelConfig.from_dict(LLAMA_RAW).is_gqa is False


def test_rope_is_read_from_both_the_flat_and_the_nested_spelling() -> None:
    """``transformers`` 5 nests RoPE settings; both layouts must give the same config."""
    flat = ModelConfig.from_dict({**QWEN2_RAW, "rope_theta": 5000.0})
    nested_raw = {k: v for k, v in QWEN2_RAW.items() if k != "rope_theta"}
    nested = ModelConfig.from_dict(
        {**nested_raw, "rope_parameters": {"rope_type": "default", "rope_theta": 5000.0}}
    )
    assert flat.rope_theta == nested.rope_theta == 5000.0
    assert flat.rope_scaling_factor == nested.rope_scaling_factor == 1.0


def test_linear_rope_scaling_is_supported() -> None:
    """Linear position scaling is a divide on the position ids, which the tables apply."""
    config = ModelConfig.from_dict(
        {**QWEN2_RAW, "rope_scaling": {"rope_type": "linear", "factor": 4.0}}
    )
    assert config.rope_scaling_factor == 4.0


@pytest.mark.parametrize("rope_type", ["dynamic", "llama3", "yarn"])
def test_unimplemented_rope_variants_are_rejected(rope_type: str) -> None:
    """Silently ignoring a frequency schedule would degrade long-context quality only."""
    with pytest.raises(UnsupportedModelError, match="rope_type"):
        ModelConfig.from_dict(
            {**QWEN2_RAW, "rope_scaling": {"rope_type": rope_type, "factor": 8.0}}
        )


def test_unknown_architecture_is_rejected() -> None:
    """Only the two architectures this package actually implements are accepted."""
    with pytest.raises(UnsupportedModelError, match="architecture"):
        ModelConfig.from_dict({**QWEN2_RAW, "architectures": ["MixtralForCausalLM"]})
    with pytest.raises(UnsupportedModelError, match="architecture"):
        ModelConfig.from_dict({k: v for k, v in QWEN2_RAW.items() if k != "architectures"})


def test_unknown_activation_is_rejected() -> None:
    """An activation with no exact equivalent would shift every logit slightly."""
    with pytest.raises(UnsupportedModelError, match="hidden_act"):
        ModelConfig.from_dict({**QWEN2_RAW, "hidden_act": "relu2"})


def test_sliding_window_attention_is_rejected() -> None:
    """The paged cache keeps a sequence's whole context; a window would change the maths."""
    with pytest.raises(UnsupportedModelError, match="sliding-window"):
        ModelConfig.from_dict({**QWEN2_RAW, "use_sliding_window": True, "sliding_window": 4096})


def test_non_divisible_head_counts_are_rejected() -> None:
    """GQA expansion is only defined when the group size is a whole number."""
    with pytest.raises(UnsupportedModelError, match="GQA"):
        ModelConfig.from_dict({**QWEN2_RAW, "num_key_value_heads": 3})


def test_missing_required_field_is_rejected() -> None:
    """A truncated config must fail loudly rather than defaulting to a guessed shape."""
    with pytest.raises(UnsupportedModelError, match="hidden_size"):
        ModelConfig.from_dict({k: v for k, v in QWEN2_RAW.items() if k != "hidden_size"})


def test_dtype_is_read_from_either_field_name() -> None:
    """``torch_dtype`` was renamed to ``dtype``; checkpoints of both vintages are cached."""
    assert ModelConfig.from_dict(QWEN2_RAW).torch_dtype is torch.bfloat16
    renamed = {k: v for k, v in QWEN2_RAW.items() if k != "torch_dtype"}
    assert ModelConfig.from_dict({**renamed, "dtype": "float16"}).torch_dtype is torch.float16
    assert ModelConfig.from_dict(renamed).torch_dtype is None
    with pytest.raises(UnsupportedModelError, match="dtype"):
        ModelConfig.from_dict({**renamed, "dtype": "int4"})


def test_eos_token_ids_are_normalised_to_a_tuple() -> None:
    """Some chat checkpoints list several stop tokens; the sampler wants one shape."""
    assert ModelConfig.from_dict(QWEN2_RAW).eos_token_ids == (151645,)
    multi = ModelConfig.from_dict({**QWEN2_RAW, "eos_token_id": [1, 2]})
    assert multi.eos_token_ids == (1, 2)
    none = {k: v for k, v in QWEN2_RAW.items() if k != "eos_token_id"}
    assert ModelConfig.from_dict(none).eos_token_ids == ()


def test_kv_bytes_per_token_agrees_with_the_cache() -> None:
    """Capacity planning uses this number, so it must match what ``KVCache`` allocates."""
    config = ModelConfig.from_dict(QWEN2_RAW)
    block_size = 16
    cache = KVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float16,
    )
    assert config.kv_bytes_per_token(torch.float16) * block_size == cache.bytes_per_block


def test_to_dict_is_json_serialisable() -> None:
    """The config is embedded in result files, which are plain JSON."""
    data = ModelConfig.from_dict(QWEN2_RAW, name_or_path="Qwen/x").to_dict()
    assert json.loads(json.dumps(data))["torch_dtype"] == "bfloat16"
    assert data["name_or_path"] == "Qwen/x"


def test_from_hf_reads_the_cached_tiny_checkpoints(
    tiny_qwen2_path: Path, tiny_llama_path: Path
) -> None:
    """Both cached tiny models parse, and their architectures are the supported ones."""
    for path in (tiny_qwen2_path, tiny_llama_path):
        config = ModelConfig.from_hf(path, local_files_only=True)
        assert config.architecture in SUPPORTED_ARCHITECTURES
        assert config.model_type == SUPPORTED_ARCHITECTURES[config.architecture]
        assert config.num_hidden_layers >= 1


def test_from_hf_without_a_config_raises(tmp_path: Path) -> None:
    """A directory that is not a checkpoint is a configuration error, not a crash later."""
    with pytest.raises(UnsupportedModelError, match="config.json"):
        ModelConfig.from_hf(tmp_path, local_files_only=True)


# -- weights ---------------------------------------------------------------------------


def test_map_weight_name_is_identity_for_hf_names() -> None:
    """Engine parameter paths mirror HF ones, which keeps the mapping auditable."""
    for name in (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.11.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ):
        assert map_weight_name(name) == name


def test_map_weight_name_rewrites_legacy_prefixes() -> None:
    """Older exports prefix the decoder with ``transformer.``."""
    assert map_weight_name("transformer.layers.0.mlp.up_proj.weight") == (
        "model.layers.0.mlp.up_proj.weight"
    )
    assert map_weight_name("language_model.lm_head.weight") == "lm_head.weight"


def test_map_weight_name_skips_derived_buffers() -> None:
    """``inv_freq`` is recomputed from ``rope_theta``; loading it would be a no-op at best."""
    assert map_weight_name("model.layers.0.self_attn.rotary_emb.inv_freq") is None


def test_layer_index_of() -> None:
    """Per-layer grouping (LoRA stacking, debug logs) relies on this one parser."""
    assert layer_index_of("model.layers.7.self_attn.k_proj.weight") == 7
    assert layer_index_of("model.norm.weight") is None


def _write_shards(tmp_path: Path, shards: dict[str, dict[str, torch.Tensor]]) -> None:
    for filename, tensors in shards.items():
        save_file(tensors, str(tmp_path / filename))


def test_safetensors_files_single_file(tmp_path: Path) -> None:
    """The common case: one ``model.safetensors``."""
    _write_shards(tmp_path, {"model.safetensors": {"a": torch.zeros(2)}})
    assert [p.name for p in safetensors_files(tmp_path)] == ["model.safetensors"]


def test_safetensors_files_follows_the_shard_index(tmp_path: Path) -> None:
    """Sharded checkpoints are enumerated from the index, not from a glob."""
    _write_shards(
        tmp_path,
        {
            "model-00001-of-00002.safetensors": {"a": torch.zeros(2)},
            "model-00002-of-00002.safetensors": {"b": torch.zeros(2)},
        },
    )
    (tmp_path / SAFETENSORS_INDEX).write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    assert len(safetensors_files(tmp_path)) == 2


def test_safetensors_files_reports_a_missing_shard(tmp_path: Path) -> None:
    """A shard named in the index but absent would otherwise look like missing parameters."""
    _write_shards(tmp_path, {"model-00001-of-00002.safetensors": {"a": torch.zeros(2)}})
    (tmp_path / SAFETENSORS_INDEX).write_text(
        json.dumps({"weight_map": {"b": "model-00002-of-00002.safetensors"}})
    )
    with pytest.raises(WeightLoadError, match="absent"):
        safetensors_files(tmp_path)


def test_safetensors_files_rejects_pickle_only_checkpoints(tmp_path: Path) -> None:
    """``torch.load`` on a third-party pickle is an arbitrary-code-execution hazard."""
    (tmp_path / "pytorch_model.bin").write_bytes(b"not really a checkpoint")
    with pytest.raises(WeightLoadError, match="safetensors"):
        safetensors_files(tmp_path)


def test_safetensors_files_on_an_empty_directory(tmp_path: Path) -> None:
    """No weights at all is its own error message."""
    with pytest.raises(WeightLoadError, match="no safetensors"):
        safetensors_files(tmp_path)


class _Tiny(nn.Module):
    """A two-parameter module standing in for the decoder in loader tests."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2, bias=False)
        self.register_buffer("derived", torch.zeros(2), persistent=False)


def test_load_weights_fills_every_parameter(tmp_path: Path) -> None:
    """A matching checkpoint loads completely and reports no missing or unexpected keys."""
    module = _Tiny()
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    _write_shards(tmp_path, {"model.safetensors": {"lin.weight": weight}})

    report = load_weights(module, safetensors_files(tmp_path))

    assert report.ok
    assert report.num_tensors == 1
    assert report.missing == [] and report.unexpected == []
    torch.testing.assert_close(module.lin.weight, weight)


def test_load_weights_ignores_non_persistent_buffers(tmp_path: Path) -> None:
    """Rotary tables are derived, not stored; they must not count as missing weights."""
    module = _Tiny()
    _write_shards(tmp_path, {"model.safetensors": {"lin.weight": torch.zeros(2, 3)}})
    report = load_weights(module, safetensors_files(tmp_path))
    assert "derived" not in report.missing


def test_load_weights_reports_missing_and_unexpected(tmp_path: Path) -> None:
    """A mismatch raises by default; ``strict=False`` returns the diagnosis instead."""
    module = _Tiny()
    _write_shards(tmp_path, {"model.safetensors": {"other.weight": torch.zeros(2, 3)}})
    files = safetensors_files(tmp_path)

    with pytest.raises(WeightLoadError, match="does not match"):
        load_weights(module, files)

    report = load_weights(module, files, strict=False)
    assert report.missing == ["lin.weight"]
    assert report.unexpected == ["other.weight"]
    assert not report.ok


def test_load_weights_rejects_a_shape_mismatch(tmp_path: Path) -> None:
    """Copying a wrongly-shaped tensor would either throw deep in torch or broadcast."""
    module = _Tiny()
    _write_shards(tmp_path, {"model.safetensors": {"lin.weight": torch.zeros(3, 3)}})
    with pytest.raises(WeightLoadError, match="shape mismatch"):
        load_weights(module, safetensors_files(tmp_path))


def test_load_weights_accepts_tied_parameters(tmp_path: Path) -> None:
    """A tied ``lm_head`` is absent from the checkpoint by design, not by accident."""

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(4, 2)
            self.head = nn.Linear(2, 4, bias=False)
            self.head.weight = self.embed.weight

    module = Tied()
    _write_shards(tmp_path, {"model.safetensors": {"embed.weight": torch.ones(4, 2)}})
    report = load_weights(module, safetensors_files(tmp_path), tied_parameters=("head.weight",))
    assert report.ok
    assert report.tied == []  # named_parameters de-duplicates the shared tensor


def test_load_weights_casts_to_the_requested_dtype(tmp_path: Path) -> None:
    """Serving dtype is a deployment decision; the checkpoint's dtype does not decide it."""
    module = _Tiny().to(torch.float16)
    _write_shards(tmp_path, {"model.safetensors": {"lin.weight": torch.ones(2, 3)}})
    load_weights(module, safetensors_files(tmp_path), dtype=torch.float16)
    assert module.lin.weight.dtype is torch.float16


def test_load_report_summary_is_informative() -> None:
    """The summary line is what ends up in the engine's startup log."""
    report = LoadReport(num_tensors=3, num_bytes=48, files=["a.safetensors"], missing=["x"])
    assert "3 tensors" in report.summary()
    assert "missing=1" in report.summary()
    assert not report.ok


def test_resolve_model_path_returns_existing_directories(tmp_path: Path) -> None:
    """A local path is never resolved through the Hub, which keeps offline runs offline."""
    assert resolve_model_path(tmp_path) == tmp_path
    assert resolve_model_path(str(tmp_path)) == tmp_path


def test_resolve_model_path_rejects_a_file(tmp_path: Path) -> None:
    """Pointing at the weights file instead of its directory is a common mistake."""
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"")
    with pytest.raises(WeightLoadError, match="expected a checkpoint directory"):
        resolve_model_path(target)


def test_resolve_model_path_offline_miss_is_a_weight_load_error() -> None:
    """Offline resolution of an uncached repo id must not raise a Hub-internal exception."""
    with pytest.raises(WeightLoadError, match="could not resolve"):
        resolve_model_path("turboserve-tests/definitely-not-cached", local_files_only=True)
