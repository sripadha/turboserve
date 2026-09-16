"""Qwen2/Llama transformer implementation that reads and writes the paged KV cache.

The engine builds a :class:`~turboserve.engine.model.model.CausalLM` from a Hugging Face
checkpoint and drives it with packed varlen batches described by
:class:`~turboserve.engine.core.types.AttnMetadata`. See ``docs/model.md`` for the layout
and ``tests/unit/test_model_parity.py`` for the equivalence that is maintained against
``transformers``.
"""

from turboserve.engine.model.attention import (
    BatchPlan,
    PagedAttention,
    causal_block_mask,
    gather_sequence_kv,
    paged_attention_reference,
)
from turboserve.engine.model.layers import (
    ACTIVATIONS,
    LORA_TARGET_MODULES,
    MLP,
    Attention,
    DecoderLayer,
    LinearBase,
    RMSNorm,
    RotaryEmbedding,
    named_linears,
    rotate_half,
    use_linear_factory,
)
from turboserve.engine.model.model import CausalLM, Decoder
from turboserve.engine.model.model_config import (
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_ROPE_TYPES,
    ModelConfig,
    UnsupportedModelError,
)
from turboserve.engine.model.triton_attention import (
    TRITON_AVAILABLE,
    can_use_triton_decode,
    paged_attention_decode_triton,
)
from turboserve.engine.model.weights import (
    LoadReport,
    WeightLoadError,
    load_weights,
    map_weight_name,
    resolve_model_path,
    safetensors_files,
)

__all__ = [
    "ACTIVATIONS",
    "LORA_TARGET_MODULES",
    "MLP",
    "SUPPORTED_ACTIVATIONS",
    "SUPPORTED_ARCHITECTURES",
    "SUPPORTED_ROPE_TYPES",
    "TRITON_AVAILABLE",
    "Attention",
    "BatchPlan",
    "CausalLM",
    "Decoder",
    "DecoderLayer",
    "LinearBase",
    "LoadReport",
    "ModelConfig",
    "PagedAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "UnsupportedModelError",
    "WeightLoadError",
    "can_use_triton_decode",
    "causal_block_mask",
    "gather_sequence_kv",
    "load_weights",
    "map_weight_name",
    "named_linears",
    "paged_attention_decode_triton",
    "paged_attention_reference",
    "resolve_model_path",
    "rotate_half",
    "safetensors_files",
    "use_linear_factory",
]
