"""Architecture description read from a Hugging Face ``config.json``.

The engine deliberately does **not** instantiate ``transformers`` modelling classes: it
needs a transformer whose attention reads and writes a *paged* KV cache, which the HF
implementation has no hook for. What it does need is the handful of numbers that describe
the architecture, and those live in the checkpoint's ``config.json``. :class:`ModelConfig`
is that normalised view.

Only two architectures are accepted, ``Qwen2ForCausalLM`` and ``LlamaForCausalLM``. They
share a decoder block (RMSNorm, GQA attention with rotary embeddings, SwiGLU MLP) and
differ in exactly three places, all captured here:

* **Bias.** Qwen2 always puts a bias on ``q_proj``/``k_proj``/``v_proj`` and never on
  ``o_proj``; Llama drives both from ``attention_bias`` (false in every released
  checkpoint) and the MLP from ``mlp_bias``.
* **Activation.** Both normally use SiLU, but the field is honoured rather than assumed,
  because several tiny test checkpoints are built with GELU and silently substituting
  SiLU would turn a parity failure into a mystery.
* **RoPE parameterisation.** ``transformers`` 5 moved ``rope_theta``/``rope_scaling``
  into a nested ``rope_parameters`` block; both spellings are read.

Anything this module cannot model faithfully -- an unknown architecture, sliding-window
attention, a RoPE variant other than ``default``/``linear`` -- raises
:class:`UnsupportedModelError` at load time instead of producing quietly wrong logits.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: HF ``architectures[0]`` values this package implements, mapped to their ``model_type``.
SUPPORTED_ARCHITECTURES: dict[str, str] = {
    "Qwen2ForCausalLM": "qwen2",
    "LlamaForCausalLM": "llama",
}

#: MLP activations with an exact equivalent in :mod:`turboserve.engine.model.layers`.
SUPPORTED_ACTIVATIONS: frozenset[str] = frozenset(
    {"silu", "swish", "gelu", "gelu_new", "gelu_pytorch_tanh"}
)

#: RoPE variants implemented by :class:`~turboserve.engine.model.layers.RotaryEmbedding`.
#: ``linear`` divides positions by ``factor``; every other variant (``dynamic``,
#: ``llama3``, ``yarn``, ...) changes the frequency schedule and is rejected.
SUPPORTED_ROPE_TYPES: frozenset[str] = frozenset({"default", "linear"})

#: ``config.json`` dtype strings, mapped to torch dtypes. ``float8`` variants are absent
#: on purpose: the engine has no quantised path.
_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


class UnsupportedModelError(ValueError):
    """The checkpoint describes something this engine does not implement.

    Raised rather than approximated: a wrong activation or an unhandled RoPE scaling
    produces plausible-looking logits that are subtly wrong, and the failure would only
    surface as poor generation quality much later.
    """


def _require_int(data: Mapping[str, Any], key: str, *, default: int | None = None) -> int:
    value = data.get(key, default)
    if value is None:
        raise UnsupportedModelError(f"config.json is missing the required field {key!r}")
    if not isinstance(value, int) or isinstance(value, bool):
        raise UnsupportedModelError(f"config.json field {key!r} must be an int, got {value!r}")
    if value < 1:
        raise UnsupportedModelError(f"config.json field {key!r} must be positive, got {value}")
    return value


def _rope_block(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the RoPE settings, accepting both the flat and the nested spelling.

    ``transformers`` < 5 stores ``rope_theta`` and ``rope_scaling`` at the top level;
    ``transformers`` 5 nests them under ``rope_parameters``. Checkpoints of both vintages
    are in circulation (and in the test cache), so both are read.
    """
    nested = data.get("rope_parameters")
    if isinstance(nested, Mapping):
        return nested
    flat: dict[str, Any] = {}
    if "rope_theta" in data:
        flat["rope_theta"] = data["rope_theta"]
    scaling = data.get("rope_scaling")
    if isinstance(scaling, Mapping):
        flat.update(scaling)
    return flat


def _parse_rope(data: Mapping[str, Any]) -> tuple[float, float]:
    """Return ``(rope_theta, rope_scaling_factor)``, rejecting unimplemented variants."""
    block = _rope_block(data)
    theta = float(block.get("rope_theta", 10000.0))
    if theta <= 0:
        raise UnsupportedModelError(f"rope_theta must be positive, got {theta}")
    rope_type = str(block.get("rope_type") or block.get("type") or "default")
    if rope_type not in SUPPORTED_ROPE_TYPES:
        raise UnsupportedModelError(
            f"rope_type {rope_type!r} is not implemented; supported: {sorted(SUPPORTED_ROPE_TYPES)}"
        )
    factor = float(block.get("factor", 1.0)) if rope_type == "linear" else 1.0
    if factor <= 0:
        raise UnsupportedModelError(f"rope scaling factor must be positive, got {factor}")
    return theta, factor


def _parse_dtype(data: Mapping[str, Any]) -> torch.dtype | None:
    """Read the checkpoint's storage dtype (``dtype`` in tf5, ``torch_dtype`` before)."""
    raw = data.get("dtype", data.get("torch_dtype"))
    if raw is None:
        return None
    if isinstance(raw, torch.dtype):
        return raw
    name = str(raw).removeprefix("torch.")
    dtype = _DTYPE_BY_NAME.get(name)
    if dtype is None:
        raise UnsupportedModelError(f"unknown checkpoint dtype {raw!r}")
    return dtype


def _parse_eos(data: Mapping[str, Any]) -> tuple[int, ...]:
    """Normalise ``eos_token_id`` to a tuple; some checkpoints list several."""
    raw = data.get("eos_token_id")
    if raw is None:
        return ()
    if isinstance(raw, int):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(int(tok) for tok in raw)
    raise UnsupportedModelError(f"unsupported eos_token_id {raw!r}")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Normalised architecture parameters for one decoder-only checkpoint.

    Frozen because it is shared by the model, the runtime and (through
    :meth:`kv_bytes_per_token`) the KV-cache sizing code; a mutable copy passed around
    those three would make a capacity mismatch possible after the cache was allocated.
    """

    architecture: str
    """HF ``architectures[0]``, one of :data:`SUPPORTED_ARCHITECTURES`."""

    model_type: str
    """HF ``model_type`` (``"qwen2"`` / ``"llama"``)."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    hidden_act: str
    rms_norm_eps: float
    rope_theta: float
    rope_scaling_factor: float
    """Positions are divided by this before RoPE (1.0 for unscaled ``default`` RoPE)."""

    max_position_embeddings: int
    tie_word_embeddings: bool
    qkv_bias: bool
    """Whether ``q_proj``/``k_proj``/``v_proj`` carry a bias (true for Qwen2)."""

    o_proj_bias: bool
    """Whether ``o_proj`` carries a bias (never for Qwen2; ``attention_bias`` for Llama)."""

    mlp_bias: bool
    torch_dtype: torch.dtype | None = None
    eos_token_ids: tuple[int, ...] = ()
    bos_token_id: int | None = None
    pad_token_id: int | None = None
    name_or_path: str = ""

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise UnsupportedModelError(
                f"num_attention_heads={self.num_attention_heads} is not a multiple of "
                f"num_key_value_heads={self.num_key_value_heads}; GQA requires it"
            )
        if self.hidden_act not in SUPPORTED_ACTIVATIONS:
            raise UnsupportedModelError(
                f"hidden_act {self.hidden_act!r} is not implemented; supported: "
                f"{sorted(SUPPORTED_ACTIVATIONS)}"
            )

    # -- derived shapes ----------------------------------------------------------------

    @property
    def num_kv_groups(self) -> int:
        """Query heads sharing one KV head (1 means multi-head attention, no GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_proj_size(self) -> int:
        """Output width of ``q_proj``; not always ``hidden_size`` (``head_dim`` may differ)."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_size(self) -> int:
        """Output width of ``k_proj`` and ``v_proj``."""
        return self.num_key_value_heads * self.head_dim

    @property
    def attn_scale(self) -> float:
        """Softmax scale ``1/sqrt(head_dim)`` used by every attention backend."""
        return float(self.head_dim) ** -0.5

    @property
    def is_gqa(self) -> bool:
        """Whether fewer KV heads than query heads are stored (the paged cache is smaller)."""
        return self.num_kv_groups > 1

    def kv_bytes_per_token(self, dtype: torch.dtype) -> int:
        """Bytes one token occupies in the KV cache across all layers, keys and values.

        This is what the engine multiplies by ``block_size`` to size the block pool, so it
        must agree with :attr:`turboserve.engine.core.kv_cache.KVCache.bytes_per_block`.
        """
        return (
            2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * dtype.itemsize
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-data view for logging and for the result files' ``config`` block."""
        data = asdict(self)
        dtype = data.pop("torch_dtype")
        data["torch_dtype"] = None if dtype is None else str(dtype).removeprefix("torch.")
        data["eos_token_ids"] = list(self.eos_token_ids)
        return data

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name_or_path: str = "") -> ModelConfig:
        """Build a config from a parsed ``config.json`` mapping.

        Kept separate from :meth:`from_hf` so tests can exercise architecture handling
        (bias placement, RoPE spellings, rejection paths) without any file on disk.
        """
        architectures = data.get("architectures") or []
        architecture = str(architectures[0]) if architectures else ""
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise UnsupportedModelError(
                f"architecture {architecture or '<missing>'!r} is not implemented; supported: "
                f"{sorted(SUPPORTED_ARCHITECTURES)}"
            )
        model_type = str(data.get("model_type") or SUPPORTED_ARCHITECTURES[architecture])

        if data.get("use_sliding_window") and data.get("sliding_window"):
            raise UnsupportedModelError(
                "sliding-window attention is not implemented; the paged cache keeps the "
                "full context per sequence"
            )

        hidden_size = _require_int(data, "hidden_size")
        num_heads = _require_int(data, "num_attention_heads")
        num_kv_heads = _require_int(data, "num_key_value_heads", default=num_heads)
        head_dim_raw = data.get("head_dim")
        head_dim = int(head_dim_raw) if head_dim_raw else hidden_size // num_heads
        if head_dim < 1:
            raise UnsupportedModelError(f"head_dim resolved to {head_dim}")

        rope_theta, rope_factor = _parse_rope(data)
        attention_bias = bool(data.get("attention_bias", False))
        # Qwen2's q/k/v bias is architectural, not configurable; its o_proj never has one.
        qkv_bias = True if architecture == "Qwen2ForCausalLM" else attention_bias
        o_proj_bias = False if architecture == "Qwen2ForCausalLM" else attention_bias

        return cls(
            architecture=architecture,
            model_type=model_type,
            hidden_size=hidden_size,
            num_hidden_layers=_require_int(data, "num_hidden_layers"),
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=_require_int(data, "intermediate_size"),
            vocab_size=_require_int(data, "vocab_size"),
            hidden_act=str(data.get("hidden_act", "silu")),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=rope_theta,
            rope_scaling_factor=rope_factor,
            max_position_embeddings=_require_int(data, "max_position_embeddings", default=2048),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            qkv_bias=qkv_bias,
            o_proj_bias=o_proj_bias,
            mlp_bias=bool(data.get("mlp_bias", False)),
            torch_dtype=_parse_dtype(data),
            eos_token_ids=_parse_eos(data),
            bos_token_id=data.get("bos_token_id"),
            pad_token_id=data.get("pad_token_id"),
            name_or_path=name_or_path,
        )

    @classmethod
    def from_hf(
        cls,
        path_or_id: str | Path,
        *,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> ModelConfig:
        """Read ``config.json`` from a local checkpoint directory or a Hub repo id.

        ``local_files_only=True`` forbids network access, which is how the unit tests run:
        they point at snapshots already in the shared Hugging Face cache.
        """
        from turboserve.engine.model.weights import resolve_model_path

        model_dir = resolve_model_path(
            path_or_id, local_files_only=local_files_only, revision=revision
        )
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise UnsupportedModelError(f"no config.json under {model_dir}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise UnsupportedModelError(f"{config_path} does not contain a JSON object")
        config = cls.from_dict(data, name_or_path=str(path_or_id))
        logger.debug(
            "loaded %s config from %s: %d layers, hidden %d, %d/%d heads, vocab %d",
            config.architecture,
            model_dir,
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.vocab_size,
        )
        return config
