"""Transformer building blocks: norms, rotary embeddings, projections, MLP, decoder layer.

Two design decisions in this module exist to serve modules that are written elsewhere:

* **Every projection is a :class:`LinearBase`, created through
  :meth:`LinearBase.create`.** That classmethod consults a class-level factory hook, so
  :mod:`turboserve.engine.lora` can substitute a LoRA-aware linear for every
  ``q_proj``/``k_proj``/``v_proj``/``o_proj``/``gate_proj``/``up_proj``/``down_proj`` in a
  freshly built model without this module importing (or knowing about) LoRA at all. The
  forward signature carries an optional
  :class:`~turboserve.engine.core.types.LoRAContext` for the same reason: the decoder
  layer threads it through unconditionally and the base implementation ignores it.
* **The rotary tables are precomputed, shared across layers and indexed by position id.**
  A packed varlen batch has no ``[batch, seq]`` structure, so the positions of the tokens
  in one step are an arbitrary int vector (a decode token at position 900 next to a
  prefill chunk at positions 12-15). Indexing a table with that vector is one gather;
  recomputing ``cos``/``sin`` per step per layer from ``inv_freq`` would be 28x that work
  and would also make every layer pay a device round trip.

The numerics deliberately mirror ``transformers``' Qwen2/Llama implementation element for
element (fp32 RMSNorm accumulation, ``rotate_half`` RoPE, SwiGLU with the checkpoint's own
activation), because ``tests/unit/test_model_parity.py`` asserts logit equality against it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from turboserve.engine.model.attention import BatchPlan, PagedAttention

if TYPE_CHECKING:
    from collections.abc import Iterator as _Iterator

    from turboserve.engine.core.kv_cache import KVCache
    from turboserve.engine.core.types import AttnMetadata, LoRAContext
    from turboserve.engine.model.model_config import ModelConfig

logger = logging.getLogger(__name__)

#: Linear module names the LoRA layer wraps by default (see ``docs/multi-lora.md``).
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Activation functions by ``config.hidden_act``. ``gelu`` is the exact erf form and
#: ``gelu_new``/``gelu_pytorch_tanh`` the tanh approximation, matching ``transformers``'
#: ``ACT2FN`` table; substituting one for the other shifts logits in the fourth decimal.
ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
}


class LinearBase(nn.Module):
    """A dense projection with a LoRA-shaped forward signature and a substitution hook.

    Functionally ``torch.nn.Linear`` without autograd bookkeeping, plus two things the
    engine needs:

    * ``forward(x, lora_ctx=None)`` -- the second argument lets a subclass add per-token
      adapter deltas. The base class ignores it, so a model built with no LoRA runs
      exactly one ``F.linear`` per projection.
    * :attr:`linear_factory` -- a class-level hook. While it is set,
      :meth:`create` returns whatever the factory builds, which is how LoRA layers get
      installed into a model without editing this file or the model file.

    ``name`` is the dotted module path the projection will have (for example
    ``model.layers.3.self_attn.q_proj``); it is passed to the factory so a LoRA
    implementation can decide per target module and per layer whether to wrap.
    """

    linear_factory: ClassVar[Callable[..., LinearBase] | None] = None
    """Optional builder consulted by :meth:`create`; see :func:`use_linear_factory`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        name: str = "",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        name: str = "",
    ) -> LinearBase:
        """Build a projection, honouring :attr:`linear_factory` when one is installed.

        Always called through ``LinearBase.create`` (never through a subclass) so that a
        factory cannot recurse into itself when it builds the base linear it wraps.
        """
        factory = LinearBase.linear_factory
        if factory is None:
            return LinearBase(
                in_features, out_features, bias=bias, dtype=dtype, device=device, name=name
            )
        return factory(in_features, out_features, bias=bias, dtype=dtype, device=device, name=name)

    def forward(self, x: torch.Tensor, lora_ctx: LoRAContext | None = None) -> torch.Tensor:
        """Apply the projection. ``lora_ctx`` is accepted and ignored by the base class."""
        del lora_ctx
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, name={self.name!r}"
        )


@contextmanager
def use_linear_factory(factory: Callable[..., LinearBase] | None) -> _Iterator[None]:
    """Install ``factory`` as :attr:`LinearBase.linear_factory` for the duration of a block.

    Scoped rather than global because model construction is the only moment the hook may
    be active: leaving it set would silently wrap the *next* model built in the same
    process (a draft model, or the parity reference) with another model's adapters.
    """
    previous = LinearBase.linear_factory
    LinearBase.linear_factory = factory
    try:
        yield
    finally:
        LinearBase.linear_factory = previous


def named_linears(module: nn.Module) -> Iterator[tuple[str, LinearBase]]:
    """Yield ``(dotted_name, linear)`` for every :class:`LinearBase` under ``module``.

    The LoRA registry uses this to find the projections it must stack adapters for,
    without depending on the decoder's attribute layout.
    """
    for name, child in module.named_modules():
        if isinstance(child, LinearBase):
            yield name, child


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019), matching HF numerics.

    The variance is accumulated in fp32 even when the weights are fp16: at
    ``hidden_size`` 4096 the sum of squares of a fp16 activation vector overflows the fp16
    range for perfectly ordinary activation magnitudes, and the resulting ``inf`` turns
    the whole hidden state into ``nan``.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension and rescale by the learned gain."""
        input_dtype = x.dtype
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return self.weight * x32.to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]`` over the last dimension (the GPT-NeoX RoPE layout).

    Kept as a named function because the *layout* is a compatibility contract: Qwen2 and
    Llama checkpoints are trained with the halves split this way, and the interleaved
    variant used by some other families would rotate the wrong pairs.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings, indexed by an arbitrary position vector.

    Tables are stored in fp32 and cast at use, which is what ``transformers`` does: the
    angles are computed once in full precision and only the multiply happens in the model
    dtype, so an fp16 model's positions past a few thousand do not lose resolution.

    The table grows on demand through :meth:`ensure_capacity`, which the model calls once
    per step with ``AttnMetadata.max_context_len`` -- a host-side int, so growing never
    forces a device synchronisation.
    """

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        *,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        initial_capacity: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for rotary embeddings, got {head_dim}")
        self.head_dim = head_dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.ensure_capacity(initial_capacity or min(max_position_embeddings, 1024))

    @property
    def capacity(self) -> int:
        """Highest position id + 1 currently present in the tables."""
        return int(self.cos_cached.shape[0])

    def ensure_capacity(self, num_positions: int) -> None:
        """Grow the tables so positions ``[0, num_positions)`` can be looked up.

        Growth doubles rather than fitting exactly, so a generation that walks a sequence
        out to 8k tokens rebuilds the table a handful of times, not once per step.
        """
        if num_positions <= self.capacity:
            return
        target = max(num_positions, 2 * self.capacity, 16)
        positions = torch.arange(target, dtype=torch.float32, device=self.inv_freq.device)
        if self.scaling_factor != 1.0:
            positions = positions / self.scaling_factor
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        logger.debug("rotary table grown to %d positions (head_dim=%d)", target, self.head_dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``q``/``k`` in place of their positions.

        ``q`` is ``[T, num_heads, head_dim]``, ``k`` is ``[T, num_kv_heads, head_dim]`` and
        ``positions`` is ``[T]`` int64 -- the packed varlen layout, with no batch axis.
        """
        cos = self.cos_cached.index_select(0, positions).to(q.dtype).unsqueeze(1)
        sin = self.sin_cached.index_select(0, positions).to(q.dtype).unsqueeze(1)
        q_out = q * cos + rotate_half(q) * sin
        k_out = k * cos.to(k.dtype) + rotate_half(k) * sin.to(k.dtype)
        return q_out, k_out

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, base={self.base}, "
            f"scaling_factor={self.scaling_factor}, capacity={self.capacity}"
        )


class Attention(nn.Module):
    """Grouped-query self-attention over the paged KV cache.

    Owns the four projections and the rotary application; the actual attention maths
    (writing K/V into the cache at ``slot_mapping`` and reading it back through the block
    tables) is delegated to :class:`~turboserve.engine.model.attention.PagedAttention` so
    the kernel choice is one decision in one place.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rotary: RotaryEmbedding,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary = rotary
        self.q_proj = LinearBase.create(
            config.hidden_size,
            config.q_proj_size,
            bias=config.qkv_bias,
            name=f"{prefix}q_proj",
            dtype=dtype,
            device=device,
        )
        self.k_proj = LinearBase.create(
            config.hidden_size,
            config.kv_proj_size,
            bias=config.qkv_bias,
            name=f"{prefix}k_proj",
            dtype=dtype,
            device=device,
        )
        self.v_proj = LinearBase.create(
            config.hidden_size,
            config.kv_proj_size,
            bias=config.qkv_bias,
            name=f"{prefix}v_proj",
            dtype=dtype,
            device=device,
        )
        self.o_proj = LinearBase.create(
            config.q_proj_size,
            config.hidden_size,
            bias=config.o_proj_bias,
            name=f"{prefix}o_proj",
            dtype=dtype,
            device=device,
        )
        self.attn = PagedAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=config.attn_scale,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache,
        meta: AttnMetadata,
        lora_ctx: LoRAContext | None = None,
        plan: BatchPlan | None = None,
    ) -> torch.Tensor:
        """Project, rotate, cache and attend for one packed batch of ``T`` tokens."""
        num_tokens = hidden.shape[0]
        q = self.q_proj(hidden, lora_ctx).view(num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden, lora_ctx).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden, lora_ctx).view(num_tokens, self.num_kv_heads, self.head_dim)
        q, k = self.rotary(q, k, positions)
        out = self.attn(q, k, v, kv_cache, self.layer_idx, meta, plan=plan)
        return self.o_proj(out.reshape(num_tokens, self.num_heads * self.head_dim), lora_ctx)


class MLP(nn.Module):
    """SwiGLU feed-forward block (Shazeer, 2020) as used by Qwen2 and Llama.

    The activation is read from the checkpoint rather than hard-coded to SiLU because
    several tiny test checkpoints are published with GELU, and a mismatch there is
    invisible except as a parity failure.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.act_fn = ACTIVATIONS[config.hidden_act]
        self.gate_proj = LinearBase.create(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
            name=f"{prefix}gate_proj",
            dtype=dtype,
            device=device,
        )
        self.up_proj = LinearBase.create(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
            name=f"{prefix}up_proj",
            dtype=dtype,
            device=device,
        )
        self.down_proj = LinearBase.create(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            name=f"{prefix}down_proj",
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor, lora_ctx: LoRAContext | None = None) -> torch.Tensor:
        """``down(act(gate(x)) * up(x))``."""
        gated = self.act_fn(self.gate_proj(x, lora_ctx)) * self.up_proj(x, lora_ctx)
        return self.down_proj(gated, lora_ctx)


class DecoderLayer(nn.Module):
    """One pre-norm transformer block: attention, residual, MLP, residual."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rotary: RotaryEmbedding,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device
        )
        self.self_attn = Attention(
            config, layer_idx, rotary, dtype=dtype, device=device, prefix=f"{prefix}self_attn."
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device
        )
        self.mlp = MLP(config, dtype=dtype, device=device, prefix=f"{prefix}mlp.")

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache,
        meta: AttnMetadata,
        lora_ctx: LoRAContext | None = None,
        plan: BatchPlan | None = None,
    ) -> torch.Tensor:
        """Run the block over ``[T, hidden_size]`` packed tokens."""
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(hidden, positions, kv_cache, meta, lora_ctx, plan)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden, lora_ctx)
        return residual + hidden
