"""``CausalLM``: a Qwen2/Llama decoder that reads and writes the engine's paged KV cache.

This is the model the engine actually runs. It is not a ``transformers`` subclass and does
not accept ``past_key_values``; instead every attention layer reads its context through
:class:`~turboserve.engine.core.types.AttnMetadata` -- a slot mapping for the tokens being
computed now and a block table for everything already cached. That single change is what
makes continuous batching, chunked prefill, prefix sharing and speculative verification
expressible at all, because none of them can be described by a rectangular
``[batch, seq]`` KV tensor.

The batch layout is *packed varlen*: one flat vector of ``T`` tokens, the concatenation of
every scheduled sequence's new tokens, with no padding. A single step legitimately mixes a
768-token prefill chunk, five decode tokens and a 5-token speculative verification; padding
them to a common length would multiply both the compute and the KV traffic by the ratio of
the longest to the mean.

Correspondence with ``transformers`` is a maintained property, not a coincidence:
``tests/unit/test_model_parity.py`` asserts that prefill logits match
``AutoModelForCausalLM`` within 1e-4 in fp32 on two checkpoints, that 20 greedy decode steps
reproduce ``generate`` exactly, that a chunked prefill equals a whole one, and that a
prefix-cached prefill equals an uncached one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from turboserve.engine.model.attention import BatchPlan
from turboserve.engine.model.layers import DecoderLayer, RMSNorm, RotaryEmbedding
from turboserve.engine.model.layers import LinearBase as _LinearBase
from turboserve.engine.model.model_config import ModelConfig
from turboserve.engine.model.triton_attention import TRITON_AVAILABLE
from turboserve.engine.model.weights import LoadReport, load_weights, safetensors_files

if TYPE_CHECKING:
    from turboserve.engine.core.kv_cache import KVCache
    from turboserve.engine.core.types import AttnMetadata, LoRAContext

logger = logging.getLogger(__name__)

#: Parameter name of the output projection, tied to the embedding when the config says so.
LM_HEAD_WEIGHT = "lm_head.weight"


class Decoder(nn.Module):
    """Embedding, the stack of :class:`~turboserve.engine.model.layers.DecoderLayer`, norm.

    Named ``model`` inside :class:`CausalLM` so that parameter paths line up with Hugging
    Face checkpoints (``model.layers.0.self_attn.q_proj.weight``), which keeps the weight
    name map in :mod:`turboserve.engine.model.weights` close to the identity and therefore
    easy to audit.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=dtype, device=device
        )
        # One rotary table for the whole stack: the angles depend only on position and
        # head_dim, so per-layer copies would cost num_layers times the memory and the
        # same number of redundant gathers.
        self.rotary = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
            device=device,
        )
        self.layers = nn.ModuleList(
            DecoderLayer(
                config,
                layer_idx,
                self.rotary,
                dtype=dtype,
                device=device,
                prefix=f"model.layers.{layer_idx}.",
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache,
        meta: AttnMetadata,
        lora_ctx: LoRAContext | None = None,
        plan: BatchPlan | None = None,
    ) -> torch.Tensor:
        """Run every layer over ``[T]`` packed tokens and return normed ``[T, hidden]``."""
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, positions, kv_cache, meta, lora_ctx, plan)
        return self.norm(hidden)


class CausalLM(nn.Module):
    """Decoder-only language model over a paged KV cache.

    Usage is deliberately two-step -- :meth:`forward` returns hidden states and
    :meth:`compute_logits` turns a *subset* of them into logits. The vocabulary projection
    is the single largest matmul in a decode step (``hidden_size x vocab_size``, over
    150k columns for Qwen2), and only the last token of each sequence needs it. Fusing the
    two would multiply that cost by the prefill chunk length for no benefit.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Decoder(config, dtype=dtype, device=device)
        self.lm_head = _LinearBase.create(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
            device=device,
            name="lm_head",
        )
        if config.tie_word_embeddings:
            # Share storage, do not copy: a 7B Qwen2 embedding is ~0.6 GB and the
            # checkpoint omits lm_head.weight entirely when the two are tied.
            self.lm_head.weight = cast(nn.Parameter, self.model.embed_tokens.weight)
        self.requires_grad_(False)
        self.eval()

    # -- properties --------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        """Element type of the model's parameters."""
        return self.model.embed_tokens.weight.dtype

    @property
    def device(self) -> torch.device:
        """Device the parameters live on."""
        return self.model.embed_tokens.weight.device

    @property
    def num_layers(self) -> int:
        """Decoder layers, which is also the number of KV-cache layers required."""
        return self.config.num_hidden_layers

    def num_parameters(self) -> int:
        """Distinct parameter elements, counting a tied ``lm_head`` only once."""
        seen: dict[int, int] = {}
        for param in self.parameters():
            seen[id(param)] = param.numel()
        return sum(seen.values())

    def tied_parameter_names(self) -> tuple[str, ...]:
        """Parameters the checkpoint may legitimately omit because they share storage."""
        return (LM_HEAD_WEIGHT,) if self.config.tie_word_embeddings else ()

    # -- forward -----------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache,
        meta: AttnMetadata,
        lora_ctx: LoRAContext | None = None,
    ) -> torch.Tensor:
        """Compute hidden states for one packed batch.

        Args:
            input_ids: ``[T]`` int64 token ids, every scheduled token concatenated.
            positions: ``[T]`` int64 absolute positions of those tokens in their sequences.
            kv_cache: the engine's block pool; this step's K/V is written into it.
            meta: batch description, with ``context_lens`` already including this step.
            lora_ctx: per-token adapter slots, or ``None`` for base weights only.

        Returns:
            ``[T, hidden_size]`` hidden states after the final norm, in token order.
        """
        if input_ids.dim() != 1 or positions.dim() != 1:
            raise ValueError(
                f"expected packed 1-D input_ids/positions, got {tuple(input_ids.shape)} "
                f"and {tuple(positions.shape)}"
            )
        if input_ids.shape[0] != positions.shape[0]:
            raise ValueError(
                f"input_ids has {input_ids.shape[0]} tokens but positions has {positions.shape[0]}"
            )
        if kv_cache.num_layers != self.num_layers:
            raise ValueError(
                f"kv cache has {kv_cache.num_layers} layers, the model has {self.num_layers}"
            )
        self.model.rotary.ensure_capacity(max(meta.max_context_len, int(input_ids.shape[0])))

        # The host-side plan costs three device reads. It is only needed by the reference
        # attention path, so a CUDA decode step that the Triton kernel will serve skips it;
        # every other batch builds it once here instead of once per layer.
        plan: BatchPlan | None = None
        if not (TRITON_AVAILABLE and self.device.type == "cuda" and meta.is_decode_only):
            plan = BatchPlan.from_metadata(meta, block_size=kv_cache.block_size)

        return self.model(input_ids, positions, kv_cache, meta, lora_ctx, plan)

    def compute_logits(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Project selected hidden states to vocabulary logits.

        Args:
            hidden: ``[T, hidden_size]`` states returned by :meth:`forward`.
            indices: positions within ``hidden`` to project (normally the last token of
                each sequence). ``None`` projects every row, which the parity tests use.

        Returns:
            ``[len(indices), vocab_size]`` logits in fp32. Sampling in fp32 keeps the
            softmax of a fp16 model numerically stable and costs one cast of a vector that
            is about to be reduced anyway.
        """
        if indices is not None:
            index = (
                indices
                if isinstance(indices, torch.Tensor)
                else torch.tensor(list(indices), dtype=torch.long, device=hidden.device)
            )
            hidden = hidden.index_select(0, index.to(hidden.device, dtype=torch.long))
        return self.lm_head(hidden).float()

    # -- construction ------------------------------------------------------------------

    def init_weights(self, *, seed: int | None = None, std: float = 0.02) -> None:
        """Fill the parameters with a seeded random initialisation.

        Parameters are allocated empty (loading a checkpoint overwrites every one of them,
        and zeroing 7B values first is pure waste). Tests that want a small model without a
        checkpoint -- attention shape tests, scheduler tests, speculative-decoding
        acceptance tests -- call this to get deterministic, finite weights.
        """
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith(".bias"):
                    param.zero_()
                elif isinstance(self.get_submodule(name.rsplit(".", 1)[0]), RMSNorm):
                    param.fill_(1.0)
                else:
                    values = torch.empty(param.shape, dtype=torch.float32)
                    values.normal_(0.0, std, generator=generator)
                    param.copy_(values.to(param.dtype))

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str | Path,
        *,
        dtype: torch.dtype | str = "auto",
        device: torch.device | str = "cpu",
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> CausalLM:
        """Build the model and load a Hugging Face safetensors checkpoint into it.

        ``dtype="auto"`` takes the checkpoint's own storage dtype, falling back to fp32 when
        the config does not record one. The engine normally passes an explicit dtype from
        :meth:`turboserve.engine.core.types.EngineConfig.resolved_dtype` instead, because
        the serving dtype is a deployment decision (fp16 on this project's Turing dev GPU,
        where bf16 has no tensor-core support) rather than a property of the checkpoint.
        """
        from turboserve.engine.model.weights import resolve_model_path

        model_dir = resolve_model_path(
            path_or_id, local_files_only=local_files_only, revision=revision
        )
        config = ModelConfig.from_hf(model_dir, local_files_only=local_files_only)
        resolved = _resolve_load_dtype(dtype, config)
        model = cls(config, dtype=resolved, device=device)
        report = model.load_checkpoint(model_dir, dtype=resolved)
        logger.info(
            "loaded %s from %s (%s, %s): %s",
            config.architecture,
            model_dir,
            resolved,
            torch.device(device),
            report.summary(),
        )
        return model

    def load_checkpoint(self, model_dir: Path, *, dtype: torch.dtype | None = None) -> LoadReport:
        """Load safetensors weights from ``model_dir`` into this already-built model."""
        return load_weights(
            self,
            safetensors_files(model_dir),
            dtype=dtype or self.dtype,
            tied_parameters=self.tied_parameter_names(),
        )


def _resolve_load_dtype(dtype: torch.dtype | str, config: ModelConfig) -> torch.dtype:
    """Resolve the ``dtype`` argument of :meth:`CausalLM.from_pretrained`."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return config.torch_dtype or torch.float32
    from turboserve.engine.core.types import resolve_dtype

    return resolve_dtype(dtype)
