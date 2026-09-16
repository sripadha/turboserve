"""The model runner: one :class:`SchedulerOutput` in, one batch of sampled tokens out.

This is the only place that knows about tensors *and* about sequences. The scheduler
decides, in pure Python, what the step contains; the model consumes packed tensors and knows
nothing about requests; the runner is the seam. Keeping the seam explicit is what lets the
speculative-decoding engine reuse the same forward pass with ``k+1`` query tokens per
sequence, and what lets the LoRA module inject per-token adapter slots without any other
file changing.

One step is four phases, each a separate method so that a subclass or a hook can replace
exactly one of them:

1. :meth:`ModelRunner.build_batch` -- packed ``input_ids``/``positions``, the
   :class:`~turboserve.engine.core.types.AttnMetadata`, the sampling gather index and the
   per-token LoRA context, all built on the host and moved once.
2. :meth:`ModelRunner.forward` -- the decoder, writing this step's K/V into the paged cache.
3. :meth:`ModelRunner.logits` -- the vocabulary projection, applied only to the rows that
   sample. A prefill chunk that stops short of the end of the prompt contributes no row.
4. :meth:`ModelRunner.sample` -- the sampler, per-sequence parameters and RNG included.

Everything runs under ``torch.inference_mode``: no autograd graph is built, which is both
faster and the difference between a KV cache that can be written in place and one that
cannot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from turboserve.engine.core.sampler import Sampler, SamplerOutput
from turboserve.engine.core.types import NO_LORA, AttnMetadata, LoRAContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable
    from collections.abc import Sequence as SequenceABC

    from turboserve.engine.core.kv_cache import KVCache
    from turboserve.engine.core.scheduler import ScheduledSeq, SchedulerOutput
    from turboserve.engine.model.model import CausalLM

logger = logging.getLogger(__name__)

__all__ = ["BatchTensors", "LoRAContextBuilder", "ModelRunner", "StepOutput"]

type LoRAContextBuilder = Callable[[SchedulerOutput, torch.device], LoRAContext | None]
"""Signature of the hook that supplies per-token adapter slots.

The engine exposes it as ``LLMEngine.lora_ctx_builder`` so :mod:`turboserve.engine.lora` can
own adapter placement -- which slot a request's adapter currently occupies, and how that
changes as adapters are evicted -- without editing the runtime.
"""


@dataclass(slots=True)
class BatchTensors:
    """Device tensors describing one packed step.

    Held as a dataclass rather than passed as five arguments because the speculative and
    LoRA paths build a *modified* batch (extra query tokens, different adapter slots) from
    one the runner already built, and copying a small record is clearer than threading
    parallel argument lists.
    """

    input_ids: torch.Tensor
    """``[T]`` int64 token ids, every scheduled sequence's tokens concatenated."""

    positions: torch.Tensor
    """``[T]`` int64 absolute positions, for the rotary embedding."""

    meta: AttnMetadata
    """Block tables, slot mapping and context lengths for the paged attention."""

    sample_indices: torch.Tensor
    """Row indices into ``[T]`` whose hidden state predicts the next token."""

    lora_ctx: LoRAContext | None = None
    """Per-token adapter slots, or ``None`` when every token uses the base weights."""

    @property
    def num_tokens(self) -> int:
        """Tokens in the packed batch."""
        return int(self.input_ids.shape[0])

    @property
    def num_sampled(self) -> int:
        """Sequences that will produce a token this step."""
        return int(self.sample_indices.shape[0])

    @property
    def device(self) -> torch.device:
        """Device the batch lives on."""
        return self.input_ids.device


@dataclass(slots=True)
class StepOutput:
    """What one forward-and-sample produced, before the tokens are fed back.

    ``items`` and ``token_ids`` are aligned one to one; ``items`` is the subset of the
    step's scheduled sequences that sampled, in batch order.
    """

    items: list[ScheduledSeq]
    token_ids: list[int]
    logprobs: list[float] | None = None
    """Log-probability of each sampled token for the rows that asked for it, else ``nan``."""

    num_batched_tokens: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    def __post_init__(self) -> None:
        if len(self.items) != len(self.token_ids):
            raise ValueError(
                f"{len(self.token_ids)} sampled tokens for {len(self.items)} sampling sequences"
            )

    def __len__(self) -> int:
        return len(self.items)

    def pairs(self) -> list[tuple[ScheduledSeq, int]]:
        """``(scheduled sequence, sampled token)`` pairs, in batch order."""
        return list(zip(self.items, self.token_ids, strict=True))


class ModelRunner:
    """Executes one scheduler step against a model and a paged KV cache.

    The runner owns no request state: everything it needs for a step arrives in the
    :class:`~turboserve.engine.core.scheduler.SchedulerOutput`, and everything it produces
    goes back to the engine. That is what makes it safe to call from a worker thread while
    the event loop keeps accepting requests -- the engine serialises the calls, and the
    runner holds nothing that a second caller could observe half-updated.
    """

    def __init__(
        self,
        model: CausalLM,
        kv_cache: KVCache,
        *,
        sampler: Sampler | None = None,
        lora_ctx_builder: LoRAContextBuilder | None = None,
    ) -> None:
        if kv_cache.num_layers != model.num_layers:
            raise ValueError(
                f"kv cache has {kv_cache.num_layers} layers, the model has {model.num_layers}"
            )
        if kv_cache.num_kv_heads != model.config.num_key_value_heads:
            raise ValueError(
                f"kv cache has {kv_cache.num_kv_heads} kv heads, the model has "
                f"{model.config.num_key_value_heads}"
            )
        if kv_cache.head_dim != model.config.head_dim:
            raise ValueError(
                f"kv cache head_dim {kv_cache.head_dim} disagrees with the model's "
                f"{model.config.head_dim}"
            )
        self.model = model
        self.kv_cache = kv_cache
        self.sampler = sampler if sampler is not None else Sampler()
        self.lora_ctx_builder = lora_ctx_builder
        self._device = model.device

    @property
    def device(self) -> torch.device:
        """Device the model and the KV cache live on."""
        return self._device

    @property
    def block_size(self) -> int:
        """Tokens per KV block."""
        return self.kv_cache.block_size

    # -- phase 1: tensors ----------------------------------------------------------------

    def build_batch(self, out: SchedulerOutput) -> BatchTensors:
        """Turn the scheduler's decision into device tensors.

        The host-side lists are built once and moved once. ``input_ids`` and ``positions``
        are created directly on the target device rather than created on the host and
        copied, which for a decode step of a few dozen tokens is the difference between one
        small H2D transfer and three.
        """
        if out.is_empty:
            raise ValueError("cannot build a batch for an empty scheduler step")
        device = self._device
        input_ids = torch.tensor(out.input_token_ids(), dtype=torch.long, device=device)
        positions = torch.tensor(out.positions(), dtype=torch.long, device=device)
        meta = out.build_attn_metadata(self.block_size, device=device)
        sample_indices = torch.tensor(out.sample_indices(), dtype=torch.long, device=device)
        return BatchTensors(
            input_ids=input_ids,
            positions=positions,
            meta=meta,
            sample_indices=sample_indices,
            lora_ctx=self.build_lora_context(out),
        )

    def build_lora_context(self, out: SchedulerOutput) -> LoRAContext | None:
        """Per-token adapter slots for this step, or ``None`` for a base-only batch.

        The default skips building the tensor entirely when every scheduled sequence runs on
        the base model, which is the case for every request in a deployment with no adapters
        and for most steps in one with adapters. An installed ``lora_ctx_builder`` overrides
        the decision completely: it is the LoRA module's job to know which slot a request's
        adapter currently occupies, and that mapping can change between steps as adapters
        are evicted.
        """
        if self.lora_ctx_builder is not None:
            return self.lora_ctx_builder(out, self._device)
        if all(item.lora_id == NO_LORA for item in out.scheduled):
            return None
        return out.build_lora_context(device=self._device)

    # -- phases 2-4: forward, logits, sample ----------------------------------------------

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        """Run the decoder over the packed batch, writing K/V into the paged cache."""
        return self.model(
            batch.input_ids, batch.positions, self.kv_cache, batch.meta, batch.lora_ctx
        )

    def logits(self, hidden: torch.Tensor, batch: BatchTensors) -> torch.Tensor:
        """Project the sampling rows of ``hidden`` to fp32 vocabulary logits."""
        return self.model.compute_logits(hidden, batch.sample_indices)

    def sample(self, logits: torch.Tensor, items: SequenceABC[ScheduledSeq]) -> SamplerOutput:
        """Draw one token per sampling sequence, honouring per-request parameters."""
        return self.sampler.sample_sequences(logits, [item.seq for item in items])

    # -- the whole step --------------------------------------------------------------------

    @torch.inference_mode()
    def execute(self, out: SchedulerOutput) -> StepOutput:
        """Run one full step: build, forward, project, sample.

        Returns an empty :class:`StepOutput` for a step made entirely of prefill chunks that
        do not reach the end of their prompts. That is a real and common state under chunked
        prefill -- the forward pass still runs and still fills the KV cache, there is simply
        nothing to sample.
        """
        batch = self.build_batch(out)
        hidden = self.forward(batch)
        items = out.sampled()
        if not items:
            return StepOutput(
                items=[],
                token_ids=[],
                num_batched_tokens=out.num_batched_tokens,
                num_prefill_tokens=out.num_prefill_tokens,
                num_decode_tokens=out.num_decode_tokens,
            )
        logits = self.logits(hidden, batch)
        sampled = self.sample(logits, items)
        return StepOutput(
            items=items,
            token_ids=list(sampled.token_ids),
            logprobs=list(sampled.logprobs) if sampled.logprobs is not None else None,
            num_batched_tokens=out.num_batched_tokens,
            num_prefill_tokens=out.num_prefill_tokens,
            num_decode_tokens=out.num_decode_tokens,
        )

    def __repr__(self) -> str:
        return (
            f"ModelRunner(model={self.model.config.architecture}, device={self._device}, "
            f"blocks={self.kv_cache.num_blocks}x{self.kv_cache.block_size})"
        )
