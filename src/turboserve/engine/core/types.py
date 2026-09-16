"""Data contracts shared by the engine, the gateway and the benchmark client.

Nothing in this module depends on a model, a tokenizer or a scheduler: it is the
vocabulary the other modules use to talk to each other, so that ``engine/model`` can be
written against the same batch layout the scheduler emits and the gateway can report the
same timings the benchmark client measures.

Three decisions are encoded here and are relied on everywhere downstream:

* **Packed varlen batches.** A scheduler step produces one flat token vector, the
  concatenation of every scheduled sequence's new tokens, described by
  :class:`AttnMetadata`. There is no ``[batch, seq]`` padding, because prefill chunks and
  decode steps of wildly different lengths are mixed in a single step; padding them to a
  common length would waste both compute and memory bandwidth.
* **``-1``-padded block tables.** ``block_tables`` is the one rectangular tensor in the
  batch (``[num_seqs, max_blocks]``). ``-1`` marks a slot past the end of a sequence's
  table: it is not a valid block id, so a kernel or a reference implementation that
  accidentally reads it produces an index error instead of silently attending to another
  tenant's KV blocks. Padding with ``0`` would alias block 0 and fail silently.
* **Monotonic float seconds for timings.** :class:`RequestTiming` stores
  ``time.perf_counter()`` seconds, not wall-clock: the derived quantities (TTFT, ITL,
  TPOT, E2E) are differences, and a clock that can jump backwards makes them negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from collections.abc import Sequence

    from turboserve.config import Settings

__all__ = [
    "DTYPE_BY_NAME",
    "NO_LORA",
    "PAD_BLOCK",
    "AttnMetadata",
    "DeviceName",
    "DTypeName",
    "EngineConfig",
    "FinishReason",
    "LoRAContext",
    "RequestOutput",
    "RequestTiming",
    "SamplingParams",
    "SchedulerConfig",
    "SchedulerPolicy",
    "build_query_start_loc",
    "pad_block_tables",
    "resolve_dtype",
]

#: LoRA slot reserved for "no adapter": the base model's weights are used unchanged.
#: Adapter slots are ``>= 1`` so that a zero-filled ``token_lora_slot`` vector is a valid
#: base-model batch and the grouped LoRA kernels can skip slot 0 without a special case.
NO_LORA = 0

#: Fill value for unused entries of a block table. See the module docstring.
PAD_BLOCK = -1

DeviceName = Literal["auto", "cuda", "cpu"]
DTypeName = Literal["auto", "float16", "bfloat16", "float32"]
SchedulerPolicy = Literal["fcfs", "tenant_fair"]

#: The dtypes the engine supports. bf16 is listed but is not selected by ``auto``, because
#: pre-Ampere CUDA devices have no bf16 tensor cores.
DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Map a dtype name from configuration onto a ``torch.dtype``.

    Accepts a ``torch.dtype`` unchanged so call sites can take either without branching.
    ``"auto"`` is deliberately rejected: resolving it needs to know the device, which is
    :meth:`EngineConfig.resolved_dtype`'s job, not this function's.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return DTYPE_BY_NAME[dtype]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(DTYPE_BY_NAME)}"
        ) from None


class SamplingParams(BaseModel):
    """Per-request sampling configuration.

    Validated at construction so the sampler never has to defend against a negative
    temperature or an empty top-p mass: a bad request is rejected at the gateway edge with
    a 422 instead of producing NaNs several layers down.

    ``temperature == 0`` means greedy decoding (argmax); ``top_k == 0`` means "no top-k
    filter", matching the OpenAI-style conventions the gateway exposes.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = None
    stop_token_ids: list[int] = Field(default_factory=list)
    stop: list[str] = Field(default_factory=list)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    ignore_eos: bool = False
    logprobs: bool = False

    @property
    def is_greedy(self) -> bool:
        """Whether this request decodes greedily, i.e. the sampler takes the argmax."""
        return self.temperature == 0.0


class FinishReason(StrEnum):
    """Why a sequence stopped producing tokens.

    A :class:`~enum.StrEnum` so it serialises to the plain strings the OpenAI API uses
    without a custom encoder on every boundary (JSON responses, result files, Prometheus
    labels), and so an interpolated member renders as ``"stop"`` rather than
    ``"FinishReason.STOP"``.
    """

    STOP = "stop"
    """An EOS or a configured stop token/string was produced."""

    LENGTH = "length"
    """``max_tokens`` was reached."""

    ABORT = "abort"
    """The client disconnected or the request was cancelled server-side."""


@dataclass(slots=True)
class RequestTiming:
    """The four timestamps every layer records for a request, plus cache accounting.

    Timestamps are ``time.perf_counter()`` seconds (monotonic, process-local) and are
    ``None`` until the corresponding event happens, which is what makes the derived
    metrics safe: a request that never produced a token has ``ttft() is None`` rather than
    a zero that would quietly drag a percentile down.

    ``num_cached_prompt_tokens`` is the number of prompt tokens served from the prefix
    cache; it is carried here because TTFT is only interpretable next to it (a request
    whose whole prompt hit the cache did no prefill work at all).
    """

    t_arrival: float | None = None
    t_first_scheduled: float | None = None
    t_first_token: float | None = None
    t_finish: float | None = None
    num_cached_prompt_tokens: int = 0

    def ttft(self) -> float | None:
        """Time to first token, in seconds, or ``None`` if no token was emitted yet."""
        if self.t_arrival is None or self.t_first_token is None:
            return None
        return self.t_first_token - self.t_arrival

    def e2e(self) -> float | None:
        """End-to-end latency, in seconds, or ``None`` if the request is unfinished."""
        if self.t_arrival is None or self.t_finish is None:
            return None
        return self.t_finish - self.t_arrival

    def tpot(self, n_out: int) -> float | None:
        """Mean time per output token after the first, in seconds.

        ``None`` when fewer than two tokens were produced: TPOT is the slope of the decode
        phase and a single token gives no slope. The first token is excluded because its
        latency is prefill, already reported as TTFT.
        """
        if n_out < 2 or self.t_first_token is None or self.t_finish is None:
            return None
        return (self.t_finish - self.t_first_token) / (n_out - 1)

    def queue(self) -> float | None:
        """Seconds spent waiting before the scheduler first picked the request up."""
        if self.t_arrival is None or self.t_first_scheduled is None:
            return None
        return self.t_first_scheduled - self.t_arrival


def build_query_start_loc(
    query_lens: Sequence[int],
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the ``[num_seqs + 1]`` cumulative offset vector of a packed batch.

    Entry ``i`` is the index of sequence ``i``'s first token in the flat token vector and
    the last entry is the total token count, so sequence ``i`` owns ``tokens[loc[i]:
    loc[i + 1]]``. This is the same layout FlashAttention calls ``cu_seqlens_q``.
    """
    out = torch.zeros(len(query_lens) + 1, dtype=torch.long, device=device)
    if query_lens:
        out[1:] = torch.tensor(list(query_lens), dtype=torch.long, device=device).cumsum(0)
    return out


def pad_block_tables(
    rows: Sequence[Sequence[int]],
    *,
    device: torch.device | str = "cpu",
    max_blocks: int | None = None,
) -> torch.Tensor:
    """Stack per-sequence block tables into one ``[num_seqs, max_blocks]`` tensor.

    Rows shorter than the widest one are padded with :data:`PAD_BLOCK`. ``max_blocks``
    forces a wider tensor, which the runtime uses to keep one pre-allocated staging buffer
    across steps instead of reallocating whenever the longest sequence grows.
    """
    width = max((len(row) for row in rows), default=0)
    if max_blocks is not None:
        if max_blocks < width:
            raise ValueError(f"max_blocks={max_blocks} is narrower than the widest row ({width})")
        width = max_blocks
    out = torch.full((len(rows), width), PAD_BLOCK, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        if row:
            out[index, : len(row)] = torch.tensor(list(row), dtype=torch.long, device=device)
    return out


def _on_device(tensor: torch.Tensor, device: torch.device) -> bool:
    """Whether ``tensor`` already lives on ``device``, treating ``cuda`` as ``cuda:*``."""
    if tensor.device.type != device.type:
        return False
    return device.index is None or tensor.device.index == device.index


@dataclass(slots=True)
class AttnMetadata:
    """Everything the attention layers need to know about one packed batch.

    One instance describes a whole scheduler step: ``num_prefill_seqs`` prefill (or
    chunked-prefill) sequences followed by ``num_decode_seqs`` decoding sequences, their
    new tokens concatenated into a single flat vector. Attention kernels read positions
    out of ``query_start_loc`` and their KV out of ``block_tables``/``context_lens``; the
    model writes the new K/V into the paged cache at ``slot_mapping`` first.

    Invariants (checked by :meth:`validate`, which the runtime calls in tests and when
    debug logging is on, not on every step -- verifying ``query_start_loc`` means reading
    a device tensor, and that would force a GPU synchronisation per layer):

    * ``slot_mapping`` has one entry per token in the batch, in batch order.
    * ``query_start_loc`` is non-decreasing, starts at 0 and ends at ``num_tokens``.
    * ``context_lens[i]`` counts *all* tokens sequence ``i`` attends to, including the
      ones contributed by this step, so the causal mask offset is
      ``context_lens[i] - query_len[i]``.
    * ``block_tables[i]`` holds at least ``ceil(context_lens[i] / block_size)`` valid
      entries; the rest are :data:`PAD_BLOCK`.
    """

    slot_mapping: torch.Tensor
    """``[num_tokens]`` int64. Flat KV slot ``block_id * block_size + offset`` per token."""

    block_tables: torch.Tensor
    """``[num_seqs, max_blocks]`` int64, padded with :data:`PAD_BLOCK`."""

    context_lens: torch.Tensor
    """``[num_seqs]`` int64. Total attended length per sequence after this step."""

    query_start_loc: torch.Tensor
    """``[num_seqs + 1]`` int64. Cumulative token offsets (``cu_seqlens_q``)."""

    max_query_len: int
    max_context_len: int
    num_prefill_seqs: int = 0
    num_decode_seqs: int = 0

    @property
    def num_seqs(self) -> int:
        """Number of sequences in the batch, prefill and decode together."""
        return self.num_prefill_seqs + self.num_decode_seqs

    @property
    def num_tokens(self) -> int:
        """Number of tokens in the packed batch."""
        return int(self.slot_mapping.shape[0])

    @property
    def is_prefill_only(self) -> bool:
        """Whether no sequence in this batch is decoding."""
        return self.num_decode_seqs == 0 and self.num_prefill_seqs > 0

    @property
    def is_decode_only(self) -> bool:
        """Whether every sequence contributes exactly one query token.

        This is the condition under which the Triton paged-decode kernel may be used: it
        assumes ``query_len == 1`` and indexes KV directly by block table row.
        """
        return self.num_prefill_seqs == 0 and self.num_decode_seqs > 0 and self.max_query_len == 1

    @property
    def device(self) -> torch.device:
        """Device the batch tensors live on."""
        return self.slot_mapping.device

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> AttnMetadata:
        """Return this metadata with every tensor on ``device`` (``self`` if already there)."""
        target = torch.device(device)
        if _on_device(self.slot_mapping, target):
            return self
        return replace(
            self,
            slot_mapping=self.slot_mapping.to(target, non_blocking=non_blocking),
            block_tables=self.block_tables.to(target, non_blocking=non_blocking),
            context_lens=self.context_lens.to(target, non_blocking=non_blocking),
            query_start_loc=self.query_start_loc.to(target, non_blocking=non_blocking),
        )

    def query_lens(self) -> list[int]:
        """Per-sequence query lengths, read back from ``query_start_loc``.

        Synchronises with the device; intended for tests, logging and the reference
        attention path, never for a kernel's hot loop.
        """
        locs = self.query_start_loc.tolist()
        return [locs[i + 1] - locs[i] for i in range(len(locs) - 1)]

    def validate(self, *, block_size: int | None = None) -> None:
        """Raise :class:`ValueError` if any documented invariant is violated."""
        for name, tensor, ndim in (
            ("slot_mapping", self.slot_mapping, 1),
            ("block_tables", self.block_tables, 2),
            ("context_lens", self.context_lens, 1),
            ("query_start_loc", self.query_start_loc, 1),
        ):
            if tensor.dim() != ndim:
                raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(tensor.shape)}")
            if tensor.dtype != torch.long:
                raise ValueError(f"{name} must be int64, got {tensor.dtype}")
        if self.num_prefill_seqs < 0 or self.num_decode_seqs < 0:
            raise ValueError("sequence counts must be non-negative")
        if self.block_tables.shape[0] != self.num_seqs:
            raise ValueError(
                f"block_tables has {self.block_tables.shape[0]} rows "
                f"but the batch holds {self.num_seqs} sequences"
            )
        if self.context_lens.shape[0] != self.num_seqs:
            raise ValueError(
                f"context_lens has {self.context_lens.shape[0]} entries "
                f"but the batch holds {self.num_seqs} sequences"
            )
        if self.query_start_loc.shape[0] != self.num_seqs + 1:
            raise ValueError(
                f"query_start_loc must have num_seqs + 1 = {self.num_seqs + 1} entries, "
                f"got {self.query_start_loc.shape[0]}"
            )
        locs = self.query_start_loc.tolist()
        if locs[0] != 0:
            raise ValueError(f"query_start_loc must start at 0, got {locs[0]}")
        if any(b < a for a, b in zip(locs[:-1], locs[1:], strict=True)):
            raise ValueError("query_start_loc must be non-decreasing")
        if locs[-1] != self.num_tokens:
            raise ValueError(
                f"query_start_loc ends at {locs[-1]} but slot_mapping holds "
                f"{self.num_tokens} tokens"
            )
        query_lens = [b - a for a, b in zip(locs[:-1], locs[1:], strict=True)]
        if query_lens and max(query_lens) != self.max_query_len:
            raise ValueError(
                f"max_query_len={self.max_query_len} disagrees with the batch ({max(query_lens)})"
            )
        contexts = self.context_lens.tolist()
        if contexts and max(contexts) != self.max_context_len:
            raise ValueError(
                f"max_context_len={self.max_context_len} disagrees with the batch ({max(contexts)})"
            )
        for seq, (ctx, q_len) in enumerate(zip(contexts, query_lens, strict=True)):
            if ctx < q_len:
                raise ValueError(
                    f"sequence {seq} attends to {ctx} tokens but contributes {q_len} queries"
                )
            if block_size is None:
                continue
            needed = -(-ctx // block_size)
            row = self.block_tables[seq].tolist()
            valid = [b for b in row[:needed] if b != PAD_BLOCK]
            if len(valid) != needed:
                raise ValueError(
                    f"sequence {seq} needs {needed} blocks for {ctx} tokens "
                    f"but its block table holds {len(valid)}"
                )


@dataclass(slots=True)
class LoRAContext:
    """Which LoRA adapter each token in a packed batch belongs to.

    The engine never groups a batch by adapter: continuous batching would collapse if a
    step could only contain one tenant's adapter. Instead every token carries its GPU slot
    index and :mod:`turboserve.engine.lora` sorts the batch by slot and runs one grouped
    matmul pair per active slot (SGMV-style), skipping :data:`NO_LORA`.
    """

    token_lora_slot: torch.Tensor
    """``[num_tokens]`` int64. GPU slot per token; :data:`NO_LORA` means base weights."""

    active_slots: list[int] = field(default_factory=list)
    """Sorted, de-duplicated non-zero slots present in this batch.

    Precomputed on the host so the LoRA layers do not have to run a device-side unique
    (and synchronise) once per linear per layer.
    """

    @property
    def num_tokens(self) -> int:
        """Number of tokens described by this context."""
        return int(self.token_lora_slot.shape[0])

    @property
    def is_base_only(self) -> bool:
        """Whether no adapter is active, in which case LoRA layers are a pass-through."""
        return not self.active_slots

    @property
    def device(self) -> torch.device:
        """Device ``token_lora_slot`` lives on."""
        return self.token_lora_slot.device

    @classmethod
    def base_only(cls, num_tokens: int, *, device: torch.device | str = "cpu") -> LoRAContext:
        """Build a context in which every token uses the base model."""
        return cls(
            token_lora_slot=torch.full(
                (num_tokens,), NO_LORA, dtype=torch.long, device=torch.device(device)
            ),
            active_slots=[],
        )

    @classmethod
    def from_slots(cls, slots: Sequence[int], *, device: torch.device | str = "cpu") -> LoRAContext:
        """Build a context from a per-token slot list, deriving ``active_slots``."""
        tensor = torch.tensor(list(slots), dtype=torch.long, device=torch.device(device))
        return cls(
            token_lora_slot=tensor,
            active_slots=sorted({int(slot) for slot in slots if slot != NO_LORA}),
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> LoRAContext:
        """Return this context with its tensor on ``device`` (``self`` if already there)."""
        target = torch.device(device)
        if _on_device(self.token_lora_slot, target):
            return self
        return replace(
            self, token_lora_slot=self.token_lora_slot.to(target, non_blocking=non_blocking)
        )

    def validate(self) -> None:
        """Raise :class:`ValueError` if the tensor and ``active_slots`` disagree."""
        if self.token_lora_slot.dim() != 1:
            raise ValueError(
                f"token_lora_slot must be 1-D, got shape {tuple(self.token_lora_slot.shape)}"
            )
        if self.token_lora_slot.dtype != torch.long:
            raise ValueError(f"token_lora_slot must be int64, got {self.token_lora_slot.dtype}")
        if any(slot <= NO_LORA for slot in self.active_slots):
            raise ValueError(f"active_slots must all be > {NO_LORA}, got {self.active_slots}")
        if len(set(self.active_slots)) != len(self.active_slots):
            raise ValueError(f"active_slots contains duplicates: {self.active_slots}")
        present = {int(slot) for slot in self.token_lora_slot.tolist() if slot != NO_LORA}
        if present - set(self.active_slots):
            raise ValueError(
                f"tokens reference slots {sorted(present - set(self.active_slots))} "
                "that are not listed in active_slots"
            )


class SchedulerConfig(BaseModel):
    """Limits and policy of one scheduler step.

    ``max_num_batched_tokens`` is the compute budget of a step and ``max_num_seqs`` the
    memory/bookkeeping budget; chunked prefill exists so that a single long prompt is
    split across steps instead of blocking every decoding sequence for the duration of its
    prefill. ``num_blocks`` is ``None`` by default, meaning "profile the device and size
    the KV cache from :attr:`EngineConfig.gpu_memory_utilization`"; an explicit value is
    what tests use to force preemption with a deliberately tiny cache.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_num_seqs: int = Field(default=64, ge=1)
    max_num_batched_tokens: int = Field(default=2048, ge=1)
    block_size: int = Field(default=16, ge=1)
    num_blocks: int | None = Field(default=None, ge=1)
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    policy: SchedulerPolicy = "fcfs"
    tenant_weights: dict[str, float] = Field(default_factory=dict)
    """Relative service weights per tenant, used only by the ``tenant_fair`` policy."""

    @model_validator(mode="after")
    def _check(self) -> SchedulerConfig:
        if self.block_size & (self.block_size - 1):
            raise ValueError(f"block_size must be a power of two, got {self.block_size}")
        if self.max_num_batched_tokens < self.block_size:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be at least "
                f"one block ({self.block_size}), otherwise no prefill chunk can ever fill a block"
            )
        if any(weight <= 0 for weight in self.tenant_weights.values()):
            raise ValueError("tenant weights must be positive")
        return self


class EngineConfig(BaseModel):
    """Everything needed to build an engine, apart from the fleet data in ``configs/``.

    Speculative decoding and multi-LoRA are configured through loosely typed dicts so that
    the groups that own those subsystems can define their own validated models without a
    coordinated change to this shared contract; the engine passes the dict straight to
    them and they validate it.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str
    """HuggingFace repo id or local path of the target model."""

    tokenizer: str | None = None
    """Tokenizer repo id or path; defaults to :attr:`model` when ``None``."""

    dtype: DTypeName = "auto"
    device: DeviceName = "auto"
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    max_model_len: int | None = Field(default=None, ge=1)
    seed: int | None = None
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    speculative: dict[str, Any] | None = None
    """Speculative-decoding options, validated by :mod:`turboserve.engine.spec`."""

    lora: dict[str, Any] | None = None
    """Multi-LoRA options, validated by :mod:`turboserve.engine.lora`."""

    @property
    def block_size(self) -> int:
        """KV block size, in tokens (owned by the scheduler config)."""
        return self.scheduler.block_size

    @property
    def num_blocks(self) -> int | None:
        """Explicit KV block count, or ``None`` to size it by memory profiling."""
        return self.scheduler.num_blocks

    def resolved_device(self) -> str:
        """Return ``"cuda"`` or ``"cpu"``, probing torch only when ``device == "auto"``."""
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_dtype(self) -> torch.dtype:
        """Return a concrete dtype: fp16 on CUDA, fp32 on CPU when ``dtype == "auto"``.

        bf16 is never chosen automatically because pre-Ampere CUDA devices have no bf16
        tensor cores; an H100 run sets ``dtype="bfloat16"`` explicitly.
        """
        if self.dtype != "auto":
            return resolve_dtype(self.dtype)
        return torch.float16 if self.resolved_device() == "cuda" else torch.float32

    @classmethod
    def from_settings(cls, settings: Settings) -> EngineConfig:
        """Build an engine config from the process-wide ``TURBOSERVE_*`` settings."""
        return cls(
            model=settings.model,
            dtype=settings.dtype,
            device=settings.device,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            scheduler=SchedulerConfig(
                max_num_seqs=settings.max_num_seqs,
                max_num_batched_tokens=settings.max_num_batched_tokens,
                block_size=settings.block_size,
                num_blocks=settings.num_blocks,
                enable_chunked_prefill=settings.enable_chunked_prefill,
                enable_prefix_caching=settings.enable_prefix_caching,
                policy=settings.scheduler_policy,
            ),
        )


@dataclass(slots=True)
class RequestOutput:
    """One engine step's worth of progress on one request.

    The engine emits deltas, not accumulated text: the gateway forwards them as SSE chunks
    and the benchmark client timestamps each one to build the inter-token-latency series,
    so re-sending the whole text every step would be both wasteful and useless for ITL.
    Token counts, in contrast, are cumulative totals for the request, because that is what
    a ``usage`` block must report and what a late subscriber needs.
    """

    request_id: str
    new_token_ids: list[int] = field(default_factory=list)
    text_delta: str = ""
    finished: bool = False
    finish_reason: FinishReason | None = None
    timing: RequestTiming = field(default_factory=RequestTiming)
    prompt_tokens: int = 0
    """Total prompt length in tokens, including tokens served from the prefix cache."""

    output_tokens: int = 0
    """Total tokens generated for this request so far, not just in this step."""

    cached_prompt_tokens: int = 0
    """Prompt tokens whose KV came from the prefix cache instead of being recomputed."""

    @property
    def total_tokens(self) -> int:
        """Prompt plus generated tokens, the figure billed in an OpenAI ``usage`` block."""
        return self.prompt_tokens + self.output_tokens

    def usage(self) -> dict[str, int]:
        """Token accounting in the shape the gateway puts into a response ``usage`` field."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
        }
