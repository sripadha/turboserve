"""The unit of work the scheduler moves around: one request, one :class:`Sequence`.

A ``Sequence`` is the whole server-side state of a single generation request: its tokens,
the KV blocks holding their attention state, how far the engine has got through them, and
the timestamps the gateway and the benchmark client report. Everything the scheduler does
is expressed as changes to these fields, which is why they live in one small module with
no dependency on a model, a device or a tokenizer.

Two ideas carry most of the weight:

* **``num_computed_tokens`` is the only progress counter.** It counts tokens whose KV is
  already in the cache, whether they got there by a prefill chunk, by a full prefill, by a
  decode step or by a prefix-cache hit. The number of tokens still to compute is therefore
  always ``num_tokens - num_computed_tokens``, which makes prefill, chunked prefill and
  decode the same arithmetic instead of three special cases: a decoding sequence simply
  has exactly one uncomputed token, the one it just sampled.
* **Preemption is recompute, not swap.** :meth:`Sequence.reset_for_recompute` throws the
  progress away and returns the sequence to the waiting queue. That is cheap to implement
  and, with the prefix cache in front of it, usually cheap to undo as well, because the
  blocks the sequence had just computed are still in the cache when it is resumed. See
  ``docs/adr/`` for the decision record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import torch

from turboserve.engine.core.types import (
    NO_LORA,
    FinishReason,
    RequestOutput,
    RequestTiming,
    SamplingParams,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

__all__ = ["DEFAULT_TENANT", "SeqStatus", "Sequence"]

#: Tenant used when a caller does not supply one. The gateway always supplies one; the
#: default exists so that engine-level tests and single-tenant benchmarks stay readable.
DEFAULT_TENANT = "default"


class SeqStatus(StrEnum):
    """Where a sequence is in the scheduler's lifecycle.

    A :class:`~enum.StrEnum` so the value can be used directly as a Prometheus label and
    in JSON without a converter. The transitions the scheduler performs are
    ``WAITING -> RUNNING``, ``RUNNING -> PREEMPTED`` (blocks reclaimed, progress reset),
    ``PREEMPTED -> RUNNING`` (resumed ahead of the waiting queue) and
    ``* -> FINISHED``, which is terminal.
    """

    WAITING = "waiting"
    """Queued, holding no KV blocks."""

    RUNNING = "running"
    """Admitted: holds blocks and is scheduled for prefill chunks or decode steps."""

    PREEMPTED = "preempted"
    """Was running, lost its blocks to a higher-priority sequence, will recompute."""

    FINISHED = "finished"
    """Terminal: stopped, hit ``max_tokens``, or was aborted."""

    @property
    def is_terminal(self) -> bool:
        """Whether no further transition out of this status is possible."""
        return self is SeqStatus.FINISHED


@dataclass(slots=True, eq=False)
class Sequence:
    """One request's tokens, KV blocks, progress and timing.

    ``eq=False`` keeps the default identity hash, so sequences can go into sets and dict
    keys while still being mutable: the scheduler holds the same object in its queues, in
    its ``request_id`` index and inside the step output, and those must be one object, not
    copies that drift apart.

    Invariants maintained by this class and relied on by the scheduler and block manager:

    * ``0 <= num_computed_tokens <= num_tokens``.
    * ``len(block_table) * block_size >= num_computed_tokens`` (the block manager's job).
    * ``status is FINISHED`` implies ``finish_reason is not None`` and vice versa.
    * ``output_token_ids`` only ever grows, except across
      :meth:`reset_for_recompute`, which does not touch it -- preemption discards
      *computed KV*, never generated text.
    """

    seq_id: int
    """Engine-local monotonically increasing id; also the tie-break for preemption."""

    request_id: str
    """Client-facing id, unique among unfinished requests; what ``abort`` addresses."""

    prompt_token_ids: list[int]
    """Tokenised prompt. Never empty: an empty prompt has no position to sample from."""

    sampling: SamplingParams = field(default_factory=SamplingParams)
    tenant_id: str = DEFAULT_TENANT
    priority: int = 0
    """Higher runs first when the scheduler has to pick a victim to preempt."""

    lora_id: int = NO_LORA
    """GPU adapter slot; :data:`~turboserve.engine.core.types.NO_LORA` means base weights.

    It is part of the prefix-cache hash chain: two tenants with the same prompt but
    different adapters must not share KV blocks, because the adapter changes the K/V the
    attention layers produce.
    """

    output_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    """KV block ids in position order; owned by the block manager, not by this class."""

    num_computed_tokens: int = 0
    """Tokens whose KV is in the cache: prefix hits plus chunked-prefill/decode progress."""

    status: SeqStatus = SeqStatus.WAITING
    finish_reason: FinishReason | None = None
    stop_token_id: int | None = None
    """The token that triggered a ``stop`` finish, for clients that report it."""

    timing: RequestTiming = field(default_factory=RequestTiming)
    generator: torch.Generator | None = None
    """Per-request RNG, present only when ``sampling.seed`` is set. See
    :meth:`ensure_generator` for why it is per request rather than global."""

    num_preemptions: int = 0
    """How often this sequence lost its blocks; surfaced in the scheduler's stats."""

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError(f"sequence {self.request_id!r} has an empty prompt")
        if self.num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be non-negative")
        if self.lora_id < NO_LORA:
            raise ValueError(f"lora_id must be >= {NO_LORA}, got {self.lora_id}")
        if self.sampling.seed is not None and self.generator is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.sampling.seed)

    # -- token accounting ---------------------------------------------------------------

    @property
    def num_prompt_tokens(self) -> int:
        """Length of the prompt in tokens."""
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        """Tokens generated so far."""
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Prompt plus generated tokens: the sequence's current length."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        """Tokens whose KV still has to be computed, prefill and decode alike."""
        return self.num_tokens - self.num_computed_tokens

    @property
    def token_ids(self) -> list[int]:
        """Prompt and output concatenated. Builds a new list; see :meth:`block_tokens`."""
        return [*self.prompt_token_ids, *self.output_token_ids]

    @property
    def is_prefill(self) -> bool:
        """Whether the next scheduled tokens are still prompt tokens."""
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        """Whether the sequence reached a terminal state."""
        return self.status is SeqStatus.FINISHED

    def token_at(self, position: int) -> int:
        """Token at an absolute position, without materialising :attr:`token_ids`."""
        if not 0 <= position < self.num_tokens:
            raise IndexError(f"position {position} out of range for {self.num_tokens} tokens")
        prompt_len = len(self.prompt_token_ids)
        if position < prompt_len:
            return self.prompt_token_ids[position]
        return self.output_token_ids[position - prompt_len]

    def tokens_in_range(self, start: int, end: int) -> list[int]:
        """Tokens at absolute positions ``[start, end)``, prompt and output stitched."""
        if start < 0 or end > self.num_tokens or start > end:
            raise ValueError(
                f"range [{start}, {end}) is not inside [0, {self.num_tokens}) "
                f"for sequence {self.request_id!r}"
            )
        prompt_len = len(self.prompt_token_ids)
        if end <= prompt_len:
            return self.prompt_token_ids[start:end]
        if start >= prompt_len:
            return self.output_token_ids[start - prompt_len : end - prompt_len]
        return [*self.prompt_token_ids[start:], *self.output_token_ids[: end - prompt_len]]

    def block_tokens(self, index: int, block_size: int) -> list[int]:
        """The ``block_size`` tokens covered by block ``index``.

        Only complete blocks can be hashed into the prefix cache, so this refuses to
        return a short tail: a half-filled block will still receive tokens and hashing it
        early would cache KV under a hash that does not describe its final contents.
        """
        start = index * block_size
        end = start + block_size
        if end > self.num_tokens:
            raise ValueError(
                f"block {index} of sequence {self.request_id!r} is not full: "
                f"needs tokens up to {end}, sequence has {self.num_tokens}"
            )
        return self.tokens_in_range(start, end)

    def num_blocks_needed(self, block_size: int) -> int:
        """Blocks required to hold the whole sequence's KV, including a partial tail."""
        return -(-self.num_tokens // block_size)

    def num_new_tokens_to_compute(self, budget: int) -> int:
        """How many tokens this sequence would compute next, capped by ``budget``.

        One expression covers every phase: a fresh prompt returns its whole length (or the
        budget, which is a prefill chunk), a partially prefilled prompt returns the rest,
        and a decoding sequence returns 1 because the token it sampled last step is its
        only uncomputed token. Finished sequences return 0.
        """
        if self.is_finished or budget <= 0:
            return 0
        return max(0, min(budget, self.num_uncomputed_tokens))

    def advance_computed(self, num_tokens: int) -> int:
        """Record that ``num_tokens`` more tokens now have KV in the cache."""
        if num_tokens < 0:
            raise ValueError(f"cannot advance by {num_tokens} tokens")
        target = self.num_computed_tokens + num_tokens
        if target > self.num_tokens:
            raise ValueError(
                f"sequence {self.request_id!r} would have {target} computed tokens "
                f"but only holds {self.num_tokens}"
            )
        self.num_computed_tokens = target
        return target

    def reset_for_recompute(self) -> None:
        """Forget all computed KV so the sequence can be scheduled again from scratch.

        The caller (the block manager) frees the blocks; this only resets the bookkeeping
        and counts the preemption. ``output_token_ids`` survives: the sequence will
        recompute the KV of the tokens it already emitted, not re-emit them.
        """
        self.num_computed_tokens = 0
        self.block_table.clear()
        self.timing.num_cached_prompt_tokens = 0
        self.num_preemptions += 1
        self.status = SeqStatus.PREEMPTED

    # -- generation ---------------------------------------------------------------------

    def append_token(self, token_id: int, *, now: float | None = None) -> None:
        """Append a sampled token and stamp the first-token timestamp when it is the first."""
        if self.is_finished:
            raise ValueError(
                f"sequence {self.request_id!r} finished with {self.finish_reason}; "
                "no further tokens can be appended"
            )
        self.output_token_ids.append(int(token_id))
        if self.timing.t_first_token is None:
            self.timing.t_first_token = time.perf_counter() if now is None else now

    def check_stop(self, token_id: int, *, eos_token_id: int | None = None) -> FinishReason | None:
        """Decide whether the just-appended ``token_id`` ends the sequence.

        Stop *strings* are deliberately not handled here: matching them needs detokenised
        text, which lives in the runtime's incremental detokeniser, and a token id cannot
        tell whether it completed a multi-byte character. This function owns the three
        conditions that are decidable from ids alone.
        """
        if token_id in self.sampling.stop_token_ids:
            return FinishReason.STOP
        if eos_token_id is not None and token_id == eos_token_id and not self.sampling.ignore_eos:
            return FinishReason.STOP
        if self.num_output_tokens >= self.sampling.max_tokens:
            return FinishReason.LENGTH
        return None

    def finish(
        self,
        reason: FinishReason,
        *,
        now: float | None = None,
        stop_token_id: int | None = None,
    ) -> None:
        """Move to the terminal state, stamping ``t_finish`` once.

        Idempotent for the same reason, because a client disconnect and a natural stop can
        race: the first reason recorded wins, so a finished request is never re-labelled.
        """
        if self.is_finished:
            return
        self.status = SeqStatus.FINISHED
        self.finish_reason = reason
        self.stop_token_id = stop_token_id
        self.timing.t_finish = time.perf_counter() if now is None else now

    def make_output(
        self,
        *,
        new_token_ids: Iterable[int] | None = None,
        text_delta: str = "",
    ) -> RequestOutput:
        """Build the per-step delta the gateway streams to the client.

        Token *ids* are a delta (what this step produced) while the counts are cumulative,
        matching :class:`~turboserve.engine.core.types.RequestOutput`'s contract.
        """
        return RequestOutput(
            request_id=self.request_id,
            new_token_ids=list(new_token_ids) if new_token_ids is not None else [],
            text_delta=text_delta,
            finished=self.is_finished,
            finish_reason=self.finish_reason,
            timing=self.timing,
            prompt_tokens=self.num_prompt_tokens,
            output_tokens=self.num_output_tokens,
            cached_prompt_tokens=self.timing.num_cached_prompt_tokens,
        )

    # -- sampling RNG -------------------------------------------------------------------

    def ensure_generator(self, device: torch.device | str = "cpu") -> torch.Generator | None:
        """Return this request's RNG, creating it on ``device`` the first time.

        Seeded requests get their own generator rather than sharing the global one because
        continuous batching interleaves them: with a shared RNG the tokens a request gets
        would depend on which other requests happened to be in the same batch, and a
        seeded request would not be reproducible. ``torch.multinomial`` requires the
        generator and the probability tensor to live on the same device, so the generator
        is (re)created for the device sampling actually happens on.
        """
        seed = self.sampling.seed
        if seed is None:
            return None
        target = torch.device(device)
        current = self.generator
        if (
            current is not None
            and current.device.type == target.type
            and (target.index is None or current.device.index == target.index)
        ):
            return current
        generator = torch.Generator(device=target)
        generator.manual_seed(seed)
        self.generator = generator
        return generator

    def __repr__(self) -> str:
        return (
            f"Sequence(seq_id={self.seq_id}, request_id={self.request_id!r}, "
            f"tenant={self.tenant_id!r}, status={self.status.value}, "
            f"prompt={self.num_prompt_tokens}, output={self.num_output_tokens}, "
            f"computed={self.num_computed_tokens}, blocks={len(self.block_table)})"
        )
