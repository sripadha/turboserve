"""Drafters: cheap guesses at the next ``k`` tokens, and the KV bookkeeping they need.

A drafter answers one question -- "what do you think the next ``k`` tokens of each of these
sequences are?" -- and reports how confident it is, because verification needs the draft
*distribution*, not only the draft tokens. The two implementations in this package answer it
very differently:

* :class:`ModelDrafter` runs a second, smaller causal LM. It is a miniature engine in its
  own right: its own weights, its own paged KV cache, its own block manager, and one mirror
  :class:`~turboserve.engine.core.sequence.Sequence` per target sequence so that the draft
  model's cache is grown and rewound independently of the target's.
* :class:`~turboserve.engine.spec.ngram.NgramDrafter` runs no model at all.

Both satisfy the :class:`Drafter` protocol, so
:class:`~turboserve.engine.spec.spec_engine.SpeculativeLLMEngine` never learns which one it
has.

**Why a mirror sequence rather than the target's own.** The draft model has its own layer
count, KV head count and head dimension, so it cannot share the target's blocks; and its
cache has to survive a rejection. When the target rejects draft tokens, the draft model has
already written their KV. :meth:`ModelDrafter.propose` therefore recomputes, on every call,
how much of what it has cached is still a prefix of what the target actually holds, and
rewinds the mirror to that point. The check is against the token ids themselves, so the
drafter needs no report of the verification outcome and cannot drift out of step with the
target if one were lost.

**The invariant everything else rests on.** Between calls, a mirror's KV covers all but the
last of its tokens (``num_computed_tokens == num_tokens - 1``). That last token is the query
of the next draft step, whose output is the next draft token -- so a draft step is exactly an
ordinary decode step, and ``k`` of them cost ``k`` small forward passes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from turboserve.engine.core.block_manager import BlockManager
from turboserve.engine.core.sampler import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
)
from turboserve.engine.core.scheduler import ScheduledSeq, SchedulerOutput
from turboserve.engine.core.sequence import Sequence
from turboserve.engine.core.types import SamplingParams
from turboserve.engine.runtime.worker import ModelRunner

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable
    from collections.abc import Sequence as SequenceABC
    from pathlib import Path

    from turboserve.engine.core.kv_cache import KVCache
    from turboserve.engine.model.model import CausalLM

logger = logging.getLogger(__name__)

__all__ = ["DraftProposal", "Drafter", "ModelDrafter"]


@dataclass(frozen=True, slots=True)
class DraftProposal:
    """What a drafter proposes for one step, for every sequence it was given.

    ``token_ids`` is rectangular and padded so the shape is predictable, and ``lengths`` says
    how much of each row is real. Ragged proposals are the normal case, not an edge case: an
    n-gram drafter finds no continuation for some sequences, a model drafter runs out of
    draft blocks for others, and a sequence near its length limit gets a shortened draft. A
    row of length zero means "no speculation for this sequence this step", which the engine
    turns into an ordinary single-token decode.

    ``probs`` is ``None`` when the drafter proposes deterministically. The verifier reads
    that as a point mass on the drafted token, which is exact -- see
    :mod:`turboserve.engine.spec.verifier`.
    """

    token_ids: torch.Tensor
    """``[num_seqs, max_drafts]`` int64 token ids, padded with zeros, on the host."""

    lengths: tuple[int, ...]
    """Valid draft length of each row."""

    probs: torch.Tensor | None = None
    """``[num_seqs, max_drafts, vocab]`` draft distributions, on the drafter's device."""

    def __post_init__(self) -> None:
        if self.token_ids.dim() != 2:
            raise ValueError(
                f"token_ids must be [num_seqs, max_drafts], got {tuple(self.token_ids.shape)}"
            )
        rows, width = (int(dim) for dim in self.token_ids.shape)
        if len(self.lengths) != rows:
            raise ValueError(f"{len(self.lengths)} lengths for {rows} rows of draft tokens")
        if any(length < 0 or length > width for length in self.lengths):
            raise ValueError(f"draft lengths {self.lengths} do not fit a width of {width}")
        if self.probs is not None and tuple(self.probs.shape[:2]) != (rows, width):
            raise ValueError(
                f"probs must start with [{rows}, {width}], got {tuple(self.probs.shape)}"
            )

    @property
    def num_seqs(self) -> int:
        """Sequences this proposal covers, including the ones it proposes nothing for."""
        return int(self.token_ids.shape[0])

    @property
    def max_drafts(self) -> int:
        """Width of the padded token matrix."""
        return int(self.token_ids.shape[1])

    @property
    def total_drafts(self) -> int:
        """Number of real draft tokens proposed across all sequences."""
        return sum(self.lengths)

    @property
    def is_empty(self) -> bool:
        """Whether nothing at all was proposed."""
        return self.total_drafts == 0

    def tokens_for(self, index: int) -> list[int]:
        """The real draft tokens of row ``index``, as Python ints."""
        return [int(token) for token in self.token_ids[index, : self.lengths[index]].tolist()]

    def probs_for(self, index: int) -> torch.Tensor | None:
        """The ``[length, vocab]`` draft distributions of row ``index``, or ``None``."""
        if self.probs is None:
            return None
        return self.probs[index, : self.lengths[index]]

    @classmethod
    def empty(cls, num_seqs: int) -> DraftProposal:
        """A proposal that speculates on nothing, so every sequence decodes normally."""
        return cls(
            token_ids=torch.zeros((num_seqs, 0), dtype=torch.long),
            lengths=(0,) * num_seqs,
        )

    @classmethod
    def from_lists(
        cls,
        token_lists: SequenceABC[SequenceABC[int]],
        *,
        probs: torch.Tensor | None = None,
    ) -> DraftProposal:
        """Build a padded proposal from per-sequence token lists.

        ``probs`` may be wider than the widest row (the drafter allocates it for the
        configured ``k`` before it knows how many tokens it will manage); it is trimmed here
        rather than at every call site.
        """
        lengths = tuple(len(tokens) for tokens in token_lists)
        width = max(lengths, default=0)
        matrix = torch.zeros((len(token_lists), width), dtype=torch.long)
        for row, tokens in enumerate(token_lists):
            if tokens:
                matrix[row, : len(tokens)] = torch.tensor(
                    [int(token) for token in tokens], dtype=torch.long
                )
        return cls(
            token_ids=matrix,
            lengths=lengths,
            probs=None if probs is None else probs[:, :width],
        )


@runtime_checkable
class Drafter(Protocol):
    """Proposes continuations for the sequences that are about to decode.

    Implementations must be pure with respect to the *target* sequences: they may read them
    but must not append tokens, touch ``num_computed_tokens`` or take blocks from the
    engine's pool. Everything a drafter needs to remember between steps belongs to the
    drafter.
    """

    name: str
    """Short identifier recorded in benchmark results and engine statistics."""

    def propose(self, seqs: SequenceABC[Sequence], k: int) -> DraftProposal:
        """Propose up to ``k`` tokens for each sequence, in the order they were given."""
        ...

    def release(self, seq_id: int) -> None:
        """Forget a finished or aborted sequence and free whatever it held."""
        ...

    def prune(self, live_seq_ids: Iterable[int]) -> int:
        """Forget every sequence that is no longer in flight; returns how many."""
        ...

    def reset(self) -> None:
        """Forget every sequence."""
        ...

    def stats(self) -> dict[str, int | float]:
        """Counters for the engine's ``stats()`` output."""
        ...


@dataclass(slots=True)
class _Row:
    """One sequence's participation in a :meth:`ModelDrafter.propose` call."""

    index: int
    """Position of the sequence in the caller's list, so results land in the right row."""

    target: Sequence
    mirror: Sequence
    budget: int
    """How many more tokens this sequence may be drafted; zero retires it from the call."""

    tokens: list[int] = field(default_factory=list)


class ModelDrafter:
    """Draft with a second, smaller causal LM that has its own paged KV cache.

    The drafter is driven exactly like the main engine -- packed variable-length batches,
    slot mappings, block tables -- because it *is* the same machinery: a
    :class:`~turboserve.engine.runtime.worker.ModelRunner` over a
    :class:`~turboserve.engine.core.block_manager.BlockManager`. Only the scheduling differs,
    and it is much simpler: every sequence in the call advances together, one token per draft
    step, for ``k`` steps.

    The prefix cache is deliberately *not* enabled on the draft pool. Draft blocks live only
    as long as their target request and are never shared between requests, so a cache would
    add hashing work and an eviction policy for no hits.
    """

    name = "model"

    def __init__(
        self,
        model: CausalLM,
        kv_cache: KVCache,
        *,
        block_manager: BlockManager | None = None,
        max_num_batched_tokens: int = 2048,
        max_model_len: int | None = None,
    ) -> None:
        """Wrap a loaded draft model and its cache.

        Args:
            model: the draft model. Its vocabulary must match the target's; the speculative
                engine checks that, because a mismatch produces plausible-looking nonsense
                rather than an error.
            kv_cache: the draft model's own KV pool, sized independently of the target's.
            block_manager: an already-built manager over that pool. Defaults to a fresh one
                with prefix caching off.
            max_num_batched_tokens: token budget for the catch-up forward passes that bring a
                newly seen sequence's draft cache up to date. Draft steps themselves are one
                token per sequence and are never chunked.
            max_model_len: refuse to draft past this many tokens, for a draft model whose
                context window is shorter than the target's.
        """
        if max_num_batched_tokens < 1:
            raise ValueError(
                f"max_num_batched_tokens must be positive, got {max_num_batched_tokens}"
            )
        if max_model_len is not None and max_model_len < 1:
            raise ValueError(f"max_model_len must be positive, got {max_model_len}")
        self.model = model
        self.runner = ModelRunner(model, kv_cache)
        self.block_manager = block_manager or BlockManager.create(
            kv_cache.num_blocks, kv_cache.block_size, enable_prefix_caching=False
        )
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self._mirrors: dict[int, Sequence] = {}
        self._num_proposals = 0
        self._num_draft_tokens = 0
        self._num_forwards = 0
        self._num_starved = 0

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        num_blocks: int | None = None,
        block_size: int = 16,
        gpu_memory_utilization: float = 0.15,
        max_num_batched_tokens: int = 2048,
        max_model_len: int | None = None,
        local_files_only: bool = False,
    ) -> ModelDrafter:
        """Load a draft model and give it a KV pool of its own.

        ``num_blocks`` sizes the pool exactly; leaving it ``None`` profiles the device and
        takes ``gpu_memory_utilization`` of it. Either way the pool is built before the target
        engine profiles memory, so the target sees the draft model's weights and blocks as
        already spent -- the only ordering in which both fit on one device without the caller
        doing the arithmetic by hand.
        """
        from turboserve.engine.core.types import SchedulerConfig
        from turboserve.engine.model.model import CausalLM
        from turboserve.engine.runtime.memory import build_kv_cache, size_kv_cache

        model = CausalLM.from_pretrained(
            path_or_id, dtype=dtype, device=device, local_files_only=local_files_only
        )
        sizing = size_kv_cache(
            model.config,
            SchedulerConfig(
                block_size=block_size,
                num_blocks=num_blocks,
                max_num_batched_tokens=max(max_num_batched_tokens, block_size),
            ),
            device=model.device,
            dtype=model.dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            num_blocks=num_blocks,
            max_model_len=max_model_len,
        )
        cache = build_kv_cache(model.config, sizing, device=model.device, dtype=model.dtype)
        return cls(
            model,
            cache,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
        )

    # -- properties -----------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device the draft model and its cache live on."""
        return self.runner.device

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the draft model; must match the target's."""
        return self.model.config.vocab_size

    @property
    def num_tracked(self) -> int:
        """Sequences with a live mirror in the draft pool."""
        return len(self._mirrors)

    # -- the drafter protocol ---------------------------------------------------------------

    @torch.inference_mode()
    def propose(self, seqs: SequenceABC[Sequence], k: int) -> DraftProposal:
        """Run the draft model forward up to ``k`` times over all of ``seqs`` at once.

        Sequences the draft pool cannot fund are dropped from the batch and come back with a
        zero-length row rather than failing the step: running out of draft blocks must cost
        speculation, never correctness or availability.
        """
        if k <= 0 or not seqs:
            return DraftProposal.empty(len(seqs))
        rows = self._prepare(seqs, k)
        if not rows:
            return DraftProposal.empty(len(seqs))
        self._num_proposals += 1
        self._catch_up(rows)
        probs: torch.Tensor | None = None
        if any(not row.target.sampling.is_greedy for row in rows):
            probs = torch.zeros(
                (len(seqs), k, self.vocab_size), dtype=torch.float32, device=self.device
            )
        for step in range(k):
            active = self._active(rows, step)
            if not active:
                break
            step_probs = self._draft_step(active)
            if probs is not None:
                for position, row in enumerate(active):
                    probs[row.index, step] = step_probs[position]
        proposal = DraftProposal.from_lists(self._token_lists(len(seqs), rows), probs=probs)
        self._num_draft_tokens += proposal.total_drafts
        return proposal

    def release(self, seq_id: int) -> None:
        """Return a finished sequence's draft blocks to the draft pool."""
        mirror = self._mirrors.pop(seq_id, None)
        if mirror is not None:
            self.block_manager.free(mirror)

    def prune(self, live_seq_ids: Iterable[int]) -> int:
        """Release every mirror whose target is no longer in flight. Returns how many."""
        live = set(live_seq_ids)
        stale = [seq_id for seq_id in self._mirrors if seq_id not in live]
        for seq_id in stale:
            self.release(seq_id)
        return len(stale)

    def reset(self) -> None:
        """Drop every mirror and return the whole draft pool."""
        self._mirrors.clear()
        self.block_manager.reset()

    def stats(self) -> dict[str, int | float]:
        """Counters describing how much work the drafter has done."""
        return {
            "draft_proposals": self._num_proposals,
            "draft_tokens_proposed": self._num_draft_tokens,
            "draft_forwards": self._num_forwards,
            "draft_starved": self._num_starved,
            "draft_sequences": len(self._mirrors),
            "draft_blocks_free": self.block_manager.num_free_blocks,
        }

    # -- internals ---------------------------------------------------------------------------

    def _prepare(self, seqs: SequenceABC[Sequence], k: int) -> list[_Row]:
        """Build one row per draftable sequence, rewinding each mirror to the shared prefix."""
        rows: list[_Row] = []
        for index, seq in enumerate(seqs):
            budget = k
            if self.max_model_len is not None:
                budget = min(budget, self.max_model_len - seq.num_tokens)
            if budget <= 0:
                self._num_starved += 1
                continue
            mirror = self._mirror_for(seq)
            self._rewind(mirror, seq, max_rollback=k + 1)
            rows.append(_Row(index=index, target=seq, mirror=mirror, budget=budget))
        return rows

    def _mirror_for(self, seq: Sequence) -> Sequence:
        """The draft-side sequence shadowing ``seq``, created on first sight."""
        mirror = self._mirrors.get(seq.seq_id)
        if mirror is not None:
            return mirror
        # The mirror carries default sampling parameters on purpose: the *target* request's
        # parameters and generator are what drafting uses, so a seeded request draws from one
        # RNG stream rather than from two identically seeded ones.
        mirror = Sequence(
            seq_id=seq.seq_id,
            request_id=f"draft:{seq.request_id}",
            prompt_token_ids=list(seq.prompt_token_ids),
            sampling=SamplingParams(),
        )
        self.block_manager.allocate(mirror)
        self._mirrors[seq.seq_id] = mirror
        return mirror

    @staticmethod
    def _rewind(mirror: Sequence, target: Sequence, *, max_rollback: int) -> None:
        """Rewind the mirror to the longest prefix it shares with the target's tokens.

        Only tokens the drafter itself proposed can ever differ, so a divergence is always
        inside the last draft; scanning backwards from the cached length is therefore ``O(k)``
        rather than ``O(sequence length)``. A deeper divergence cannot happen, and if one
        somehow did the mirror is rebuilt from scratch rather than trusted.

        The result is also capped at one token short of the sequence, which is the invariant
        the draft steps need: there must always be one token left to use as a query. Giving up
        a token of cache here costs one position of recomputation and nothing else, because
        re-feeding a token writes the same KV into the same slot.
        """
        tokens = target.token_ids
        cached = mirror.token_ids
        length = min(mirror.num_computed_tokens, len(tokens) - 1, len(cached))
        limit = max(0, length - max_rollback)
        while length > limit and cached[length - 1] != tokens[length - 1]:
            length -= 1
        if length > 0 and length == limit and cached[length - 1] != tokens[length - 1]:
            logger.warning(
                "draft mirror for %s diverged further than one draft; recomputing it",
                target.request_id,
            )
            length = 0
        mirror.num_computed_tokens = max(length, 0)
        mirror.output_token_ids[:] = tokens[mirror.num_prompt_tokens :]

    def _active(self, rows: SequenceABC[_Row], step: int) -> list[_Row]:
        """The rows that may take draft step ``step``, given budgets and pool space.

        Blocks are reserved as the list is built: two rows asking for the last free block in
        the same step must not both be told yes.
        """
        reserved = 0
        active: list[_Row] = []
        for row in rows:
            if row.budget <= step:
                continue
            mirror = row.mirror
            if mirror.num_computed_tokens >= mirror.num_tokens:
                row.budget = step  # nothing left to use as a query; retire the row
                continue
            needed = self._blocks_needed(mirror, 1)
            if needed > self.block_manager.allocator.num_allocatable - reserved:
                row.budget = step
                self._num_starved += 1
                continue
            reserved += needed
            active.append(row)
        return active

    def _blocks_needed(self, mirror: Sequence, num_new_tokens: int) -> int:
        """How many new blocks ``num_new_tokens`` more computed tokens would require."""
        end = mirror.num_computed_tokens + num_new_tokens
        block_size = self.block_manager.block_size
        return max(0, -(-end // block_size) - len(mirror.block_table))

    def _catch_up(self, rows: SequenceABC[_Row]) -> None:
        """Feed every mirror the tokens it has not seen, up to its last known token.

        In steady state this does nothing: the previous call left the mirror exactly one
        token short of the target. It does real work once per sequence, when the drafter
        first sees it and has to process the prompt, and again after the target was preempted
        and resumed with tokens the drafter never fed.
        """
        while True:
            pending = [
                row
                for row in rows
                if row.budget > 0 and row.mirror.num_computed_tokens < row.mirror.num_tokens - 1
            ]
            if not pending:
                return
            budget = self.max_num_batched_tokens
            reserved = 0
            entries: list[tuple[_Row, int]] = []
            for row in pending:
                if budget <= 0:
                    break
                mirror = row.mirror
                want = min(budget, mirror.num_tokens - 1 - mirror.num_computed_tokens)
                free = self.block_manager.allocator.num_allocatable - reserved
                while want > 0 and self._blocks_needed(mirror, want) > free:
                    want -= 1
                if want <= 0:
                    row.budget = 0
                    self._num_starved += 1
                    continue
                reserved += self._blocks_needed(mirror, want)
                entries.append((row, want))
                budget -= want
            if not entries:
                return
            self._forward(self._pack(entries))

    def _draft_step(self, rows: SequenceABC[_Row]) -> torch.Tensor:
        """Feed one token per row, sample the next, and return the draft distributions.

        Returns ``[len(rows), vocab]`` probabilities even when every row is greedy; the
        caller drops them in that case. Producing the distribution and the token from one
        transform pass is why this does not simply call
        :class:`~turboserve.engine.core.sampler.Sampler`, which returns only the token.
        """
        out = self._pack([(row, 1) for row in rows])
        batch = self.runner.build_batch(out)
        hidden = self.runner.forward(batch)
        self._num_forwards += 1
        logits = self.model.compute_logits(hidden, batch.sample_indices)
        params = [row.target.sampling for row in rows]
        histories = [row.mirror.token_ids for row in rows]
        work = logits.detach().to(dtype=torch.float32, copy=True)
        if any(p.repetition_penalty != 1.0 for p in params):
            work = apply_repetition_penalty(work, [p.repetition_penalty for p in params], histories)
        work = apply_temperature(work, [p.temperature for p in params])
        work = apply_top_k(work, [p.top_k for p in params])
        work = apply_top_p(work, [p.top_p for p in params])
        probs = work.softmax(dim=-1)
        for position, token in enumerate(self._draw(work, probs, rows)):
            rows[position].tokens.append(token)
            rows[position].mirror.output_token_ids.append(token)
        return probs

    def _draw(
        self, logits: torch.Tensor, probs: torch.Tensor, rows: SequenceABC[_Row]
    ) -> list[int]:
        """Draw one draft token per row, greedy or multinomial, with the target's RNG.

        Mirrors :class:`~turboserve.engine.core.sampler.Sampler`'s own draw: unseeded rows
        share one ``multinomial`` call and each seeded row gets its own, so that a seeded
        request's draft does not depend on which requests happened to share its batch.
        """
        tokens = logits.argmax(dim=-1)
        stochastic = [
            position for position, row in enumerate(rows) if not row.target.sampling.is_greedy
        ]
        if not stochastic:
            return [int(token) for token in tokens.tolist()]
        device = probs.device
        shared: list[int] = []
        for position in stochastic:
            generator = rows[position].target.ensure_generator(device)
            if generator is None:
                shared.append(position)
                continue
            tokens[position] = torch.multinomial(probs[position], 1, generator=generator)[0]
        if shared:
            index = torch.tensor(shared, dtype=torch.long, device=device)
            drawn = torch.multinomial(probs.index_select(0, index), 1).squeeze(1)
            tokens = tokens.index_copy(0, index, drawn)
        return [int(token) for token in tokens.tolist()]

    def _pack(self, entries: SequenceABC[tuple[_Row, int]]) -> SchedulerOutput:
        """Turn ``(row, tokens to feed)`` pairs into a batch the model runner understands.

        Reuses :class:`~turboserve.engine.core.scheduler.SchedulerOutput` rather than building
        tensors by hand so that the draft model sees exactly the batch layout the target model
        sees, including the prefills-then-decodes ordering the attention metadata documents.
        """
        prefills: list[ScheduledSeq] = []
        decodes: list[ScheduledSeq] = []
        for row, num_new in entries:
            mirror = row.mirror
            start = mirror.num_computed_tokens
            end = start + num_new
            slots = self.block_manager.append_slots(mirror, num_new)
            mirror.advance_computed(num_new)
            is_prefill = start < mirror.num_prompt_tokens
            item = ScheduledSeq(
                seq=mirror,
                num_new_tokens=num_new,
                context_len=end,
                token_ids=mirror.tokens_in_range(start, end),
                slot_mapping=slots,
                block_table=list(mirror.block_table),
                is_prefill=is_prefill,
                is_chunk=is_prefill and end < mirror.num_prompt_tokens,
            )
            (prefills if is_prefill else decodes).append(item)
        out = SchedulerOutput(scheduled=[*prefills, *decodes])
        out.num_prefill_seqs = len(prefills)
        out.num_decode_seqs = len(decodes)
        out.num_prefill_tokens = sum(item.num_new_tokens for item in prefills)
        out.num_decode_tokens = sum(item.num_new_tokens for item in decodes)
        return out

    def _forward(self, out: SchedulerOutput) -> None:
        """Run the draft model over a batch purely to fill its KV cache."""
        self.runner.forward(self.runner.build_batch(out))
        self._num_forwards += 1

    @staticmethod
    def _token_lists(num_seqs: int, rows: SequenceABC[_Row]) -> list[list[int]]:
        """Per-sequence draft tokens, in the caller's order, with gaps filled by empties."""
        lists: list[list[int]] = [[] for _ in range(num_seqs)]
        for row in rows:
            lists[row.index] = row.tokens
        return lists

    def __repr__(self) -> str:
        return (
            f"ModelDrafter(model={self.model.config.architecture}, "
            f"blocks={self.block_manager.allocator.num_total}, tracked={len(self._mirrors)})"
        )
