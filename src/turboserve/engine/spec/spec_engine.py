"""The speculative engine: draft ``k`` tokens cheaply, verify them in one target forward.

Ordinary decoding is bound by memory bandwidth, not arithmetic: producing one token reads
every weight of the model and multiplies each by a single row of activations. Speculative
decoding (Leviathan et al., ICML 2023; Chen et al., 2023) spends that same weight read on
``k+1`` candidate positions at once. A cheap drafter guesses the next ``k`` tokens, the
target model scores all ``k+1`` positions in a single forward pass, and a verification rule
decides how many of the guesses to keep -- in a way that leaves the output distribution
exactly as it would have been. Nothing is approximated; what changes is how many forward
passes it takes to emit a token.

**Where it fits.** :class:`SpeculativeLLMEngine` is a subclass of
:class:`~turboserve.engine.runtime.engine.LLMEngine` that replaces two methods. Scheduling,
admission, chunked prefill, preemption, the prefix cache, detokenisation, stop conditions and
statistics are inherited unchanged -- prefill is not speculated, so a prefill-only step runs
the base engine's path verbatim.

**The step.**

1. The scheduler produces a step as usual. Every decoding sequence in it has exactly one
   uncomputed token (the one it sampled last step).
2. The drafter proposes up to ``k`` tokens per decoding sequence. Proposals are ragged: a
   sequence may get fewer, or none, and one with none simply decodes normally.
3. Each drafted sequence is *provisionally extended* by its draft: the tokens are appended,
   KV slots are allocated for them, and the query length for that sequence becomes ``1 + k``.
   The batch is otherwise an ordinary packed variable-length batch, which is why verification
   needs no new attention kernel -- the causal mask over a ``k+1``-token query against a
   cached context is the same shape a chunked prefill produces.
4. One target forward pass scores every position. Row ``j`` of a sequence's block is the
   distribution for its ``j``-th draft token; the last row is the bonus position.
5. :mod:`turboserve.engine.spec.verifier` decides how many draft tokens survive.
6. **Rollback.** The provisional extension is undone completely: the draft tokens are removed
   from the sequence and ``num_computed_tokens`` goes back to where the scheduler left it.
   The accepted tokens are then appended one at a time through the ordinary path, so stop
   conditions, timing stamps, detokenisation and block release all behave exactly as in a
   non-speculative engine -- and the KV that was computed for accepted tokens is credited
   back, so nothing is recomputed. Rejected draft tokens leave their KV behind in slots that
   are simply overwritten by the tokens that take their place.

**Why the rollback is expressed this way.** The obvious alternative -- keep the accepted
tokens in place and only report the bonus -- would bypass the engine's stop checks for every
accepted token, so a request could emit tokens past its ``max_tokens`` or past a stop string.
Undoing everything and replaying through :meth:`LLMEngine._process` costs a few list
operations per step and makes it impossible for a speculative request to observe different
stopping behaviour from a non-speculative one.

**Memory.** When a model drafter is used, the draft model and its KV pool are built *before*
the target engine profiles device memory, so the target's pool is sized from what is actually
left. Doing it the other way round would leave nothing for the draft model on a device the
target had already filled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from turboserve.engine.core.scheduler import ScheduledSeq, SchedulerOutput
from turboserve.engine.runtime.engine import LLMEngine
from turboserve.engine.runtime.worker import StepOutput
from turboserve.engine.spec.drafter import Drafter, DraftProposal, ModelDrafter
from turboserve.engine.spec.ngram import NgramDrafter
from turboserve.engine.spec.verifier import Verification, verify_sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping
    from collections.abc import Sequence as SequenceABC

    from turboserve.config import Settings
    from turboserve.engine.core.sequence import Sequence
    from turboserve.engine.core.types import EngineConfig, RequestOutput

logger = logging.getLogger(__name__)

__all__ = ["SpecStats", "SpeculativeConfig", "SpeculativeLLMEngine"]

DEFAULT_NUM_SPECULATIVE_TOKENS = 4
"""``k``. Small enough that a wholly rejected draft wastes little, large enough to matter."""


class SpeculativeConfig(BaseModel):
    """Everything speculative decoding needs, parsed from ``EngineConfig.speculative``.

    That field is an untyped mapping on purpose: the shared engine configuration must not
    have to grow a field every time a drafter gains an option. This model is where the
    mapping becomes typed, and ``extra="forbid"`` means a misspelt key is an error at startup
    rather than a silently ignored setting.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    method: Literal["model", "ngram"] = "model"
    """``model`` runs a small draft LM; ``ngram`` copies from the sequence's own history."""

    draft_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("draft_model", "model"),
        description="Repo id or path of the draft model; required when method is 'model'.",
    )
    num_speculative_tokens: int = Field(
        default=DEFAULT_NUM_SPECULATIVE_TOKENS,
        ge=1,
        le=32,
        validation_alias=AliasChoices("num_speculative_tokens", "k"),
    )
    draft_num_blocks: int | None = Field(default=None, ge=1)
    """Explicit draft KV pool size. ``None`` profiles device memory instead."""

    draft_block_size: int | None = Field(default=None, ge=1)
    """Draft pool block size; defaults to the target's."""

    draft_gpu_memory_utilization: float = Field(default=0.15, gt=0.0, le=1.0)
    """Share of device memory the draft pool may take, used only when it is profiled."""

    draft_max_model_len: int | None = Field(default=None, ge=1)
    """Stop drafting past this length, for a draft model with a shorter context window."""

    draft_max_num_batched_tokens: int = Field(default=2048, ge=1)
    """Token budget for the catch-up passes that bring a new sequence's draft cache up."""

    ngram_min: int = Field(default=2, ge=1)
    ngram_max: int = Field(default=4, ge=1)
    max_batch_size: int | None = Field(default=None, ge=1)
    """Skip speculation in steps with more decoding sequences than this.

    Speculation trades arithmetic for forward passes, and that trade stops paying once the
    batch is large enough to keep the device busy on its own. Leaving this unset speculates
    at every batch size; setting it is how a deployment draws the line for its own hardware.
    """

    @model_validator(mode="after")
    def _check(self) -> SpeculativeConfig:
        if self.method == "model" and not self.draft_model:
            raise ValueError("speculative method 'model' needs a draft model ('draft_model')")
        if self.ngram_max < self.ngram_min:
            raise ValueError(f"ngram_max {self.ngram_max} is below ngram_min {self.ngram_min}")
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> SpeculativeConfig:
        """Build from ``EngineConfig.speculative``; an absent mapping is an error.

        A speculative engine with no speculative configuration is a configuration mistake,
        not a default: the caller asked for the subclass.
        """
        if not data:
            raise ValueError(
                "speculative decoding is enabled but EngineConfig.speculative is empty; "
                "set at least {'method': 'ngram'} or {'draft_model': '<repo id>'}"
            )
        return cls.model_validate(dict(data))


@dataclass(frozen=True, slots=True)
class SpecStats:
    """How speculation is going, in the form a benchmark result records.

    ``acceptance_rate`` is the share of drafted tokens that survived verification and
    ``mean_accepted_len`` the average number of them kept per verified sequence-step. Both
    are lifetime figures over the engine's whole run; a benchmark that wants them per phase
    takes two snapshots and subtracts.
    """

    num_spec_steps: int
    num_verified_seqs: int
    num_drafted: int
    num_accepted: int
    num_emitted: int
    num_target_forwards: int
    num_draft_calls: int
    acceptance_rate: float
    mean_accepted_len: float
    mean_emitted_len: float

    def to_dict(self) -> dict[str, int | float]:
        """Flat, JSON-safe view, prefixed by the engine's ``stats()``."""
        return {
            "num_spec_steps": self.num_spec_steps,
            "num_verified_seqs": self.num_verified_seqs,
            "num_drafted": self.num_drafted,
            "num_accepted": self.num_accepted,
            "num_emitted": self.num_emitted,
            "num_target_forwards": self.num_target_forwards,
            "num_draft_calls": self.num_draft_calls,
            "acceptance_rate": self.acceptance_rate,
            "mean_accepted_len": self.mean_accepted_len,
            "mean_emitted_len": self.mean_emitted_len,
        }


@dataclass(slots=True)
class _Verified:
    """One decoding sequence's provisional extension, kept until the tokens are fed back."""

    item: ScheduledSeq
    """The scheduler's original entry; what the engine feeds tokens back against."""

    verify_item: ScheduledSeq
    """The extended entry that went into the verification batch."""

    num_drafted: int
    base_computed: int
    """``num_computed_tokens`` as the scheduler left it, the point rollback returns to."""

    base_outputs: int
    """``len(seq.output_token_ids)`` before the draft was appended."""


class SpeculativeLLMEngine(LLMEngine):
    """An :class:`~turboserve.engine.runtime.engine.LLMEngine` whose decode phase speculates.

    Construct it exactly like the base engine; the extra configuration comes from
    ``EngineConfig.speculative``::

        config = EngineConfig(model="...", speculative={"method": "ngram"})
        with SpeculativeLLMEngine(config) as engine:
            engine.add_request("r1", "hello")
            while engine.has_unfinished():
                for output in engine.step():
                    ...

    Every observable behaviour of the base class is preserved: greedy requests produce the
    target model's greedy continuation token for token, sampled requests produce draws from
    the target's own distribution, and stop conditions, timing and streaming are unchanged.
    """

    def __init__(
        self,
        config: EngineConfig,
        *,
        drafter: Drafter | None = None,
        spec_config: SpeculativeConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the drafter first, then the engine, then check they agree on the vocabulary.

        Args:
            config: the ordinary engine configuration. ``config.speculative`` supplies the
                speculative settings unless ``spec_config`` is passed.
            drafter: an already-built drafter, which skips loading one. Tests use this to
                share a single tiny model between target and draft, and a deployment can use
                it to hand the engine a drafter with a pool it sized itself.
            spec_config: parsed speculative settings, overriding ``config.speculative``.
            **kwargs: forwarded to :class:`~turboserve.engine.runtime.engine.LLMEngine`.
        """
        self.spec_config = (
            spec_config
            if spec_config is not None
            else SpeculativeConfig.from_mapping(config.speculative)
        )
        # Built before super().__init__ so that the target engine's memory profiling sees the
        # draft model's weights and pool as already-used memory.
        self.drafter: Drafter = drafter if drafter is not None else self._build_drafter(config)
        super().__init__(config, **kwargs)
        self._check_vocabulary()
        self._caps: dict[int, int] = {}
        self._num_spec_steps = 0
        self._num_verified_seqs = 0
        self._num_drafted = 0
        self._num_accepted = 0
        self._num_emitted = 0
        self._num_target_forwards = 0
        self._num_draft_calls = 0
        logger.info(
            "speculative decoding enabled: %s drafter, k=%d",
            self.drafter.name,
            self.spec_config.num_speculative_tokens,
        )

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> SpeculativeLLMEngine:
        """Build from the process-wide ``TURBOSERVE_*`` settings."""
        from turboserve.engine.core.types import EngineConfig

        engine = cls(EngineConfig.from_settings(settings), **kwargs)
        return engine

    def _build_drafter(self, config: EngineConfig) -> Drafter:
        """Instantiate the drafter the configuration asks for."""
        spec = self.spec_config
        if spec.method == "ngram":
            return NgramDrafter(min_ngram=spec.ngram_min, max_ngram=spec.ngram_max)
        assert spec.draft_model is not None  # guaranteed by SpeculativeConfig validation
        return ModelDrafter.from_pretrained(
            spec.draft_model,
            dtype=config.resolved_dtype(),
            device=config.resolved_device(),
            num_blocks=spec.draft_num_blocks,
            block_size=spec.draft_block_size or config.block_size,
            gpu_memory_utilization=spec.draft_gpu_memory_utilization,
            max_num_batched_tokens=spec.draft_max_num_batched_tokens,
            max_model_len=spec.draft_max_model_len,
        )

    def _check_vocabulary(self) -> None:
        """Refuse a draft model whose vocabulary differs from the target's.

        A draft token id only means anything if both models index the same vocabulary. A
        mismatch is not detectable from the output -- it produces fluent nonsense at a
        plausible acceptance rate -- so it is refused here.
        """
        drafter = self.drafter
        if not isinstance(drafter, ModelDrafter):
            return
        if drafter.vocab_size != self.model.config.vocab_size:
            raise ValueError(
                f"draft model vocabulary is {drafter.vocab_size} tokens but the target's is "
                f"{self.model.config.vocab_size}; the two models must share a tokenizer"
            )
        if drafter.device != self.device:
            raise ValueError(
                f"draft model is on {drafter.device} but the target is on {self.device}"
            )

    # -- the step -------------------------------------------------------------------------

    def _execute(self, scheduled: SchedulerOutput) -> StepOutput:
        """Run a speculative step when the step has anything to speculate on.

        Falls back to the base engine's path for a prefill-only step, for a batch above
        ``max_batch_size``, and for a drafter that proposed nothing -- in each case the result
        is an ordinary decode, which is what makes speculation safe to leave switched on.
        """
        self._caps.clear()
        if self.decode_step_hook is not None:
            replaced = self.decode_step_hook(self, scheduled)
            if replaced is not None:
                return replaced
        decodes = [item for item in scheduled.scheduled if not item.is_prefill]
        limit = self.spec_config.max_batch_size
        if not decodes or (limit is not None and len(decodes) > limit):
            return self.runner.execute(scheduled)
        return self._speculative_step(scheduled, decodes)

    @torch.inference_mode()
    def _speculative_step(
        self, scheduled: SchedulerOutput, decodes: SequenceABC[ScheduledSeq]
    ) -> StepOutput:
        """Draft, verify and roll back one step. See the module docstring for the shape."""
        k = self.spec_config.num_speculative_tokens
        proposal = self.drafter.propose([item.seq for item in decodes], k)
        self._num_draft_calls += 1
        if proposal.is_empty:
            return self.runner.execute(scheduled)

        prefills = [item for item in scheduled.scheduled if item.is_prefill]
        extended: list[_Verified] = []
        reserved = 0
        for position, item in enumerate(decodes):
            drafts, reserved = self._fund(item.seq, proposal.tokens_for(position), reserved)
            extended.append(self._extend(item, drafts))

        verify_out = self._verify_batch(prefills, extended)
        batch = self.runner.build_batch(verify_out)
        hidden = self.runner.forward(batch)
        self._num_target_forwards += 1
        rows, prefill_rows, decode_spans = self._logit_rows(prefills, extended)
        logits = self.model.compute_logits(hidden, rows)

        items: list[ScheduledSeq] = []
        tokens: list[int] = []
        self._sample_prefills(logits, prefill_rows, prefills, items, tokens)
        self._verify_decodes(logits, decode_spans, extended, proposal, items, tokens)
        self._num_spec_steps += 1
        return StepOutput(
            items=items,
            token_ids=tokens,
            num_batched_tokens=verify_out.num_batched_tokens,
            num_prefill_tokens=verify_out.num_prefill_tokens,
            num_decode_tokens=verify_out.num_decode_tokens,
        )

    def _fund(
        self, seq: Sequence, drafts: SequenceABC[int], reserved: int
    ) -> tuple[list[int], int]:
        """Trim a draft to what the target's KV pool can hold, reserving its blocks.

        Speculation must never cause a preemption: a draft that does not fit is shortened,
        possibly to nothing, and the sequence decodes one token as it would have anyway.
        ``reserved`` accumulates across the step so two sequences cannot both be promised the
        last free block.
        """
        if not drafts:
            return [], reserved
        manager = self.scheduler.block_manager
        allocatable = manager.allocator.num_allocatable - reserved
        block_size = manager.block_size
        allowed = len(drafts)
        while allowed > 0:
            end = seq.num_computed_tokens + allowed
            needed = max(0, -(-end // block_size) - len(seq.block_table))
            if needed <= allocatable:
                return [int(token) for token in drafts[:allowed]], reserved + needed
            allowed -= 1
        return [], reserved

    def _extend(self, item: ScheduledSeq, drafts: SequenceABC[int]) -> _Verified:
        """Provisionally append a draft to its sequence and widen its slice of the batch."""
        seq = item.seq
        record = _Verified(
            item=item,
            verify_item=item,
            num_drafted=len(drafts),
            base_computed=seq.num_computed_tokens,
            base_outputs=len(seq.output_token_ids),
        )
        if not drafts:
            return record
        seq.output_token_ids.extend(int(token) for token in drafts)
        slots = self.scheduler.block_manager.append_slots(seq, len(drafts))
        seq.advance_computed(len(drafts))
        record.verify_item = ScheduledSeq(
            seq=seq,
            num_new_tokens=item.num_new_tokens + len(drafts),
            context_len=item.context_len + len(drafts),
            token_ids=[*item.token_ids, *(int(token) for token in drafts)],
            slot_mapping=[*item.slot_mapping, *slots],
            block_table=list(seq.block_table),
            is_prefill=False,
            is_chunk=False,
            num_cached_tokens=item.num_cached_tokens,
        )
        return record

    @staticmethod
    def _verify_batch(
        prefills: SequenceABC[ScheduledSeq], extended: SequenceABC[_Verified]
    ) -> SchedulerOutput:
        """Assemble the packed batch: prefills unchanged, decodes widened to ``1 + k``."""
        decode_items = [record.verify_item for record in extended]
        out = SchedulerOutput(scheduled=[*prefills, *decode_items])
        out.num_prefill_seqs = len(prefills)
        out.num_decode_seqs = len(decode_items)
        out.num_prefill_tokens = sum(item.num_new_tokens for item in prefills)
        out.num_decode_tokens = sum(item.num_new_tokens for item in decode_items)
        return out

    def _logit_rows(
        self,
        prefills: SequenceABC[ScheduledSeq],
        extended: SequenceABC[_Verified],
    ) -> tuple[torch.Tensor, list[int], list[tuple[int, int]]]:
        """Which rows of the packed hidden states to project, and who owns each.

        A prefill contributes its last row, as always. A verified sequence contributes *every*
        row of its block, because each one carries the target's opinion about a different
        draft position. Returns the gather index, the positions of the prefill rows within it,
        and one ``(start, length)`` span per verified sequence.
        """
        gather: list[int] = []
        prefill_rows: list[int] = []
        spans: list[tuple[int, int]] = []
        offset = 0
        for item in prefills:
            offset += item.num_new_tokens
            if item.samples_token:
                prefill_rows.append(len(gather))
                gather.append(offset - 1)
        for record in extended:
            width = record.verify_item.num_new_tokens
            spans.append((len(gather), width))
            gather.extend(range(offset, offset + width))
            offset += width
        return torch.tensor(gather, dtype=torch.long, device=self.device), prefill_rows, spans

    def _sample_prefills(
        self,
        logits: torch.Tensor,
        prefill_rows: SequenceABC[int],
        prefills: SequenceABC[ScheduledSeq],
        items: list[ScheduledSeq],
        tokens: list[int],
    ) -> None:
        """Sample the prefills that finished their prompts, exactly as the base engine does."""
        sampling_items = [item for item in prefills if item.samples_token]
        if not sampling_items:
            return
        index = torch.tensor(list(prefill_rows), dtype=torch.long, device=logits.device)
        sampled = self.runner.sample(logits.index_select(0, index), sampling_items)
        items.extend(sampling_items)
        tokens.extend(sampled.token_ids)

    def _verify_decodes(
        self,
        logits: torch.Tensor,
        spans: SequenceABC[tuple[int, int]],
        extended: SequenceABC[_Verified],
        proposal: DraftProposal,
        items: list[ScheduledSeq],
        tokens: list[int],
    ) -> None:
        """Verify every drafted sequence, roll it back, and queue its tokens for feedback.

        A sequence that was given no draft still passes through here: its block is a single
        row, verification of an empty draft is a draw from that row, and the result is
        bit-for-bit an ordinary decode step. Keeping the one path means a step that mixes
        speculated and unspeculated sequences needs no special case anywhere.
        """
        for position, (record, span) in enumerate(zip(extended, spans, strict=True)):
            start, width = span
            seq = record.item.seq
            drafts = record.verify_item.token_ids[record.item.num_new_tokens :]
            history = None
            if seq.sampling.repetition_penalty != 1.0:
                full = seq.token_ids
                history = [full[: record.base_computed + offset] for offset in range(width)]
            verification = verify_sequence(
                logits[start : start + width],
                drafts,
                seq.sampling,
                draft_probs=self._draft_probs(proposal, position, record.num_drafted),
                generator=seq.ensure_generator(logits.device),
                token_history=history,
            )
            self._rollback(record)
            self._caps[seq.seq_id] = record.base_computed + verification.num_accepted
            self._record(verification)
            for token in verification.tokens:
                items.append(record.item)
                tokens.append(token)

    @staticmethod
    def _draft_probs(
        proposal: DraftProposal, position: int, num_drafted: int
    ) -> torch.Tensor | None:
        """The draft distributions actually used, trimmed to the draft the pool could fund."""
        if num_drafted <= 0:
            return None
        probs = proposal.probs_for(position)
        return None if probs is None else probs[:num_drafted]

    @staticmethod
    def _rollback(record: _Verified) -> None:
        """Undo the provisional extension, leaving the sequence as the scheduler left it.

        The KV written for the draft stays in the cache and is simply overwritten: the slots
        belong to positions the sequence still owns, and whatever tokens end up at those
        positions will be written there by a later step.
        """
        seq = record.item.seq
        if record.num_drafted:
            del seq.output_token_ids[record.base_outputs :]
        seq.num_computed_tokens = record.base_computed

    def _record(self, verification: Verification) -> None:
        """Accumulate acceptance statistics for genuinely speculated sequences only.

        A sequence that was offered no draft contributes nothing: counting it would drag the
        mean accepted length towards zero for a step in which nothing was speculated.
        """
        if verification.num_drafted <= 0:
            return
        self._num_verified_seqs += 1
        self._num_drafted += verification.num_drafted
        self._num_accepted += verification.num_accepted
        self._num_emitted += verification.num_emitted

    # -- feedback ---------------------------------------------------------------------------

    def _process(self, step_out: StepOutput, *, now: float) -> list[RequestOutput]:
        """Feed the step's tokens back one at a time, restoring KV credit as it goes.

        The base implementation is reused per token rather than reimplemented, because every
        rule it applies -- the model's EOS set, stop token ids, ``max_tokens``, stop strings,
        block release, timing -- must apply identically to a token that came from a draft. The
        only addition is the ``num_computed_tokens`` bookkeeping: a sequence that had ``m``
        draft tokens accepted already has their KV in the cache, so as those tokens are
        appended the computed count follows them up to the cap and then stops, leaving the
        bonus token uncomputed exactly as an ordinary decode would.
        """
        outputs: list[RequestOutput] = []
        finished: set[int] = set()
        for item, token_id in step_out.pairs():
            single = StepOutput(items=[item], token_ids=[token_id])
            outputs.extend(super()._process(single, now=now))
            seq = item.seq
            cap = self._caps.get(seq.seq_id)
            if cap is not None:
                seq.num_computed_tokens = min(cap, seq.num_tokens)
            if seq.is_finished:
                finished.add(seq.seq_id)
        for seq_id in finished:
            self.drafter.release(seq_id)
        return outputs

    def abort(self, request_id: str, *, now: float | None = None) -> bool:
        """Cancel a request and release its draft-side state."""
        seq = self.scheduler.get_sequence(request_id)
        aborted = super().abort(request_id, now=now)
        if seq is not None:
            self.drafter.release(seq.seq_id)
        return aborted

    def close(self) -> None:
        """Release the drafter's pool along with the engine's. Idempotent."""
        self.drafter.reset()
        super().close()

    # -- observability -----------------------------------------------------------------------

    def spec_stats(self) -> SpecStats:
        """Acceptance accounting for this engine's lifetime."""
        drafted = self._num_drafted
        verified = self._num_verified_seqs
        return SpecStats(
            num_spec_steps=self._num_spec_steps,
            num_verified_seqs=verified,
            num_drafted=drafted,
            num_accepted=self._num_accepted,
            num_emitted=self._num_emitted,
            num_target_forwards=self._num_target_forwards,
            num_draft_calls=self._num_draft_calls,
            acceptance_rate=self._num_accepted / drafted if drafted else 0.0,
            mean_accepted_len=self._num_accepted / verified if verified else 0.0,
            mean_emitted_len=self._num_emitted / verified if verified else 0.0,
        )

    def stats(self) -> dict[str, int | float | str]:
        """The base engine's statistics plus speculation and drafter counters."""
        out = super().stats()
        out.update(self.spec_stats().to_dict())
        out.update(self.drafter.stats())
        out["drafter"] = self.drafter.name
        out["num_speculative_tokens"] = self.spec_config.num_speculative_tokens
        return out

    def __repr__(self) -> str:
        return (
            f"SpeculativeLLMEngine(model={self.config.model!r}, drafter={self.drafter.name}, "
            f"k={self.spec_config.num_speculative_tokens}, "
            f"unfinished={self.scheduler.num_unfinished})"
        )
