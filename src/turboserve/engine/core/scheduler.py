"""Continuous batching: which sequences run in the next forward pass, and with how many tokens.

The scheduler is the engine's only decision maker. Once per step it produces a
:class:`SchedulerOutput` describing one packed batch -- a flat token vector made of prefill
chunks and single decode tokens mixed together -- and the runtime turns that into tensors
with :meth:`SchedulerOutput.build_attn_metadata` and runs the model over it. Nothing else in
the engine chooses what to run.

Three mechanisms make that batch dense rather than ragged:

* **Continuous batching.** A sequence joins and leaves the batch at step boundaries instead
  of at request boundaries, so a finished request frees its slot immediately rather than
  when the whole batch finishes. There is no "batch" object at all: there is a set of
  running sequences and a per-step budget.
* **Chunked prefill.** A long prompt is split across steps to fit the token budget, so it
  cannot stall every decoding sequence for the length of its prefill. The price is paid in
  the block manager, which must hand out KV slots for a partial prompt.
* **Recompute preemption.** When the KV pool has no room for the next token of a running
  sequence, some running sequence loses its blocks and goes back to the queue rather than
  the request failing. The victim is the lowest-priority, most recently admitted sequence,
  because that is the one whose recomputation costs the least and whose SLO is least at
  risk. With the prefix cache behind it, resuming a preempted sequence usually re-adopts
  the blocks it just gave up.

**Determinism.** :meth:`Scheduler.schedule` is a pure function of the queue state and the
config. Wall-clock time enters only as timestamps recorded on sequences, and can be injected
through the ``now`` argument, so a test can replay a scheduling scenario exactly.

**Step protocol.** The runtime is expected to do, per step::

    out = scheduler.schedule()                      # 1. pick the batch, assign KV slots
    meta = out.build_attn_metadata(block_size)      # 2. build tensors
    ...                                             # 3. forward + sample
    for item in out.sampled():                      # 4. feed tokens back
        scheduler.append_token(item.seq, token_id, eos_token_id=eos)

Step 1 advances each scheduled sequence's ``num_computed_tokens`` immediately: the forward
pass is going to run over exactly those tokens, and deferring the update would mean two
sources of truth. Blocks completed by a step are published to the prefix cache at the start
of the *next* step, for the reason documented in :mod:`turboserve.engine.core.block_manager`.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING

import torch

from turboserve.engine.core.block_manager import BlockManager
from turboserve.engine.core.sequence import DEFAULT_TENANT, SeqStatus, Sequence
from turboserve.engine.core.types import (
    NO_LORA,
    AttnMetadata,
    FinishReason,
    LoRAContext,
    SamplingParams,
    SchedulerConfig,
    build_query_start_loc,
    pad_block_tables,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

__all__ = ["ScheduledSeq", "Scheduler", "SchedulerOutput"]


@dataclass(slots=True)
class ScheduledSeq:
    """One sequence's share of one step.

    Every field is a snapshot taken while the batch was assembled, not a live view of the
    sequence, so that the runtime can build tensors, run the model and feed tokens back in
    any order without the description of the batch shifting underneath it.
    """

    seq: Sequence
    num_new_tokens: int
    """Tokens this step computes for the sequence: a prefill chunk, or 1 when decoding."""

    context_len: int
    """Total attended length *after* this step; the causal offset is ``context_len - query_len``."""

    token_ids: list[int]
    """The ``num_new_tokens`` token ids fed to the model, in order."""

    slot_mapping: list[int]
    """Flat KV slot for each of those tokens, in the same order."""

    block_table: list[int]
    """The sequence's block ids at schedule time."""

    is_prefill: bool
    """Whether this step computes at least one prompt token."""

    is_chunk: bool
    """Whether this is a prefill chunk that does *not* finish the prompt (no token sampled)."""

    num_cached_tokens: int = 0
    """Prompt tokens this sequence skipped thanks to the prefix cache, at admission."""

    def __post_init__(self) -> None:
        if self.num_new_tokens < 1:
            raise ValueError("a scheduled sequence must compute at least one token")
        if len(self.token_ids) != self.num_new_tokens:
            raise ValueError(
                f"{len(self.token_ids)} token ids for {self.num_new_tokens} scheduled tokens"
            )
        if len(self.slot_mapping) != self.num_new_tokens:
            raise ValueError(
                f"{len(self.slot_mapping)} slots for {self.num_new_tokens} scheduled tokens"
            )
        if self.context_len < self.num_new_tokens:
            raise ValueError(
                f"context_len {self.context_len} is shorter than the {self.num_new_tokens} "
                "queries this step contributes"
            )

    @property
    def seq_id(self) -> int:
        """Engine-local sequence id."""
        return self.seq.seq_id

    @property
    def request_id(self) -> str:
        """Client-facing request id."""
        return self.seq.request_id

    @property
    def tenant_id(self) -> str:
        """Tenant the request belongs to."""
        return self.seq.tenant_id

    @property
    def lora_id(self) -> int:
        """Adapter slot for every token of this sequence in this step."""
        return self.seq.lora_id

    @property
    def query_len(self) -> int:
        """Alias of :attr:`num_new_tokens`, in the vocabulary attention kernels use."""
        return self.num_new_tokens

    @property
    def samples_token(self) -> bool:
        """Whether a token is sampled from this sequence's last position after the step.

        ``False`` only for a prefill chunk that stops short of the end of the prompt: its
        last hidden state is in the middle of the prompt and predicts a token the prompt
        already contains.
        """
        return not self.is_chunk

    @property
    def positions(self) -> list[int]:
        """Absolute position ids of this step's tokens, for the rotary embedding."""
        return list(range(self.context_len - self.num_new_tokens, self.context_len))


@dataclass(slots=True)
class SchedulerOutput:
    """The complete description of one step.

    ``scheduled`` is ordered prefills first, then decodes, which is the layout
    :class:`~turboserve.engine.core.types.AttnMetadata` documents and the attention
    implementations rely on to split the batch without a sort.
    """

    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    """Sequences that lost their blocks during this step and went back to the queue."""

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefill_seqs: int = 0
    num_decode_seqs: int = 0
    num_cached_tokens: int = 0
    """Prompt tokens skipped by prefix-cache hits on sequences admitted in this step."""

    @property
    def num_batched_tokens(self) -> int:
        """Total tokens in the packed batch."""
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def num_seqs(self) -> int:
        """Sequences in the batch."""
        return len(self.scheduled)

    @property
    def is_empty(self) -> bool:
        """Whether this step has nothing to run."""
        return not self.scheduled

    def __iter__(self) -> Iterator[ScheduledSeq]:
        return iter(self.scheduled)

    def __len__(self) -> int:
        return len(self.scheduled)

    def sequences(self) -> list[Sequence]:
        """The scheduled sequences, in batch order."""
        return [item.seq for item in self.scheduled]

    def sampled(self) -> list[ScheduledSeq]:
        """The entries that produce a token this step (everything but unfinished chunks)."""
        return [item for item in self.scheduled if item.samples_token]

    def sample_indices(self) -> list[int]:
        """Row indices, in the flat token vector, of the positions to take logits from.

        The model returns one hidden state per batched token; only the last token of each
        sampling sequence predicts anything, so this is the gather index the runtime uses
        before calling the sampler. Aligned with :meth:`sampled`.
        """
        indices: list[int] = []
        offset = 0
        for item in self.scheduled:
            offset += item.num_new_tokens
            if item.samples_token:
                indices.append(offset - 1)
        return indices

    def input_token_ids(self) -> list[int]:
        """The packed token vector: every scheduled sequence's new tokens, concatenated."""
        return [token for item in self.scheduled for token in item.token_ids]

    def positions(self) -> list[int]:
        """Absolute position id of every token in the packed batch."""
        return [position for item in self.scheduled for position in item.positions]

    def token_lora_ids(self) -> list[int]:
        """Adapter slot of every token in the packed batch."""
        return [item.lora_id for item in self.scheduled for _ in range(item.num_new_tokens)]

    def build_lora_context(self, *, device: torch.device | str = "cpu") -> LoRAContext:
        """Per-token adapter slots for the grouped LoRA linears."""
        return LoRAContext.from_slots(self.token_lora_ids(), device=device)

    def build_attn_metadata(
        self,
        block_size: int,
        *,
        device: torch.device | str = "cpu",
        max_blocks: int | None = None,
    ) -> AttnMetadata:
        """Turn this step into the tensors the attention layers consume.

        ``block_size`` is used to check the one invariant that cannot be recovered later:
        that each sequence's block table really covers the context length claimed for it.
        A short block table would make an attention kernel read :data:`PAD_BLOCK` or
        another tenant's block, so it is checked here, cheaply, once per sequence per step.
        """
        if not self.scheduled:
            raise ValueError("cannot build attention metadata for an empty step")
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        rows: list[list[int]] = []
        for item in self.scheduled:
            if len(item.block_table) * block_size < item.context_len:
                raise ValueError(
                    f"sequence {item.request_id!r} claims {item.context_len} context tokens "
                    f"but holds {len(item.block_table)} blocks of {block_size}"
                )
            slot_mapping.extend(item.slot_mapping)
            context_lens.append(item.context_len)
            query_lens.append(item.num_new_tokens)
            rows.append(item.block_table)
        return AttnMetadata(
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=device),
            block_tables=pad_block_tables(rows, device=device, max_blocks=max_blocks),
            context_lens=torch.tensor(context_lens, dtype=torch.long, device=device),
            query_start_loc=build_query_start_loc(query_lens, device=device),
            max_query_len=max(query_lens),
            max_context_len=max(context_lens),
            num_prefill_seqs=self.num_prefill_seqs,
            num_decode_seqs=self.num_decode_seqs,
        )


@dataclass(slots=True)
class _Budget:
    """What is left of the step's two limits."""

    tokens: int
    seqs: int

    @property
    def exhausted(self) -> bool:
        """Whether nothing more can be added to this step."""
        return self.tokens <= 0 or self.seqs <= 0


class _WaitQueue:
    """The waiting queue, under whichever admission policy the config selects.

    ``fcfs`` is a plain FIFO. ``tenant_fair`` is weighted fair queueing: each tenant carries
    a virtual time that advances by ``1 / weight`` every time one of its requests is
    admitted, and the tenant with the smallest virtual time goes next. That gives each
    tenant a share of admissions proportional to its weight and, more importantly, bounds
    how long any tenant can be made to wait: a tenant's virtual time only stands still while
    it is not being served, so it necessarily becomes the minimum after a bounded number of
    admissions by others.

    The virtual clock is self-clocked (SCFQ, Golestani 1994): ``_now`` is the virtual time
    of the request admitted most recently, and a tenant that becomes backlogged again
    restarts at ``max(its own virtual time, _now)``. Without that floor an idle tenant would
    keep the small virtual time it stopped at, bank credit for the whole idle period and
    then monopolise admissions until it caught up. The floor is what makes the fair share a
    share of the *current* epoch rather than of all time.
    """

    __slots__ = ("_fifo", "_now", "_queues", "_tenant_fair", "_vtime", "_weights")

    def __init__(self, *, tenant_fair: bool, weights: dict[str, float] | None = None) -> None:
        self._tenant_fair = tenant_fair
        self._weights = dict(weights or {})
        self._fifo: deque[Sequence] = deque()
        self._queues: dict[str, deque[Sequence]] = {}
        self._vtime: dict[str, float] = {}
        self._now = 0.0

    def weight_of(self, tenant_id: str) -> float:
        """Configured weight of a tenant; unconfigured tenants get ``1.0``."""
        return self._weights.get(tenant_id, 1.0)

    def push(self, seq: Sequence) -> None:
        """Enqueue a sequence at the back of its queue."""
        if not self._tenant_fair:
            self._fifo.append(seq)
            return
        tenant = seq.tenant_id
        queue = self._queues.get(tenant)
        if queue is None:
            # Becoming backlogged again: never earlier than the virtual clock, so the time
            # spent idle earns nothing.
            queue = deque()
            self._queues[tenant] = queue
            self._vtime[tenant] = max(self._vtime.get(tenant, 0.0), self._now)
        queue.append(seq)

    def peek(self) -> Sequence | None:
        """The sequence that would be admitted next, without admitting it."""
        if not self._tenant_fair:
            return self._fifo[0] if self._fifo else None
        tenant = self._next_tenant()
        return self._queues[tenant][0] if tenant is not None else None

    def pop(self) -> Sequence:
        """Admit the next sequence, charging its tenant's virtual time."""
        if not self._tenant_fair:
            return self._fifo.popleft()
        tenant = self._next_tenant()
        if tenant is None:
            raise IndexError("pop from an empty wait queue")
        queue = self._queues[tenant]
        seq = queue.popleft()
        self._vtime[tenant] += 1.0 / self.weight_of(tenant)
        self._now = self._vtime[tenant]
        if not queue:
            del self._queues[tenant]
        return seq

    def remove(self, seq: Sequence) -> bool:
        """Drop a specific sequence (an abort while it was queued)."""
        if not self._tenant_fair:
            try:
                self._fifo.remove(seq)
            except ValueError:
                return False
            return True
        queue = self._queues.get(seq.tenant_id)
        if queue is None:
            return False
        try:
            queue.remove(seq)
        except ValueError:
            return False
        if not queue:
            del self._queues[seq.tenant_id]
        return True

    def clear(self) -> None:
        """Empty every queue and forget the virtual times."""
        self._fifo.clear()
        self._queues.clear()
        self._vtime.clear()
        self._now = 0.0

    def _next_tenant(self) -> str | None:
        if not self._queues:
            return None
        return min(self._queues, key=lambda tenant: (self._vtime.get(tenant, 0.0), tenant))

    def __len__(self) -> int:
        if not self._tenant_fair:
            return len(self._fifo)
        return sum(len(queue) for queue in self._queues.values())

    def __iter__(self) -> Iterator[Sequence]:
        if not self._tenant_fair:
            return iter(list(self._fifo))
        return iter([seq for queue in self._queues.values() for seq in queue])


class Scheduler:
    """Continuous-batching scheduler with chunked prefill and recompute preemption."""

    def __init__(self, config: SchedulerConfig, block_manager: BlockManager | None = None) -> None:
        if block_manager is None:
            if config.num_blocks is None:
                raise ValueError(
                    "SchedulerConfig.num_blocks must be set when no BlockManager is supplied; "
                    "the engine normally sizes it by profiling device memory"
                )
            block_manager = BlockManager.create(
                config.num_blocks,
                config.block_size,
                enable_prefix_caching=config.enable_prefix_caching,
            )
        if block_manager.block_size != config.block_size:
            raise ValueError(
                f"block manager block_size {block_manager.block_size} disagrees with the "
                f"scheduler's {config.block_size}"
            )
        self.config = config
        self.block_manager = block_manager
        self._waiting = _WaitQueue(
            tenant_fair=config.policy == "tenant_fair", weights=config.tenant_weights
        )
        self._preempted: deque[Sequence] = deque()
        self._running: list[Sequence] = []
        self._by_request: dict[str, Sequence] = {}
        self._seq_counter = count()
        self._num_steps = 0
        self._num_preemptions = 0
        self._num_finished = 0
        self._num_admitted = 0
        self._num_prefill_tokens = 0
        self._num_decode_tokens = 0
        self._num_cached_tokens = 0

    # -- queue contents -------------------------------------------------------------------

    @property
    def num_waiting(self) -> int:
        """Sequences that have never been admitted, or were admitted and preempted."""
        return len(self._waiting) + len(self._preempted)

    @property
    def num_running(self) -> int:
        """Sequences holding KV blocks."""
        return len(self._running)

    @property
    def num_preempted(self) -> int:
        """Sequences waiting to be resumed after losing their blocks."""
        return len(self._preempted)

    @property
    def num_unfinished(self) -> int:
        """Everything the engine still owes a client an answer for."""
        return len(self._by_request)

    def has_unfinished(self) -> bool:
        """Whether :meth:`schedule` can still make progress."""
        return bool(self._by_request)

    def get_sequence(self, request_id: str) -> Sequence | None:
        """Look up an unfinished request by its client-facing id."""
        return self._by_request.get(request_id)

    def running(self) -> list[Sequence]:
        """Snapshot of the running set, in admission order."""
        return list(self._running)

    def waiting(self) -> list[Sequence]:
        """Snapshot of everything queued, preempted sequences first."""
        return [*self._preempted, *self._waiting]

    def __len__(self) -> int:
        return len(self._by_request)

    # -- request lifecycle ----------------------------------------------------------------

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: Iterable[int],
        sampling: SamplingParams | None = None,
        *,
        tenant_id: str = DEFAULT_TENANT,
        priority: int = 0,
        lora_id: int = NO_LORA,
        arrival: float | None = None,
    ) -> Sequence:
        """Queue a new request and return the sequence the scheduler will drive.

        Two rejections happen here rather than being discovered as a permanent stall later:
        a prompt that cannot fit in the KV pool at all, and -- when chunked prefill is off --
        a prompt longer than one step's token budget. Both are configuration errors from the
        caller's point of view, and both would otherwise sit in the queue forever.
        """
        if request_id in self._by_request:
            raise ValueError(f"request {request_id!r} is already in flight")
        tokens = list(prompt_token_ids)
        seq = Sequence(
            seq_id=next(self._seq_counter),
            request_id=request_id,
            prompt_token_ids=tokens,
            sampling=sampling if sampling is not None else SamplingParams(),
            tenant_id=tenant_id,
            priority=priority,
            lora_id=lora_id,
        )
        seq.timing.t_arrival = time.perf_counter() if arrival is None else arrival
        self.add_sequence(seq)
        return seq

    def add_sequence(self, seq: Sequence) -> None:
        """Queue an already-constructed sequence (used when the engine builds its own)."""
        if seq.request_id in self._by_request:
            raise ValueError(f"request {seq.request_id!r} is already in flight")
        capacity = self.block_manager.allocator.num_total
        needed = seq.num_blocks_needed(self.config.block_size)
        if needed > capacity:
            raise ValueError(
                f"request {seq.request_id!r} needs {needed} KV blocks but the pool holds "
                f"{capacity}; shorten the prompt or enlarge the cache"
            )
        if (
            not self.config.enable_chunked_prefill
            and seq.num_tokens > self.config.max_num_batched_tokens
        ):
            raise ValueError(
                f"request {seq.request_id!r} has {seq.num_tokens} tokens but the step budget "
                f"is {self.config.max_num_batched_tokens} and chunked prefill is disabled"
            )
        seq.status = SeqStatus.WAITING
        self._by_request[seq.request_id] = seq
        self._waiting.push(seq)

    def finish(
        self,
        seq: Sequence,
        reason: FinishReason = FinishReason.STOP,
        *,
        now: float | None = None,
        stop_token_id: int | None = None,
    ) -> None:
        """Retire a sequence: mark it finished, free its blocks, drop it from every queue."""
        seq.finish(reason, now=now, stop_token_id=stop_token_id)
        self.block_manager.free(seq)
        self._detach(seq)
        self._num_finished += 1

    def abort(self, request_id: str, *, now: float | None = None) -> bool:
        """Cancel a request wherever it is. Returns whether it was still in flight."""
        seq = self._by_request.get(request_id)
        if seq is None:
            return False
        self.finish(seq, FinishReason.ABORT, now=now)
        return True

    def append_token(
        self,
        seq: Sequence,
        token_id: int,
        *,
        eos_token_id: int | None = None,
        now: float | None = None,
    ) -> FinishReason | None:
        """Record a sampled token and retire the sequence if it just stopped.

        Returns the finish reason, or ``None`` if the sequence keeps going. This is the one
        call the runtime needs after sampling: it keeps the stop conditions, the timing
        stamps and the block release in one place instead of three.
        """
        seq.append_token(token_id, now=now)
        reason = seq.check_stop(token_id, eos_token_id=eos_token_id)
        if reason is not None:
            stop_token = token_id if reason is FinishReason.STOP else None
            self.finish(seq, reason, now=now, stop_token_id=stop_token)
        return reason

    def reset(self) -> None:
        """Drop every request and return the whole KV pool. Statistics are kept."""
        self._waiting.clear()
        self._preempted.clear()
        self._running.clear()
        self._by_request.clear()
        self.block_manager.reset()

    # -- the step -------------------------------------------------------------------------

    def schedule(self, *, now: float | None = None) -> SchedulerOutput:
        """Assemble the next batch.

        Order of business, and why:

        1. Publish the blocks finished by the previous step, so this step's admissions can
           adopt them.
        2. Running sequences first. They already hold blocks and a client is already
           waiting on their tokens; making them queue behind new arrivals would turn a
           burst of admissions into a latency spike for everybody already in flight.
        3. Preempted sequences next, ahead of the waiting queue: they are the only
           sequences that can be *losing* work, and resuming them promptly is what keeps
           preemption from turning into livelock.
        4. New admissions last, under the configured policy, until the token budget, the
           sequence budget or the KV pool says stop.
        """
        timestamp = time.perf_counter() if now is None else now
        out = SchedulerOutput()
        budget = _Budget(self.config.max_num_batched_tokens, self.config.max_num_seqs)
        for seq in self._running:
            self.block_manager.publish_computed_blocks(seq)
        scheduled_ids: set[int] = set()
        prefills: list[ScheduledSeq] = []
        decodes: list[ScheduledSeq] = []
        preempted_now: set[int] = set()

        self._schedule_running(out, budget, scheduled_ids, prefills, decodes, preempted_now)
        self._schedule_admissions(
            out, budget, scheduled_ids, prefills, decodes, preempted_now, timestamp
        )

        out.scheduled = [*prefills, *decodes]
        out.num_prefill_seqs = len(prefills)
        out.num_decode_seqs = len(decodes)
        out.num_prefill_tokens = sum(item.num_new_tokens for item in prefills)
        out.num_decode_tokens = sum(item.num_new_tokens for item in decodes)
        self._num_steps += 1
        self._num_prefill_tokens += out.num_prefill_tokens
        self._num_decode_tokens += out.num_decode_tokens
        self._num_cached_tokens += out.num_cached_tokens
        return out

    def _schedule_running(
        self,
        out: SchedulerOutput,
        budget: _Budget,
        scheduled_ids: set[int],
        prefills: list[ScheduledSeq],
        decodes: list[ScheduledSeq],
        preempted_now: set[int],
    ) -> None:
        """Give every running sequence its next chunk or its next decode token."""
        for seq in list(self._running):
            if budget.exhausted:
                break
            if seq.status is not SeqStatus.RUNNING:
                continue  # preempted earlier in this same loop as somebody else's victim
            num_new = seq.num_new_tokens_to_compute(budget.tokens)
            if num_new <= 0:
                continue
            if not self.config.enable_chunked_prefill and num_new < seq.num_uncomputed_tokens:
                continue  # the whole prompt does not fit this step and may not be split
            if not self._make_room(seq, num_new, scheduled_ids, out, preempted_now):
                continue  # the sequence itself became the victim
            self._emit(seq, num_new, budget, scheduled_ids, prefills, decodes)

    def _schedule_admissions(
        self,
        out: SchedulerOutput,
        budget: _Budget,
        scheduled_ids: set[int],
        prefills: list[ScheduledSeq],
        decodes: list[ScheduledSeq],
        preempted_now: set[int],
        timestamp: float,
    ) -> None:
        """Resume preempted sequences, then admit new ones, while the budgets allow."""
        while not budget.exhausted:
            from_preempted = bool(self._preempted)
            seq = self._preempted[0] if from_preempted else self._waiting.peek()
            if seq is None:
                return
            if seq.seq_id in preempted_now:
                return  # preempted during this very step; do not readmit it immediately
            if not self.block_manager.can_allocate(seq):
                return  # head-of-line: a later request must not jump an unfundable one
            cached_tokens = self.block_manager.allocate(seq)
            num_new = seq.num_new_tokens_to_compute(budget.tokens)
            if num_new <= 0 or (
                not self.config.enable_chunked_prefill and num_new < seq.num_uncomputed_tokens
            ):
                self.block_manager.release(seq)
                return
            if from_preempted:
                self._preempted.popleft()
            else:
                self._waiting.pop()
            seq.status = SeqStatus.RUNNING
            if seq.timing.t_first_scheduled is None:
                seq.timing.t_first_scheduled = timestamp
            self._running.append(seq)
            self._num_admitted += 1
            out.num_cached_tokens += cached_tokens
            self._emit(
                seq, num_new, budget, scheduled_ids, prefills, decodes, cached_tokens=cached_tokens
            )

    def _emit(
        self,
        seq: Sequence,
        num_new: int,
        budget: _Budget,
        scheduled_ids: set[int],
        prefills: list[ScheduledSeq],
        decodes: list[ScheduledSeq],
        *,
        cached_tokens: int = 0,
    ) -> None:
        """Allocate the slots for ``num_new`` tokens and record the sequence in the batch."""
        start = seq.num_computed_tokens
        end = start + num_new
        slots = self.block_manager.append_slots(seq, num_new)
        is_prefill = start < seq.num_prompt_tokens
        item = ScheduledSeq(
            seq=seq,
            num_new_tokens=num_new,
            context_len=end,
            token_ids=seq.tokens_in_range(start, end),
            slot_mapping=slots,
            block_table=list(seq.block_table),
            is_prefill=is_prefill,
            is_chunk=is_prefill and end < seq.num_prompt_tokens,
            num_cached_tokens=cached_tokens,
        )
        seq.advance_computed(num_new)
        budget.tokens -= num_new
        budget.seqs -= 1
        scheduled_ids.add(seq.seq_id)
        (prefills if is_prefill else decodes).append(item)

    # -- preemption -------------------------------------------------------------------------

    def _make_room(
        self,
        seq: Sequence,
        num_new: int,
        scheduled_ids: set[int],
        out: SchedulerOutput,
        preempted_now: set[int],
    ) -> bool:
        """Preempt until ``seq`` can grow by ``num_new`` tokens. Returns whether it can.

        ``False`` means the only sequence left to preempt was ``seq`` itself, which then
        goes back to the queue: the pool is full of sequences that are already scheduled in
        this step, and giving up the newest work is better than failing the request.
        """
        while not self.block_manager.can_append(seq, num_new):
            victim = self._pick_victim(scheduled_ids, seq)
            if victim is None:
                self._preempt(seq, out, preempted_now)
                return False
            self._preempt(victim, out, preempted_now)
        return True

    def _pick_victim(self, scheduled_ids: set[int], requester: Sequence) -> Sequence | None:
        """Lowest priority first, then the most recently admitted sequence.

        Sequences already scheduled in this step are off limits: their slots are in the
        batch being built and their blocks are about to be written by the forward pass.
        """
        candidates = [
            other
            for other in self._running
            if other is not requester
            and other.seq_id not in scheduled_ids
            and other.status is SeqStatus.RUNNING
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda other: (other.priority, -other.seq_id))

    def _preempt(self, seq: Sequence, out: SchedulerOutput, preempted_now: set[int]) -> None:
        """Take a sequence's blocks and send it back to the front of the queue."""
        self.block_manager.preempt(seq)
        if seq in self._running:
            self._running.remove(seq)
        self._preempted.append(seq)
        preempted_now.add(seq.seq_id)
        out.preempted.append(seq)
        self._num_preemptions += 1
        logger.debug(
            "preempted %s (tenant=%s, priority=%d, preemptions=%d)",
            seq.request_id,
            seq.tenant_id,
            seq.priority,
            seq.num_preemptions,
        )

    # -- internals ----------------------------------------------------------------------------

    def _detach(self, seq: Sequence) -> None:
        """Remove a sequence from every queue and from the request index."""
        if seq in self._running:
            self._running.remove(seq)
        elif seq in self._preempted:
            self._preempted.remove(seq)
        else:
            self._waiting.remove(seq)
        self._by_request.pop(seq.request_id, None)

    # -- observability --------------------------------------------------------------------------

    def stats(self) -> dict[str, int | float]:
        """Counters for the engine's ``stats()`` endpoint and the Prometheus exporter.

        Counts and occupancies only: rates and latencies are derived by whoever samples
        this, from their own clock.
        """
        out: dict[str, int | float] = {
            "num_waiting": len(self._waiting),
            "num_preempted": len(self._preempted),
            "num_running": len(self._running),
            "num_unfinished": len(self._by_request),
            "num_finished": self._num_finished,
            "num_admitted": self._num_admitted,
            "num_preemptions": self._num_preemptions,
            "num_steps": self._num_steps,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "num_cached_tokens": self._num_cached_tokens,
        }
        out.update(self.block_manager.stats())
        return out

    def check_invariants(self) -> None:
        """Verify queue membership, statuses and the block pool. Tests and debugging only."""
        self.block_manager.check_invariants()
        seen: set[int] = set()
        for seq in self._running:
            if seq.status is not SeqStatus.RUNNING:
                raise ValueError(f"{seq.request_id!r} is in the running set as {seq.status}")
            seen.add(seq.seq_id)
        for seq in self._preempted:
            if seq.status is not SeqStatus.PREEMPTED:
                raise ValueError(f"{seq.request_id!r} is in the preempted queue as {seq.status}")
            if seq.block_table:
                raise ValueError(f"preempted {seq.request_id!r} still holds blocks")
            seen.add(seq.seq_id)
        for seq in self._waiting:
            if seq.status is not SeqStatus.WAITING:
                raise ValueError(f"{seq.request_id!r} is in the wait queue as {seq.status}")
            seen.add(seq.seq_id)
        if len(seen) != len(self._by_request):
            raise ValueError(
                f"{len(seen)} sequences are queued but {len(self._by_request)} are in flight"
            )
        for seq in self._by_request.values():
            if seq.num_computed_tokens > seq.num_tokens:
                raise ValueError(f"{seq.request_id!r} computed more tokens than it holds")

    def __repr__(self) -> str:
        return (
            f"Scheduler(policy={self.config.policy}, running={len(self._running)}, "
            f"preempted={len(self._preempted)}, waiting={len(self._waiting)}, "
            f"blocks_free={self.block_manager.num_free_blocks})"
        )
