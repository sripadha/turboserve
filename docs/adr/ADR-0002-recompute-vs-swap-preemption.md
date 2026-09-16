# ADR-0002 — Preempt by recompute, not by swapping KV to host memory

**Status:** accepted.
**Date:** 2026-09.

## Context

Under load the block pool runs dry: a running sequence needs one more block to write the KV
of the token it is about to produce, and there is none. Something must give up its blocks.
Two classical answers:

- **Swap.** Copy the victim's KV blocks to pinned host memory, free the device blocks, and
  copy them back when the sequence is rescheduled. The victim resumes exactly where it was.
- **Recompute.** Free the victim's blocks, reset its `num_computed_tokens` to zero, keep the
  tokens it has already generated, and let it prefill again from the beginning when it is
  rescheduled.

## Decision

Recompute. `BlockManager.preempt(seq)` frees the block table and calls
`Sequence.reset_for_recompute()`, which zeroes the computed-token counter and the cached
prompt count, keeps `output_token_ids`, sets `PREEMPTED` and increments `num_preemptions`.
Preempted sequences go to the front of the waiting queue.

## Consequences

- **Preemption is a pure scheduler decision.** There is no second memory tier, no pinned
  host arena to size, no copy engine to overlap with compute, and no transfer that can fail
  halfway. `OutOfBlocksError` is caught by the scheduler and turned into a preemption; that
  is the whole mechanism.
- **The prefix cache pays most of the bill.** A recomputed sequence's prompt is usually
  still hashed in the cache, so the "recompute" is frequently a block-table rebuild plus a
  short tail. This is the specific reason recompute and hashed prefix caching are a pair:
  each makes the other cheaper (see [ADR-0003](ADR-0003-hash-based-prefix-cache.md)).
- **Long generations are the bad case.** Recompute cost grows with the tokens already
  produced, and those are never in the prefix cache of another request. A workload of very
  long outputs under heavy memory pressure will do measurable duplicate work.
- **Preemptions are counted, not hidden.** `Scheduler.stats()` exposes `num_preemptions`,
  and the engine surfaces it; a benchmark arm with a preemption count is an arm whose pool
  was undersized, and the result file says so.

## Alternatives considered

- **Swap to host memory.** Rejected for this codebase: it doubles the memory model (device
  pool plus host arena plus a transfer scheduler), and the cases where it clearly wins —
  very long outputs, very tight pools — are exactly the cases a production deployment
  handles by using the vLLM backend with a larger pool.
- **Refuse to admit rather than preempt.** Rejected: admission is already head-of-line
  blocking on the waiting queue, and a *running* sequence can still need a block that does
  not exist. Something must be able to give way, or the engine deadlocks.
- **Evict the prefix cache instead.** This does happen first — the allocator reclaims
  retained blocks from the cache before it declares itself dry — but retained blocks are by
  definition unreferenced, so exhausting them does not help a pool whose blocks are all
  live.
