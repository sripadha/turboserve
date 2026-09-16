# ADR-0005 — Serve multi-LoRA by grouping tokens per slot

**Status:** accepted.
**Date:** 2026-09.

## Context

The point of multi-tenant LoRA is that one batch mixes adapters: tenant A's request, tenant
B's request and a base-model request in the same forward pass. Each of the seven wrapped
projections must therefore apply a *different* `B @ A` to different rows of the same token
matrix. Three ways to do it:

1. **Merge.** Fold each adapter into a copy of the base weights. Exact and fast per request,
   but it costs one full copy of the model per adapter and makes a mixed batch impossible.
2. **Mask.** Compute every resident adapter's delta for every token and zero the rows that
   do not belong to it. Simple, and wastes work in proportion to the number of resident
   adapters.
3. **Group.** Sort the token indices by slot, take one contiguous segment per active slot,
   and do one pair of small matmuls per segment (SGMV — segmented gather matrix-vector, from
   the Punica work). Work is proportional to the tokens that actually use an adapter.

## Decision

Group. `LoRALinear` holds stacked `A[slots, r_max, in]`, `B[slots, out, r_max]` and
`scaling[slots]`; `LoRABatch` carries a per-token slot vector plus the `order` permutation
and `(slot, start, end)` segments; the forward pass runs one gather/matmul/scatter per
active slot and skips slot 0, which is the base model. A Triton BGMV kernel covers the
decode-shaped case (one token per sequence), where per-segment matmuls are too small to be
worth launching.

## Consequences

- **Cost scales with adapter-using tokens, not with residency.** A pool with a hundred
  adapters resident and a batch that uses three of them does three segments of work.
- **The grouping is computed on the host.** `LoRABatch.from_token_slots` sorts the slot list
  in Python; deriving it on the device would mean a `unique` and a synchronisation per
  linear per layer — dozens of syncs per decoded token. The same reasoning is why
  `LoRAContext.active_slots` is precomputed rather than inferred.
- **Slot 0 has no storage.** Slot `s` lives at row `s - 1`. Reserving a dead row for the
  base model would waste `1 / num_slots` of the adapter pool, which is real VRAM in exactly
  the configuration this feature exists to make cheap.
- **One rank budget per pool.** Slots are `max_lora_rank` wide, so a rank-8 adapter in a
  rank-64 pool pays for 64. The padded columns are zeroed on every load, so it is correct —
  just wasteful. Grouping adapters into per-rank pools is not implemented.
- **The scheduler is not adapter-aware.** It does not prefer sequences whose adapters are
  already resident, so a pool much smaller than the working set can thrash. The registry
  warns when the slot count is below `max_num_seqs`, and a step that genuinely overflows
  raises `LoRACapacityError` rather than quietly serving the base model.
- **Correctness is anchored to PEFT.** The tests compare a grouped mixed-adapter batch
  against PEFT's own merged model for each adapter; the merge path is kept as a test oracle
  precisely because it is the definition.

## Alternatives considered

- **Masking.** Rejected on the grounds above; it is, however, what the reference path
  degenerates to when exactly one adapter is active, and it is far simpler to read — which
  is why the non-Triton path is written as an explicit segment loop rather than as index
  gymnastics.
- **One merged copy per adapter.** Rejected: it is the baseline the VRAM figure is measured
  *against*, and it cannot serve a mixed batch at all.
