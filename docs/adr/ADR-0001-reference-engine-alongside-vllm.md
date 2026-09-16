# ADR-0001 — A reference engine alongside vLLM, not instead of it

**Status:** accepted.
**Date:** 2026-09.

## Context

The serving techniques this project is about — continuous batching, paged KV with prefix
caching, speculative decoding, batched multi-LoRA — all exist in vLLM. A repository could
therefore be built in two ways:

1. **vLLM only.** A gateway, a deployment path, a benchmark harness, and vLLM flags. Fast to
   write, production-shaped, and it measures a real engine.
2. **A from-scratch engine only.** Every technique implemented here. Instructive, but the
   numbers it produces have no reference point: a slow implementation of continuous batching
   still beats no batching, and nothing would say whether the result is respectable.

Neither is satisfying. (1) can demonstrate that flags were set; it cannot demonstrate that
the mechanism is understood. (2) can demonstrate the mechanism; it cannot show the mechanism
is worth anything.

## Decision

Build both, and make them interchangeable *at the gateway's backend boundary*.

- `turboserve.engine` is a real, complete implementation: block allocator, hashed prefix
  cache, chunked-prefill scheduler with recompute preemption, Qwen2/Llama with paged
  attention (reference SDPA plus a Triton decode kernel), rejection-sampling speculative
  decoding, grouped multi-LoRA.
- vLLM is reached through `OpenAICompatBackend` — the same `Backend` protocol
  `LocalEngineBackend` implements. The router cannot tell them apart.
- Every benchmark scenario takes a `--url` and can therefore run its identical prompts,
  identical load shape and identical percentile code against vLLM. The reference engine's
  numbers always sit next to a production engine's.

## Consequences

- The measurement code has exactly one client path. A bug in it moves every arm the same
  way, which is the property that makes a *comparison* trustworthy even when the absolute
  numbers are not.
- The reference engine must be honest about what it does not do: no CUDA graphs, no tensor
  parallelism, no varlen prefill kernel, eager dispatch everywhere. Each of those is stated
  in `engine.md`, and each is a reason a production deployment uses the vLLM backend.
- Two engines is more code to keep green. The cost is bounded by the shared contracts in
  `contracts.md`: both sides speak `GenerateRequest`/`TokenEvent`, so the gateway, the
  chaos harness and the bench harness are written once.
- `vllm` is an optional extra, never a hard dependency. A CPU laptop can run the whole test
  suite; only the measurement host installs it.

## Alternatives considered

- **Fork vLLM and modify it.** Rejected: the interesting parts would be diffs against a
  large codebase, unreadable as a demonstration and painful to keep rebased.
- **Reimplement vLLM's kernels.** Rejected as scope: the point is the *systems* behaviour
  (batching, memory, scheduling, multi-tenancy), and a hand-written FlashAttention would
  consume the whole budget without changing what is being shown.
