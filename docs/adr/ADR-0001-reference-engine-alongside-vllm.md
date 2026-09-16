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

## Addendum (2026-09): the gateway is engine-agnostic

SGLang was added as a second production backend, and the decision above is what made it a
configuration change rather than a port. It is worth recording what did *not* have to change.

- **No new backend type.** SGLang is served by the same `OpenAICompatBackend` as vLLM, over
  the same OpenAI-compatible protocol: the same request body (`ignore_eos` and the other
  extra members included), the same SSE framing, the same `stream_options.include_usage`,
  the same LoRA-adapter-as-`model` convention. `configs/models.yaml` names both in one pool
  and the router cannot tell them apart, which is also how an engine migration can be run a
  percentage at a time through the canary lane.
- **What the engines do not share is what they will say about themselves.**
  `OpenAICompatBackend.server_info()` reads `/version` and — SGLang's own —
  `/get_server_info`, and the benchmark scenarios record the answer in the run's
  `config["engine"]`. That is the one addition the second engine needed, and it exists
  because a client cannot otherwise see the launch flags that decide what a number means.
- **The deployment shape is a property of "a process holding a model on a GPU"**, not of
  which process. `engine.mode: sglang` renders the same Deployment, Service, PVC and
  NetworkPolicy as `vllm` with a different image and argv, and everything that has to know
  whether a separate engine workload exists asks one helper.
- **The comparison discipline is unchanged.** Each engine is driven through its own `--url`
  in its own invocation, each row records which server produced it, and each prefix-cache
  pair is measured against a control started on the same engine. An engine's arms are never
  compared against another engine's control.

The consequence to keep in view: "the production engine" is no longer a single thing, so
anything written as though it were — a doc sentence, a values file, a scenario arm — is a
place where the second engine will be forgotten. The engine names live in one mapping
(`turboserve.bench.scenarios.common.REMOTE_ENGINE_LABELS`) and one chart value for that
reason.

## Alternatives considered

- **Fork vLLM and modify it.** Rejected: the interesting parts would be diffs against a
  large codebase, unreadable as a demonstration and painful to keep rebased.
- **Reimplement vLLM's kernels.** Rejected as scope: the point is the *systems* behaviour
  (batching, memory, scheduling, multi-tenancy), and a hand-written FlashAttention would
  consume the whole budget without changing what is being shown.
