# Architecture decision records

One file per decision that would otherwise be re-litigated every time someone new reads the
code. Each records the context at the time, the decision, and the consequences we accepted —
including the ones we did not like.

| ADR | Decision |
| --- | --- |
| [0001](ADR-0001-reference-engine-alongside-vllm.md) | Build a reference engine *and* keep vLLM a first-class backend |
| [0002](ADR-0002-recompute-vs-swap-preemption.md) | Preempt by recompute, not by swapping KV to host memory |
| [0003](ADR-0003-hash-based-prefix-cache.md) | Content-addressed (hashed) prefix cache over a radix tree |
| [0004](ADR-0004-rejection-sampling-verification.md) | Verify speculative drafts with rejection sampling, not "accept if it matches" |
| [0005](ADR-0005-sgmv-grouping-for-lora.md) | Serve multi-LoRA by grouping tokens per slot (SGMV), not by masking or merging |
| [0006](ADR-0006-canary-slo-thresholds.md) | Gate canaries on error rate, absolute p95 TTFT and a ratio against stable |

Format: Status / Context / Decision / Consequences / Alternatives considered. Numbers that
would justify a decision empirically live in `results/` and are rendered into
[`../results.md`](../results.md); an ADR states the reasoning, not a measurement.
