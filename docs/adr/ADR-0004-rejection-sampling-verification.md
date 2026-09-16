# ADR-0004 — Verify speculative drafts by rejection sampling

**Status:** accepted.
**Date:** 2026-09.

## Context

Speculative decoding proposes `k` tokens with a cheap drafter and checks them with one
forward pass of the target model over `k+1` positions. The check determines whether the
feature is a *speedup* or a *different model*.

The naive check — "keep the drafted token if the target's argmax agrees" — is exact for
greedy decoding and **wrong for sampling**: it biases the output distribution towards
tokens the draft model likes, and the bias is invisible in any latency or throughput
number. A request with `temperature > 0` served this way is not the model the tenant asked
for.

## Decision

Implement the modified rejection sampling of Leviathan et al. (2023) and Chen et al. (2023):
accept draft token `x_i` with probability `min(1, p(x_i) / q(x_i))`; on the first rejection
sample the replacement from the normalised residual `norm(max(0, p - q))`; always append one
bonus token from the target's own distribution at the accepted position. Greedy requests
(`temperature == 0`) take the exact-match path, which is the same rule in the limit.

`verify_sequence(...)` chooses between the two from `SamplingParams.is_greedy`, so the
engine never has to.

## Consequences

- **The output distribution is unchanged**, token for token, whatever `k` is and whatever
  the drafter proposes. A drafter that is bad costs throughput and nothing else, which is
  what makes it safe to enable speculation by default for a model pair.
- **This is testable, and is tested**: a chi-square test over a tiny vocabulary compares the
  verified output distribution with direct target sampling, and the greedy path is asserted
  to be token-identical to plain decoding.
- **A verification step is never decode-only**, because each sequence contributes `k+1`
  query tokens. It therefore takes the reference attention path on every device — the Triton
  decode kernel requires all query lengths to be 1. Speculation trades kernel efficiency for
  fewer target forwards, and on a small `k` with a poor acceptance rate that trade can go
  the wrong way. The acceptance rate is recorded in every result file so the trade is
  visible rather than assumed.
- **An n-gram drafter has no probabilities.** `verify_rejection_sampling` reads
  `draft_probs=None` as a point mass on the drafted token, which makes prompt-lookup
  drafting exact for sampled requests too, rather than greedy-only.
- **Rollback is bookkeeping, not memory management.** Rejected positions keep their KV
  slots; `num_computed_tokens` is moved back and the slots are overwritten next step. There
  is no free/realloc on the rejection path.

## Alternatives considered

- **Greedy-verify only** (accept while the argmax matches). Rejected: it silently changes
  the model for every `temperature > 0` request, which is most chat traffic.
- **Verify against the post-temperature distribution only.** This is what the code does for
  the *target* side; the ADR's point is that the *draft* side must be divided out, which the
  naive check does not do.
- **Tree attention (Medusa/EAGLE-style branching drafts).** Out of scope, not rejected: it
  needs its own attention kernel and its own scheduler shape. Recorded in
  `speculative-decoding.md` as future work.
