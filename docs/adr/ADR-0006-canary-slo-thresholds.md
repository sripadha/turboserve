# ADR-0006 — What a canary is gated on

**Status:** accepted.
**Date:** 2026-09.

## Context

A progressive rollout needs a rule for "is the candidate still allowed to keep its
traffic?". The rule has to hold under three awkward conditions: a canary with very little
traffic, a canary that is *slower* than stable without being broken, and an infrastructure
where the metrics themselves can stop arriving.

## Decision

`CanaryConfig` gates on four things, evaluated in this order, with the weight steps
`[1, 5, 25, 50, 100]`:

1. **Sample size.** Below `min_requests` in the window, hold — never advance and never roll
   back on noise. If the canary stays below that threshold for `stall_timeout_s`, roll back:
   a rollout that is getting no traffic is a rollout that cannot be evaluated, and leaving
   it parked is worse than reverting it.
2. **Absolute error rate** over the window, against `max_error_rate` (default 0.005).
3. **Absolute p95 TTFT**, against `max_p95_ttft_ms` when configured. Optional because it is
   deployment-specific; unset means "not asserted", not "zero".
4. **p95 ratio against the stable lane**, against `max_p95_ratio_vs_stable` (default 1.25),
   measured over the *same* window. This is the gate that survives a traffic spike: both
   lanes get slower together, the ratio does not move, and a rollout is not aborted for
   something the candidate did not cause.

Only after all four pass, and `step_hold_s` has elapsed at the current weight, does the
weight advance. An overall `deadline_s` aborts a rollout that is neither failing nor
finishing.

## Consequences

- **An empty window is 0 % errors, not 100 %.** `LaneSummary.error_rate` returns 0.0 for
  zero requests. Combined with the `min_requests` hold this means "no data" can only ever
  cause a hold or the stall rollback, never an advance and never a spurious rollback.
- **Failed requests count as errors but are excluded from the percentiles.** A request that
  failed has no meaningful TTFT; including it would let a burst of fast failures *improve*
  the latency gate.
- **The same gate runs on two evidence sources.** In-process sliding windows (single-replica
  gateway) and Prometheus queries (multi-replica) both produce a `LaneSummary`, and
  `tick(now, stable=..., canary=...)` takes them. There is one implementation of the
  policy, so the two cannot disagree.
- **A metrics outage rolls back, it does not coast.** `PrometheusError` is raised rather
  than folded into "no data"; the runner tolerates `max_metric_failures` consecutive poll
  failures and then rolls back. A gate that cannot read its metrics must fail loudly.
- **One percentile definition for the whole repository.** The controller imports
  `turboserve.bench.records.percentile` (linear interpolation) rather than defining its own,
  so the canary's p95 and the benchmark's p95 agree to the digit.
- **The thresholds are configuration, not constants.** They live in `configs/canary.yaml`;
  the defaults above are starting points chosen to be strict enough to catch a bad build and
  loose enough not to trip on ordinary variance, and a real deployment should set them from
  its own error budget.

## Alternatives considered

- **Ratio only, no absolute gates.** Rejected: a stable lane that is already breaching its
  SLO would let an equally bad candidate through.
- **Absolute gates only.** Rejected: it converts every load spike into a rollback.
- **Statistical significance testing between lanes.** Attractive and rejected as
  over-fitting the tool to the demonstration: at the sample sizes a canary window actually
  collects, a fixed threshold that an operator can read off a dashboard is more useful than
  a p-value nobody will calibrate.
