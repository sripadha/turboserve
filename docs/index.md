# turboserve documentation

turboserve serves many tenants from one GPU pool. It contains a from-scratch reference LLM
engine (continuous batching, paged KV cache with automatic prefix caching, speculative
decoding, batched multi-LoRA), an OpenAI-compatible multi-tenant gateway with SLO-gated
canaries and chaos testing, and a vLLM + Kubernetes production deployment path.

**Status: scaffold.** The pages below are the planned documentation set. Pages are written
by the owner of the module they describe and appear as that module lands; a link without a
page yet means the module has not been written.

`kubernetes.md` also covers three deliverables that are not in the repository yet —
`docker-compose.yml`, `.github/workflows/kind-e2e.yml` and the `helm lint` + `kubeconform`
CI job. They are blocked on the Helm chart and the Prometheus/Grafana assets, and are
tracked against the `deploy/` owner in
[`../CONTRIBUTING.md`](../CONTRIBUTING.md#deliverables-not-built-yet).

`benchmarking.md` covers one more absent deliverable in the same table: `scripts/vastai/` and
the `make bench-h100` target that chains its scripts into the single command that runs the
whole suite on a rented H100. They land with the `bench/` scenarios.

## Pages

| Page | Contents |
| --- | --- |
| `architecture.md` | End-to-end request path, engine step loop, block and prefix cache, canary state machine, Kubernetes topology (mermaid diagrams) |
| `engine.md` | The reference engine: configuration, step loop, memory sizing, what is and is not implemented |
| `scheduler.md` | Continuous batching, chunked prefill, recompute preemption, FCFS and tenant-fair policies |
| `prefix-caching.md` | Content-addressed block hashing, lookup/insert/evict, hit-rate accounting |
| `speculative-decoding.md` | Model and n-gram drafters, rejection-sampling verification, KV rollback, acceptance rate |
| `multi-lora.md` | Adapter loading, GPU slot registry and eviction, grouped (SGMV-style) LoRA linears |
| `gateway.md` | OpenAI-compatible routes, auth and tenants, quotas, backends, routing, metrics |
| `canary-and-chaos.md` | SLO gates and the progressive-delivery state machine; fault schedules and the chaos harness |
| `kubernetes.md` | Helm chart, kustomize overlays, kind e2e, Prometheus rules and Grafana dashboards |
| `benchmarking.md` | Exact metric definitions (TTFT, ITL, TPOT, E2E, goodput), the five scenarios, profiles, how results JSON is produced |
| `results.md` | Measured results, generated from `results/*.json` — never hand-edited |
| `runbook.md` | Deploy, rotate tenant keys, add an adapter, roll a canary, roll back, read dashboards, capacity planning |
| `adr/` | Architecture decision records: reference engine vs vLLM-only, recompute vs swap preemption, hash-based prefix cache, rejection-sampling verification, SGMV grouping, canary SLO thresholds |

## Reading order

1. `architecture.md` for the shape of the system.
2. `engine.md`, then `scheduler.md`, `prefix-caching.md`, `speculative-decoding.md`,
   `multi-lora.md` for the engine internals in dependency order.
3. `gateway.md` and `canary-and-chaos.md` for the serving and delivery layer.
4. `benchmarking.md` and `results.md` for how the system is measured and what it measured.
5. `kubernetes.md` and `runbook.md` for running it.

## Conventions

- Every behavioural claim on these pages is backed by code in `src/turboserve/` or by a test
  in `tests/`.
- Every performance number cites a file under `results/`, which records the hardware,
  software versions, configuration and timestamp of the run that produced it. Numbers appear
  only in tables rendered from those files; no page states a target or an expected figure.
- Diagrams are mermaid, rendered by GitHub.

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the development workflow, test markers and
module ownership.
