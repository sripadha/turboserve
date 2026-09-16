# turboserve documentation

turboserve serves many tenants from one GPU pool. It contains a from-scratch reference LLM
engine (continuous batching, paged KV cache with automatic prefix caching, speculative
decoding, batched multi-LoRA), an OpenAI-compatible multi-tenant gateway with SLO-gated
canaries and chaos testing, and a vLLM + Kubernetes production deployment path.

Every behavioural claim on these pages is backed by code under `src/turboserve/` or by a
test under `tests/`; every number is rendered from a result file under `results/`.

## Pages

### Start here

| Page | Contents |
| --- | --- |
| [`architecture.md`](architecture.md) | The whole system in six diagrams: request path, engine step loop, block and prefix cache, canary state machine, Kubernetes topology, measurement pipeline |
| [`contracts.md`](contracts.md) | The shared types every module codes against: sampling and finish reasons, attention metadata, LoRA context, the backend protocol, the result schema |

### The engine

| Page | Contents |
| --- | --- |
| [`engine.md`](engine.md) | Configuration, the step loop, KV-pool sizing and memory profiling, the async engine, the naive baselines, what is and is not implemented |
| [`scheduler.md`](scheduler.md) | Continuous batching, chunked prefill, recompute preemption, FCFS and tenant-fair policies |
| [`prefix-caching.md`](prefix-caching.md) | Content-addressed block hashing, lookup/insert/evict, the recycler protocol, hit-rate accounting |
| [`model.md`](model.md) | Qwen2/Llama with paged attention: packed varlen batches, the Triton decode kernel, weight loading, the LoRA linear hook |
| [`speculative-decoding.md`](speculative-decoding.md) | Model and n-gram drafters, rejection-sampling verification, KV rollback, acceptance rate |
| [`multi-lora.md`](multi-lora.md) | Adapter loading, the GPU slot registry and its eviction, grouped (SGMV-style) LoRA linears, the BGMV kernel |

### Serving and delivery

| Page | Contents |
| --- | --- |
| [`gateway.md`](gateway.md) | OpenAI-compatible routes, auth and tenants, quotas, backends, routing and retries, metrics and cost attribution |
| [`canary-and-chaos.md`](canary-and-chaos.md) | SLO gates and the progressive-delivery state machine; fault schedules, breakable replicas and the chaos harness |
| [`kubernetes.md`](kubernetes.md) | Helm chart values, kustomize overlays, the kind end-to-end run, Prometheus rules and the Grafana dashboard |
| [`runbook.md`](runbook.md) | Install, rotate tenant keys, add an adapter, roll a canary, roll back, read the dashboards, capacity planning, triage |

### Measurement

| Page | Contents |
| --- | --- |
| [`benchmarking.md`](benchmarking.md) | Exact metric definitions (TTFT, ITL, TPOT, E2E, goodput), profiles, prompt sources, how a result file is produced |
| [`scenarios.md`](scenarios.md) | The five experiments: what each varies, what it holds fixed, and the `turboserve bench` commands |
| [`vastai.md`](vastai.md) | Renting the H100, running the suite on it, bringing the results home (`make bench-h100`) |
| [`results.md`](results.md) | The result tables and plots — **generated** by `make results` from `results/*.json`, never hand-edited, each table labelled measured or projected |

### Decisions

| Page | Contents |
| --- | --- |
| [`adr/`](adr/README.md) | Six architecture decision records: reference engine alongside vLLM, recompute vs swap preemption, hashed prefix cache, rejection-sampling verification, SGMV grouping for LoRA, canary SLO thresholds |

## Reading order

1. [`architecture.md`](architecture.md) for the shape of the system, then
   [`contracts.md`](contracts.md) for the types every page after it uses.
2. [`engine.md`](engine.md), then [`scheduler.md`](scheduler.md),
   [`prefix-caching.md`](prefix-caching.md), [`model.md`](model.md),
   [`speculative-decoding.md`](speculative-decoding.md), [`multi-lora.md`](multi-lora.md)
   for the engine internals in dependency order.
3. [`gateway.md`](gateway.md) and [`canary-and-chaos.md`](canary-and-chaos.md) for the
   serving and delivery layer.
4. [`benchmarking.md`](benchmarking.md), [`scenarios.md`](scenarios.md) and
   [`results.md`](results.md) for how the system is measured and what it measured.
5. [`kubernetes.md`](kubernetes.md) and [`runbook.md`](runbook.md) for running it.

## Conventions

- Every behavioural claim on these pages is backed by code in `src/turboserve/` or by a test
  in `tests/`.
- **No page states a performance number, a target or an expected figure.** Numbers appear
  only in tables rendered from `results/**/*.json`, which record the hardware, software
  versions, configuration and timestamp of the run that produced them, and each rendered
  table carries a one-line provenance note saying whether the run was measured or projected.
- Diagrams are mermaid, rendered by GitHub.
- Anything that has *not* been executed — on a GPU, against a Kubernetes cluster, against a
  rented instance — is said so explicitly on the page that claims it; see the "Limitations"
  section each module page ends with.

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the development workflow, test markers and
module ownership, and [`../README.md`](../README.md) for the quickstart and the CLI reference.
