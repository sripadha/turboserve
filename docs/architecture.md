# Architecture

turboserve is one Python package with five layers. Each layer is useful on its own and
talks to the next through a small, typed contract, which is why the same gateway can front
a from-scratch engine and a production vLLM server, and why the same benchmark harness can
measure both.

| Layer | Package | What it owns |
| --- | --- | --- |
| Gateway | `turboserve.gateway` | OpenAI-compatible HTTP, API keys to tenants, quotas, routing, SSE, Prometheus metrics, cost attribution |
| Engine | `turboserve.engine` | Continuous batching, paged KV cache with prefix caching, Qwen2/Llama with paged attention, speculative decoding, batched multi-LoRA |
| Delivery | `turboserve.canary`, `turboserve.chaos` | SLO-gated progressive rollout; fault injection and the chaos harness |
| Measurement | `turboserve.bench` | Load generation, per-request records, the five scenarios, rendered result pages |
| Deployment | `deploy/` | Helm chart, kustomize overlays, kind end-to-end, Prometheus rules, Grafana dashboard, the production vLLM path |

The contracts between them are written down once, in
[`contracts.md`](contracts.md): `SamplingParams`, `AttnMetadata`, `LoRAContext`,
`GenerateRequest`/`TokenEvent`, `RunResult`. Every page below assumes them.

---

## 1. The request path

A request enters as OpenAI JSON and leaves as an SSE stream. Between those two points it
is authenticated, charged against a quota, routed to a lane and a replica, turned into
engine tokens, and accounted for. Solid arrows are the data path; dashed arrows are control
and instrumentation.

```mermaid
flowchart LR
  client["OpenAI client<br/>or bench loadgen"] -->|"POST /v1/chat/completions"| auth

  subgraph gw["gateway (FastAPI)"]
    auth["auth.py<br/>sha256 key to tenant<br/>401 / 403"]
    limits["limits.py<br/>rpm, tpm token buckets<br/>concurrency, 429 + Retry-After"]
    tmpl["chat_template.py<br/>messages to prompt"]
    router["router.py<br/>model to pool, lane weights,<br/>health, retry before first byte"]
    usage["usage.py + metrics.py<br/>TTFT / TPOT / E2E, tokens, $"]
    auth --> limits --> tmpl --> router
    router -.-> usage
  end

  router --> local["LocalEngineBackend"]
  router --> compat["OpenAICompatBackend<br/>vLLM / TGI over HTTP"]
  router --> mock["MockBackend<br/>tests, chaos, kind e2e"]

  local --> engine["AsyncLLMEngine"]
  engine --> loop(["step loop — see §2"])

  loop -->|"RequestOutput deltas"| local
  local -->|"TokenEvent"| router
  compat -->|"TokenEvent"| router
  mock -->|"TokenEvent"| router
  router -->|"data: {...}\\n\\n ... [DONE]"| client

  usage -.-> prom[("Prometheus<br/>turboserve_gateway_*")]
  prom -.-> canary["canary controller"]
  canary -. "lane weight" .-> router
```

Two rules of the router are worth stating because everything else follows from them:

- A retry is only legal **before the first byte**. Once a token has been written to the
  socket the request cannot be moved to another replica, so a later failure is delivered as
  a terminating error event, not as an exception.
- The tenant's concurrency slot is taken **before** the backend is asked for anything, and
  released in a `finally`. A stream abandoned by the client therefore frees both the slot
  and the engine's KV blocks.

---

## 2. The engine step loop

One step produces at most one token per sampling sequence — but it can produce them for
sixty-four sequences at once, and it can spend the same step finishing someone else's
prompt. That is continuous batching: there is no notion of a batch that starts and ends.

```mermaid
flowchart TD
  A["scheduler.schedule()"] --> B{"anything to run?"}
  B -- no --> A
  B -- yes --> C["SchedulerOutput<br/>prefills first, then decodes"]
  C --> D["build_attn_metadata()<br/>slot_mapping, block_tables,<br/>context_lens, query_start_loc"]
  D --> E["CausalLM.forward<br/>one flat token vector"]
  E --> F["write K/V into the paged cache<br/>per layer, via slot_mapping"]
  F --> G["paged attention<br/>Triton on CUDA decode, else SDPA"]
  G --> H["compute_logits(hidden, sample_indices)<br/>fp32, only the sampling rows"]
  H --> I["Sampler<br/>penalty, temperature, top-k, top-p"]
  I --> J["scheduler.append_token()<br/>stop ids, EOS, max_tokens"]
  J --> K["StreamingDecoder<br/>incremental detokenise, stop strings"]
  K --> L["RequestOutput deltas"]
  L --> A
```

The scheduler's order within a step is fixed and is what keeps the system live under
pressure:

1. **Running decodes** — one token each. If a sequence cannot get a block, the scheduler
   preempts a victim (recompute mode: its blocks are freed and its progress reset, its
   generated tokens kept) and retries.
2. **Preempted sequences**, at the front of the waiting queue.
3. **Waiting sequences**, admitted while the token budget and `max_num_seqs` allow, with
   prefill chunked to whatever budget is left and prefix-cache hits subtracted from the
   work.

Admission is head-of-line: if the queue head cannot be funded, admission stops for that
step. That is deliberate — it prevents a stream of small requests from starving a large
one — and it is the reason one oversized request slows admissions rather than being
skipped.

`engine.md` and `scheduler.md` carry the details; `speculative-decoding.md` describes the
variant step in which each sequence is fed `k+1` tokens and the extra ones are verified.

---

## 3. Blocks, slots and the prefix cache

KV memory is one flat pool of fixed-size blocks. A sequence owns a *block table* — a list
of block ids — and a token's KV lives at `block_id * block_size + offset`. Nothing is
contiguous, so nothing has to be moved when a sequence grows, and two sequences that begin
with the same tokens can point at the same blocks.

```mermaid
flowchart TD
  subgraph pool["BlockAllocator — every block is in exactly one state"]
    F["free<br/>(FIFO free list)"]
    U["in use<br/>refs > 0"]
    R["retained<br/>refs == 0, held by the cache"]
    F -->|allocate| U
    U -->|"decref to 0, on_zero_refs = True"| R
    U -->|"decref to 0, on_zero_refs = False"| F
    R -->|"adopt (prefix hit)"| U
    R -->|"reclaim / evict (LRU)"| F
  end

  subgraph cache["PrefixCache — content addressed"]
    H["h_i = blake2b(h_i-1 || lora_id || tokens_i)<br/>full blocks only"]
    H --> M["hash to block id"]
    M --> LRU["evictable LRU<br/>(unpinned, refs == 0)"]
  end

  SEQ["Sequence.block_table<br/>[12, 40, 7, ...]"] --> U
  SEQ -.->|"match(token_ids, lora_id)"| M
  cache -->|"BlockRecycler protocol"| pool
```

Three consequences fall out of this shape:

- **The allocator knows nothing about hashes.** The cache is plugged in as a
  `BlockRecycler` with two methods (`on_zero_refs`, `reclaim`); returning `False` from
  `on_zero_refs` everywhere turns the allocator back into a plain free list. That is the
  whole coupling.
- **`lora_id` is mixed into the hash chain**, so two tenants using different adapters can
  never share KV blocks even when their prompts are identical.
- **Preemption is recompute, not swap.** A preempted sequence's blocks go back to the pool
  and its prompt is recomputed later — usually cheaply, because the prefix cache still has
  most of it. The trade is written up in [ADR-0002](adr/ADR-0002-recompute-vs-swap-preemption.md).

---

## 4. The canary state machine

A rollout is a small state machine over a sliding window of request outcomes. It is pure:
no clock of its own, no I/O, and it accepts evidence from either the gateway's in-process
windows or a Prometheus query — both arrive as the same `LaneSummary`.

```mermaid
stateDiagram-v2
  [*] --> IDLE
  IDLE --> CANARY: start(version)
  CANARY --> CANARY: HOLD<br/>(too few requests, or hold time not elapsed)
  CANARY --> CANARY: ADVANCE<br/>(gates pass, next step: 1, 5, 25, 50, 100)
  CANARY --> PROMOTED: PROMOTE<br/>(gates pass on the final step)
  CANARY --> ROLLED_BACK: ROLLBACK<br/>(error rate, p95 TTFT, ratio vs stable, stall, deadline)
  PROMOTED --> [*]
  ROLLED_BACK --> [*]
```

The gates, in the order they are evaluated: minimum sample size (or a stall rollback if the
canary is getting no traffic at all), absolute error rate, absolute p95 TTFT, p95 ratio
against the *stable* lane measured over the same window, then the hold time. Only after all
of them pass does the weight move. Thresholds live in `configs/canary.yaml`; the reasoning
behind the defaults is [ADR-0006](adr/ADR-0006-canary-slo-thresholds.md).

---

## 5. Kubernetes topology

The chart deploys the gateway and the engine as separate workloads, because they scale on
different things: the gateway is CPU- and connection-bound and scales horizontally, while
the engine owns a GPU and does not.

```mermaid
flowchart TB
  ing["Ingress<br/>(optional)"] --> svc

  subgraph ns["namespace: turboserve"]
    svc["Service: turboserve-gateway"]
    svcS["Service: turboserve-gateway-stable"]
    svcC["Service: turboserve-gateway-canary<br/>nginx canary-weight annotation"]
    svc --> dep
    svcS --> dep
    svcC --> depC

    dep["Deployment: turboserve-gateway<br/>(stable lane)"]
    depC["Deployment: turboserve-gateway-canary"]
    hpa["HPA<br/>CPU + turboserve_gateway_queue_depth"] -.-> dep
    pdb["PodDisruptionBudget"] -.-> dep

    dep --> esvc["Service: turboserve-engine"]
    depC --> esvc
    esvc --> edep["Deployment: turboserve-engine<br/>mode: vllm | reference | mock<br/>nvidia.com/gpu: 1"]

    cm["ConfigMap: models.yaml"] -.-> dep
    sec["Secret: tenants.yaml<br/>sha256 key digests"] -.-> dep
    pvc[("PVC: model cache")] -.-> edep
    apvc[("PVC: adapters<br/>+ sync init container")] -.-> edep

    sm["ServiceMonitor"] -.-> dep
    pr["PrometheusRule<br/>error rate, TTFT p95 SLO alerts"]
    gcm["ConfigMap: Grafana dashboard"]
    np["NetworkPolicy"] -.-> dep
  end

  sm -.-> promo[("Prometheus Operator")]
  promo -.-> pr
  promo -.-> graf["Grafana"]
  gcm -.-> graf
  promo -.-> can["turboserve canary run --kube<br/>Argo Rollout or weighted Services"]
  can -.-> depC
```

Two label facts matter when reading any dashboard or alert:

- A metric's `lane` label is the **backend's** lane from `models.yaml`, not the pod's. The
  ServiceMonitor therefore relabels the pod's `turboserve.io/lane` to `pod_lane`; a target
  label called `lane` would collide and Prometheus would rename the real one.
- The label sets are not uniform. `turboserve_gateway_rate_limited_total` is
  `(tenant, limit)` with no model or lane; `turboserve_gateway_queue_depth` is `(model)`
  with no lane. A query that assumes one label set returns nothing.

`kubernetes.md` has the values reference and the kind end-to-end run; `runbook.md` has the
operations.

---

## 6. How the pieces are measured

Every scenario drives load through the *same* `Backend` protocol the gateway uses, so the
reference engine, a naive `transformers.generate` baseline and a real vLLM server are
measured by identical client code and identical percentile definitions.

```mermaid
flowchart LR
  P["profiles.yaml<br/>h100 / dev-2060"] --> S["scenario<br/>naive_vs_cb, prefix_cache,<br/>spec_decode, multi_lora, chaos"]
  PR["prompts.py<br/>seeded synthetic token ids"] --> S
  S --> LG["loadgen.py<br/>closed loop / Poisson open loop"]
  LG --> B["Backend<br/>reference | vllm | naive | mock"]
  B --> REC["RequestRecord<br/>t_send_ns, t_first_ns, itl_ns, t_last_ns"]
  REC --> RR["RunResult<br/>hardware, software, git sha,<br/>$/GPU-hour, provenance"]
  RR --> JSON[("results/&lt;scenario&gt;/*.json")]
  JSON --> REP["report.py"]
  REP --> MD["results/README.md<br/>docs/results.md<br/>results/plots/*.png"]
```

No number in this repository is typed by a human into a markdown file. `make results`
renders every table from the JSON, and each table carries a one-line provenance note saying
what hardware produced it and whether the run was measured or projected.
`benchmarking.md` defines the metrics; `scenarios.md` defines the experiments.
