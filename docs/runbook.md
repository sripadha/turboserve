# Runbook

Operating turboserve: install it, change who may use it, ship a new build, undo one, read
the dashboards, and size the GPU. Every command below is a command this repository has —
`turboserve --help` at any level lists the real flags.

Assumptions: a Kubernetes cluster with the Prometheus Operator, the chart installed as
release `turboserve` in namespace `turboserve`, and `kubectl`/`helm` configured. The
single-node and laptop paths are noted where they differ.

| I need to… | Section |
| --- | --- |
| Install or upgrade | [1](#1-install-and-upgrade) |
| Add a tenant, rotate a key | [2](#2-tenants-and-api-keys) |
| Add or change an adapter | [3](#3-lora-adapters) |
| Ship a new build behind a gate | [4](#4-roll-a-canary) |
| Undo a bad build, fast | [5](#5-roll-back) |
| Understand an alert | [6](#6-dashboards-and-alerts) |
| Decide how much GPU this needs | [7](#7-capacity-planning) |
| Work out why requests are failing | [8](#8-triage) |

---

## 1. Install and upgrade

```bash
helm upgrade --install turboserve deploy/helm/turboserve \
  --namespace turboserve --create-namespace \
  --set engine.mode=vllm \
  --set engine.model=Qwen/Qwen2.5-7B-Instruct \
  --set tenants.existingSecret=turboserve-tenants \
  --set models.existingConfigMap=turboserve-models \
  --set monitoring.serviceMonitor.enabled=true \
  --set monitoring.prometheusRule.enabled=true
```

`deploy/vllm/values-h100.yaml` is a values file for a single H100 with prefix caching and
chunked prefill on; `-f` it instead of the `--set` lines above.
`--set engine.mode=sglang` with `-f deploy/sglang/values-h100.yaml` installs the same
deployment on SGLang instead: the gateway, the lanes, the probes and the alerts do not
change, so everything in this runbook applies to either engine unless it names a flag.

Before shipping a change to either config file, validate it without a cluster:

```bash
turboserve gateway config-check --tenants configs/tenants.yaml --models configs/models.yaml
make k8s-lint          # helm lint + helm template | kubeconform + kustomize overlays
```

**Verify the install.** Readiness is the honest check: `/healthz` answers as soon as the
process is up, while `/readyz` is 503 until every served model has a healthy replica.

```bash
kubectl -n turboserve rollout status deploy/turboserve-gateway
kubectl -n turboserve port-forward svc/turboserve-gateway 8000:8000 &
curl -s localhost:8000/readyz | jq .
curl -s localhost:8000/v1/models -H "Authorization: Bearer $TURBOSERVE_KEY" | jq .
```

**Locally, with no cluster:**

```bash
turboserve serve --engine mock --no-require-auth          # nothing to install
turboserve serve --engine reference --model <path-or-id>  # in-process engine
docker compose up -d                                      # gateway + Prometheus + Grafana
```

---

## 2. Tenants and API keys

`configs/tenants.yaml` (a Secret in the cluster) holds **sha256 digests of keys, never the
keys**. A plaintext key in that file is rejected at load time rather than silently accepted.

### Add a tenant

```bash
KEY="sk-$(openssl rand -hex 24)"          # generate outside the repo; give it to the tenant once
turboserve gateway hash-key "$KEY"        # -> 64 hex characters
```

Append to the tenants file:

```yaml
- id: acme
  name: Acme Corp
  keys_sha256: ["<digest from hash-key>"]
  rpm: 600
  tpm: 200000
  max_concurrency: 16
  allowed_models: ["Qwen/Qwen2.5-7B-Instruct"]   # fnmatch patterns; empty means all
  adapters: {support: acme-support-r16}          # tenant-facing name -> adapter
  priority: 0
  weight: 1.0
```

Apply and restart (the file is read at startup):

```bash
kubectl -n turboserve create secret generic turboserve-tenants \
  --from-file=tenants.yaml=configs/tenants.yaml --dry-run=client -o yaml | kubectl apply -f -
kubectl -n turboserve rollout restart deploy/turboserve-gateway
```

### Rotate a key with no downtime

`keys_sha256` is a **list** precisely so a rotation is not a cutover:

1. Add the new digest alongside the old one. Apply, restart. Both keys now work.
2. Give the tenant the new key; confirm traffic has moved (the *Requests/s by tenant* panel
   will not show it — the label is the tenant, not the key, so confirm with the tenant or
   by removing the old digest in a canary window).
3. Remove the old digest. Apply, restart.

### Revoke immediately

Set `enabled: false` on the tenant (every request becomes 403) or delete the digest. Restart
the gateway; there is no key cache beyond the loaded file.

### Quotas

`rpm` and `tpm` are token buckets, `max_concurrency` is a non-queueing counter that returns
429 with a computed `Retry-After`. **Quota state is per process**: with *N* gateway replicas
the fleet-wide limit is *N* times the configured value. Divide when you set it, or run one
replica for a tenant whose limit must be exact.

---

## 3. LoRA adapters

Adapters are ordinary PEFT directories (`adapter_config.json` +
`adapter_model.safetensors`). The engine refuses DoRA, `modules_to_save`, biased adapters
and embedding LoRA by name rather than ignoring them, so a rejected adapter is a clear
error, not a silently wrong answer.

### Create a set for testing

```bash
turboserve lora make-adapters --model Qwen/Qwen2.5-0.5B-Instruct --n 16 --rank 8 --out adapters/
```

### Publish one

1. Put the directory under the adapters source (`engine.adapters.sync.sourceUrl`, synced by
   an init container into the adapters PVC), or into the PVC directly.
2. Add it to the engine's LoRA options — `EngineConfig.lora` for the reference engine,
   `engine.vllm.lora.modules` for vLLM, or `engine.sglang.lora.paths` for SGLang (one
   `--lora-paths` flag takes every `name=path` pair, and `maxLorasPerBatch` bounds how many
   distinct adapters one batch may mix).
3. Map the tenant-facing name to it in that tenant's `adapters:` block. The name a tenant
   sends in the `lora` field is *theirs*; the mapping is what stops one tenant naming
   another's adapter.
4. Restart the engine (adapters are registered at startup).

An unknown adapter name is a 403 at the gateway, before any engine work.

### Capacity

`max_loras` is the number of GPU slots, `max_lora_rank` their width. Slots are
`max_lora_rank` wide regardless of what they hold, so a pool sized for rank 64 costs the
same whether the adapters are rank 8 or rank 64. Keep the slot count at or above
`max_num_seqs`; below that the engine logs a warning at install and a step that needs more
distinct adapters than there are slots fails loudly rather than serving the base model.

---

## 4. Roll a canary

### What the gate does

The controller advances through `[1, 5, 25, 50, 100]` percent, holding at each step for
`step_hold_s` and checking four gates: minimum sample size (or a stall rollback), error
rate, absolute p95 TTFT, and p95 ratio against the stable lane. The policy is
`configs/canary.yaml`; the reasoning is
[ADR-0006](adr/ADR-0006-canary-slo-thresholds.md).

```bash
turboserve canary plan                      # print the policy and every gate
```

### Rehearse

```bash
turboserve canary run --version v0.2.0 --kube --dry-run
```

`--dry-run` still reads the cluster (`kubectl get`) and skips only the mutating and waiting
commands, so it shows the real object names and the real starting state.

### Ship

```bash
helm upgrade turboserve deploy/helm/turboserve \
  --namespace turboserve --reuse-values \
  --set canary.enabled=true --set canary.weight=0 \
  --set canary.image.tag=v0.2.0

turboserve canary run --version v0.2.0 --kube --out rollout.json
```

The command prints a decision table as it goes and writes the full report — every decision
with the lane summaries it was made from — to `rollout.json`. A `PROMOTED` outcome means the
candidate image is now the stable one; `ROLLED_BACK` means the weight is back at 0 and the
candidate is scaled down.

With Argo Rollouts installed, set `canary.argoRollouts.enabled=true` and the same command
drives `kubectl argo rollouts set weight / promote / abort` instead of patching Services.
The two are mutually exclusive and the chart refuses the combination.

### Multi-replica gateways

The in-process sliding window only sees one replica's traffic. With more than one gateway
pod, gate from Prometheus — that is what `--kube` does by default, using the queries in the
`prometheus:` block of `configs/canary.yaml`. If the gateway's metric or label names are
ever changed, that block must change with them.

---

## 5. Roll back

**Fastest, no tooling** — take the traffic away, then investigate:

```bash
kubectl -n turboserve annotate svc turboserve-gateway-canary \
  nginx.ingress.kubernetes.io/canary-weight=0 --overwrite
kubectl -n turboserve scale deploy/turboserve-gateway-canary --replicas=0
```

**Through the driver** (the same two operations, from the configured policy, so the object
names and the Argo/Services choice come from `configs/canary.yaml` rather than your memory):

```bash
turboserve canary abort --dry-run      # read the cluster, print what it would change
turboserve canary abort
```

`turboserve canary run` already aborts by itself when a gate fails; `canary abort` is for
when nobody is running it, or it is running somewhere you cannot reach.

**Undo the release entirely:**

```bash
helm rollback turboserve --namespace turboserve      # previous revision
helm history turboserve --namespace turboserve
```

A rollback is safe to do while requests are in flight: the gateway drains on SIGTERM
(`terminationGracePeriodSeconds` plus `preStopSleep` so the endpoint is removed before the
process stops), and a stream that dies mid-body is reported to the client as a terminating
error event — it is never retried onto another replica, because bytes have already been
sent.

---

## 6. Dashboards and alerts

The Grafana dashboard (uid `turboserve-gateway`, shipped as a ConfigMap) is laid out as
*Service level → Traffic and latency → Rollout lanes → Tenants → Backends*. Read it in that
order: the top row says whether anything is wrong, the rest says where.

| Alert | What it means | First move |
| --- | --- | --- |
| `TurboserveGatewayDown` / `TurboserveNoGatewayTargets` | Scrape target down, or none exists | `kubectl -n turboserve get pods`; check the ServiceMonitor's label selector |
| `TurboserveHighErrorRate` | Stable-lane error ratio above the SLO | §8, and check `Requests/s by status` for which status dominates |
| `TurboserveCanaryHighErrorRate` | The canary lane is failing | Roll back (§5); the controller should already have |
| `TurboserveTTFTSLOBreach` | Stable p95 TTFT above the objective | Check `Queue depth by model` and the engine's preemption counter |
| `TurboserveCanaryLatencyRegression` | Canary p95 TTFT / stable p95 above the ratio gate | Roll back; the candidate is slower, not the cluster |
| `TurboserveQueueDepthHigh` | Requests waiting per model | Scale the gateway (HPA is on CPU *and* this metric) or add engine capacity |
| `TurboserveTenantRateLimited` | One tenant is being throttled continuously | Either their quota is wrong or they are retrying a rejection; §2 |

Two label facts that make queries return empty when ignored:

- `lane` on a metric is the **backend's** lane from `models.yaml`, not the pod's. The
  ServiceMonitor relabels the pod's `turboserve.io/lane` to `pod_lane` to avoid the
  collision.
- The label sets differ: `turboserve_gateway_rate_limited_total` is `(tenant, limit)`;
  `turboserve_gateway_queue_depth` is `(model)`.

`turboserve_gateway_cost_usd_total` is **attribution using the configured prices in
`models.yaml`**, not billing. It is labelled as such on the dashboard.

---

## 7. Capacity planning

### How many KV blocks fit

```bash
turboserve engine kv-size --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 --block-size 16 --gpu-memory-utilization 0.90
```

It reads only `config.json` — no checkpoint download, no GPU needed — and prints the same
arithmetic the engine performs at startup: bytes per block, the activation headroom it
reserves, the resulting block count, and *why* that count was chosen (`explicit`,
`memory`, or `max_model_len`).

### The arithmetic, when you need it by hand

For one token, across all layers, K and V:

```
bytes/token = 2 (K and V) x num_layers x num_kv_heads x head_dim x bytes(dtype)
bytes/block = bytes/token x block_size
num_blocks  = floor( (utilization x total - weights - activations) / bytes/block )
```

`num_kv_heads`, not `num_attention_heads`: grouped-query attention is why a 7B model's KV
cache is a fraction of what the head count suggests. `KVCache.bytes_per_block` is this
number, and `ModelConfig.kv_bytes_per_token(dtype) * block_size` is asserted equal to it in
the tests.

### Choosing the knobs

- **`gpu_memory_utilization`** (default 0.9) absorbs allocator fragmentation as well as
  everything the profiler did not see. Raise it only with a measured margin.
- **`max_num_batched_tokens`** bounds a step's activation memory, which is subtracted from
  the KV budget. A very large budget on a nearly full device is better handled by setting
  `num_blocks` explicitly.
- **`block_size`** (default 16) trades internal fragmentation (up to `block_size - 1`
  wasted slots per sequence) against block-table length and prefix-cache granularity — only
  full blocks are cacheable, so a large block size caches less of a short shared prefix.
- **`max_model_len`** caps the pool at what a request could possibly use; it never enlarges
  it.

### When it is too small

Watch `num_preemptions` in `turboserve engine`'s stats or the engine's `/metrics`. A
non-zero and growing count means the pool cannot hold the working set: reduce
`max_num_seqs`, reduce `max_model_len`, or add GPU. Preemption is recompute
([ADR-0002](adr/ADR-0002-recompute-vs-swap-preemption.md)), so the cost shows up as extra
prefill work and a longer TTFT tail rather than as failures.

---

## 8. Triage

Statuses the gateway returns, and what each one means:

| Status | Cause | Where to look |
| --- | --- | --- |
| 401 | Missing or unknown key | Was the key rotated out (§2)? |
| 403 | Tenant disabled, model not in `allowed_models`, or unknown adapter name | The tenant's block in `tenants.yaml` |
| 404 | Model not served by this gateway | `models.yaml`, and `GET /v1/models` |
| 429 | `rpm`, `tpm` or `max_concurrency` | `Rate-limited requests/s by tenant`; remember quotas are per replica |
| 502 | Backend failed in a way the router could not classify | Backend logs; `Backend health` panel |
| 503 | No healthy replica, or retries exhausted | `Backend health`; the router ejects a replica on its first transport failure and believes that for `health_ttl_s` |
| 504 | Backend timeout | Engine saturation; check queue depth and preemptions |

**A fleet that empties itself.** The router ejects a failing replica for the health TTL. A
uniform per-replica failure rate can therefore take *every* replica out at once if there is
no spare — three replicas absorb a fault that two do not. This is routing policy, not a
bug, and it is why the chart's default is three gateway replicas with a PodDisruptionBudget.

**Reproducing a failure deliberately.** The chaos harness runs the real router in front of
breakable replicas:

```bash
turboserve chaos plan --faults "kill:every=10s"        # what the schedule expands to
turboserve chaos run --replicas 3 --duration 60 --rps 20 --faults "kill:every=10s"
```

It writes an ordinary result file, so the error rate, the retry count and the recovery times
are rendered by the same report code as every benchmark. See
[`canary-and-chaos.md`](canary-and-chaos.md#part-ii--chaos).

**The in-cluster equivalent**, including the pod-deletion loop and the error-rate
assertion, is `deploy/kind/e2e.sh`.
