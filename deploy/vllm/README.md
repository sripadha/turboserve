# Running the gateway in front of real vLLM

The reference engine in `src/turboserve/engine/` exists to make the serving techniques
legible: you can read the scheduler, the block allocator and the rejection-sampling
verifier in an afternoon. It is not the thing you would run a fleet on. vLLM is, and the
gateway treats both as first-class backends so that the same requests, the same tenants and
the same metrics can be pointed at either one.

This directory is the vLLM side of that: the flags, a values file for the Helm chart, and a
launcher for hosts where Kubernetes is not available.

| File | What it is |
| --- | --- |
| `values-h100.yaml` | Helm values for one H100 80GB node, with the capacity arithmetic written out |
| `launch.sh` | Start vLLM natively with the same flags the chart renders |
| this file | What each flag buys, and the two topologies |

## Two topologies

**Kubernetes.** `engine.mode: vllm` renders a separate vLLM Deployment plus a ClusterIP
Service, and the gateway is started with `--engine vllm --engine-url
http://<release>-engine:8000/v1`. The engine Service is deliberately cluster-internal: vLLM
has no concept of a tenant, so anything that can reach it directly bypasses authentication,
quotas and per-tenant accounting. The chart's NetworkPolicy makes that explicit by allowing
ingress to the engine only from the gateway pods and the monitoring namespace.

```
client ──► Ingress ──► gateway Service ──► gateway pods ──► engine Service ──► vLLM pods (GPU)
                                            │
                                            └── auth, quotas, routing, metrics, canary lane
```

Install it with:

```bash
helm upgrade --install turboserve deploy/helm/turboserve \
  -f deploy/vllm/values-h100.yaml \
  --namespace turboserve --create-namespace
```

**A single GPU box, or a vast.ai instance.** A vast.ai instance *is* a container and there
is no nested Docker inside one, so neither the Helm chart nor `docker-compose.yml` can be
used there. vLLM is pip-installed into the instance and started with `launch.sh`, and the
gateway runs next to it as an ordinary process:

```bash
uv sync --extra vllm
deploy/vllm/launch.sh &                                   # :8000, the OpenAI API
turboserve gateway serve --engine vllm --engine-url http://127.0.0.1:8000/v1
```

`scripts/vastai/` automates exactly this; see `docs/vastai.md`. On a machine that *does*
have Docker, `docker-compose.yml` at the repository root brings up the same pair plus
Prometheus and Grafana.

## The flags that matter

Each of these is a value in `values-h100.yaml` and an environment variable in `launch.sh`;
the chart renders them into the engine container's `args`.

### `--enable-prefix-caching`

vLLM hashes each block of prompt tokens and reuses cached KV blocks when a new request
starts with the same content. Multi-tenant traffic is full of shared spans — a system
prompt, a few-shot preamble, a long document several questions are asked about — and with
caching on, that span is prefilled once per node instead of once per request. It shows up
as lower time-to-first-token and in nothing else: decode is unaffected, because the shared
work is all prefill. The cache is per node, so it interacts with routing — a load balancer
that spreads a tenant's requests evenly across pods gets fewer hits than one that keeps a
tenant's traffic on one pod.

### `--enable-chunked-prefill`

Without it, prefilling one 8k-token prompt occupies a whole scheduler step and every
sequence already decoding waits for it, which is visible to clients as an inter-token
latency spike. With it, prefill is split into chunks that share each step's token budget
with the running decodes. `maxNumBatchedTokens` is the budget being shared: larger keeps
the GPU busier on prefill, smaller keeps decode smoother.

### `--speculative-model` and `--num-speculative-tokens`

A small draft model proposes `k` tokens; the target model verifies all `k+1` positions in
one forward pass and accepts the longest correct prefix. Decode is memory-bandwidth bound —
each step re-reads the entire weight matrix to produce one token — so verifying several
tokens in the pass that would have produced one is nearly free, and the win is proportional
to how often the draft is right. That is why the draft has to come from the same family and
share the target's tokenizer. The costs are real and worth stating: the draft's weights come
out of the same 80 GB as the KV cache, and under heavy batching the target's forward pass is
already compute-saturated, so the technique helps least exactly when the server is busiest.

### `--enable-lora`, `--max-loras`, `--max-lora-rank`

One set of base weights, many per-tenant adapters, selected per request by the `model`
field. `--max-loras` is how many adapters are resident on the GPU at once, not how many may
be registered: the rest sit in host memory and are swapped in on demand, so a request for a
cold adapter pays a transfer. `--max-lora-rank` must be at least the largest rank you
intend to serve, and it sizes the preallocated adapter buffers, so setting it far above what
you use wastes memory. The gateway's tenant configuration maps a tenant's adapter name to
the `model` string vLLM expects, which is what lets a tenant ask for its own fine-tune
without knowing anything about the base model.

### `--gpu-memory-utilization`

The fraction of the GPU vLLM is allowed to claim; whatever is left after weights and
activations becomes KV cache, and the KV cache is what sets concurrency. Raising it buys
more concurrent sequences and narrows the headroom that absorbs activation spikes during a
long prefill, which is why the values file stops at 0.90. `values-h100.yaml` shows the
arithmetic from weight bytes and per-token KV bytes to the number of concurrent sequences
the setting implies.

## Metrics

vLLM exports its own `vllm:*` series on the same port as its API. They are scraped as a
separate Prometheus job (`deploy/prometheus/prometheus.yml`) so that an engine restart does
not fire the gateway's availability alerts — the gateway reports that as backend errors of
its own, which is the signal a client actually experiences. The Grafana dashboard in
`deploy/grafana/dashboards/turboserve.json` is built entirely on gateway metrics for the
same reason: it is a view of what tenants see, not of what the engine is doing internally.

## What has and has not been run here

The manifests in this repository are validated on every CI run with `helm lint`,
`helm template | kubeconform -strict`, and an end-to-end kind job that installs the chart
and runs load against it while deleting pods. That job uses `engine.mode: mock`, because
GitHub's hosted runners have no GPU.

The vLLM path — this directory — has therefore been validated as manifests and as a
command line, not executed: the development machine for this repository has a 6 GB
Turing GPU and no Docker, and per the project's own rules no benchmark or GPU run happens
there. `launch.sh` prints the exact command line it will run and exits when `DRY_RUN=1`,
which is how the flag set is checked without a GPU.
