# Running the gateway in front of real vLLM

The reference engine in `src/turboserve/engine/` exists to make the serving techniques
legible: you can read the scheduler, the block allocator and the rejection-sampling
verifier in an afternoon. It is not the thing you would run a fleet on. vLLM is, and so is
SGLang; the gateway treats all three as first-class backends so that the same requests, the
same tenants and the same metrics can be pointed at any of them.

This directory is the vLLM side of that: the flags, a values file for the Helm chart, and a
launcher for hosts where Kubernetes is not available. [`../sglang/`](../sglang/README.md) is
its mirror image, engine for engine and flag for flag; a change made here usually has a
counterpart there.

| File | What it is |
| --- | --- |
| `values-h100.yaml` | Helm values for one H100 80GB node, with the capacity arithmetic written out |
| `launch.sh` | Start vLLM natively with the same flags the chart renders |
| this file | What each flag buys, and the two topologies |

## Two topologies

**Kubernetes.** `engine.mode: vllm` renders a separate vLLM Deployment plus a ClusterIP
Service, and the gateway is started with `--engine
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
turboserve gateway serve --engine http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
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

### `--quantization fp8` and `--kv-cache-dtype fp8`

FP8 is the launch option this deployment offers on Hopper, and it is two decisions that
happen to share a flag prefix. `--quantization fp8` stores the linear weights as W8A8-FP8
and runs the matmuls on the FP8 tensor cores an H100 has and an A100 does not;
`--kv-cache-dtype fp8` stores the KV cache in E4M3 with per-tensor scales instead of bf16.
Both halve bytes: the 7B checkpoint's weights go from about 15 GiB to about 7.6, and a KV
token from 57344 bytes to 28672. Decode is memory-bandwidth bound — a step reads every
weight to produce one token — and the KV cache is what decides how many sequences fit, so
the two together move both of the quantities `values-h100.yaml`'s arithmetic is written in.

It is a *launch* option, not a request parameter: the conversion happens once, while the
weights load, and every request afterwards is served from the quantized copy. Two ways in,
and they are not the same command line:

```bash
QUANT=fp8 deploy/vllm/launch.sh                                    # quantize on the fly
QUANT=fp8 FP8_MODEL=Qwen/Qwen2.5-7B-Instruct-FP8 deploy/vllm/launch.sh   # pre-quantized
```

The first converts the bf16 checkpoint at load time, which costs a little startup and no
extra download. The second serves a repository that already ships FP8 weights; such a
checkpoint declares its own scheme in `config.json` and vLLM refuses a `--quantization`
that disagrees with it, so the launcher passes only `--kv-cache-dtype` in that case and
keeps `--served-model-name` on the bf16 name so clients address one model string either
way. In the chart the same two cases are `engine.quantization: fp8` and, for the second,
`engine.quantizedCheckpoint: true` with `engine.model` pointing at the `-FP8` repository.

What FP8 costs is accuracy, and this repository does not measure accuracy, so the honest
statement is the scope: the arms under `naive_vs_cb` in [docs/results.md](../../docs/results.md)
say what it does to throughput, latency and cost per million tokens on this hardware, and
nothing here claims it leaves the output distribution unchanged the way rejection-sampling
verification does for speculative decoding. E4M3 with per-tensor scales is the conservative
choice for the KV cache for exactly that reason; SGLang's counterpart flag takes
`fp8_e5m2`, which trades mantissa for exponent range and needs no calibration.

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
command line, not executed: vLLM needs an Ampere-or-newer GPU, and no benchmark or GPU run
is published from anywhere but the measurement host. `launch.sh` prints the exact command
line it will run and exits when `DRY_RUN=1`, which is how the flag set is checked without a
GPU.
