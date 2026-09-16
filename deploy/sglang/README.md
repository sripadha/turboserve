# Running the gateway in front of real SGLang

SGLang is the second production engine this repository deploys and measures, next to vLLM.
The reference engine in `src/turboserve/engine/` exists to make the serving techniques
legible; SGLang and vLLM are what you would actually run a fleet on, and the gateway treats
all three as first-class backends so that the same requests, the same tenants and the same
metrics can be pointed at any of them.

This directory is the SGLang side of that, and it is deliberately the mirror image of
[`../vllm/`](../vllm/README.md): the flags, a values file for the Helm chart, and a
launcher for hosts where Kubernetes is not available.

| File | What it is |
| --- | --- |
| `values-h100.yaml` | Helm values for one H100 80GB node, with the capacity arithmetic written out |
| `launch.sh` | Start SGLang natively with the same flags the chart renders |
| this file | What each flag buys, and how it differs from the vLLM flag of the same name |

## Two topologies

**Kubernetes.** `engine.mode: sglang` renders a separate SGLang Deployment plus a ClusterIP
Service — the same objects `engine.mode: vllm` renders, with a different image and a
different argv — and the gateway is started with `--engine
http://<release>-engine:30000/v1`. The engine Service is deliberately cluster-internal:
SGLang has no concept of a tenant, so anything that can reach it directly bypasses
authentication, quotas and per-tenant accounting. The chart's NetworkPolicy makes that
explicit by allowing ingress to the engine only from the gateway pods and the monitoring
namespace.

```
client ──► Ingress ──► gateway Service ──► gateway pods ──► engine Service ──► SGLang pods (GPU)
                                            │
                                            └── auth, quotas, routing, metrics, canary lane
```

Install it with:

```bash
helm upgrade --install turboserve deploy/helm/turboserve \
  -f deploy/sglang/values-h100.yaml \
  --namespace turboserve --create-namespace
```

The image is pinned to `lmsysorg/sglang:v0.5.3`, the official image, and pinned rather than
floating for the same reason the vLLM tag is: a minor release can change scheduling and make
a comparison against an earlier run meaningless.

**A single GPU box, or a vast.ai instance.** A vast.ai instance *is* a container and there
is no nested Docker inside one, so neither the Helm chart nor `docker-compose.yml` can be
used there. SGLang is installed into the instance and started with `launch.sh`, and the
gateway runs next to it as an ordinary process:

```bash
deploy/sglang/launch.sh &                                  # :30000, the OpenAI API
turboserve gateway serve --engine http://127.0.0.1:30000/v1 --model Qwen/Qwen2.5-7B-Instruct
```

This repository ships **no `sglang` extra**. `vllm` is an optional extra of the project
because the reference engine's comparison arm has always been vLLM, and adding a second
heavyweight resolution to `uv.lock` would change the environment every CPU test run
resolves. SGLang is installed into the measurement host beside the project environment
(`uv pip install "sglang[all]"` inside the instance, or rent the instance from the pinned
`lmsysorg/sglang:v0.5.3` image), and the gateway and the benchmark harness only ever need
its URL. See [`../../docs/vastai.md`](../../docs/vastai.md).

## The flags that matter

Each of these is a value in `values-h100.yaml` and an environment variable in `launch.sh`;
the chart renders them into the engine container's `args`. Where a flag is the counterpart
of a vLLM flag with a different name, the vLLM name is given, because the two servers are
measured side by side and a reader should be able to line the two command lines up.

### RadixAttention prefix caching (on by default, `--disable-radix-cache` turns it off)

SGLang keeps the KV cache of every prompt in a radix tree keyed by the token sequence, so a
new request that starts with tokens another request already computed adopts those pages
instead of recomputing them. This is the counterpart of vLLM's `--enable-prefix-caching`
with the polarity reversed: it is **on unless you disable it**, which is why
`enablePrefixCaching: false` renders a flag and `true` renders nothing.

Multi-tenant traffic is full of shared spans — a system prompt, a few-shot preamble, a long
document several questions are asked about — and with the tree in place that span is
prefilled once per node instead of once per request. It shows up as lower time-to-first-token
and in nothing else: decode is unaffected, because the shared work is all prefill. The tree
is per node, so it interacts with routing exactly as vLLM's cache does — a load balancer
that spreads a tenant's requests evenly across pods gets fewer hits than one that keeps a
tenant's traffic on one pod.

The `prefix-cache` benchmark scenario needs a server with it **off** as its control arm,
which is what `ENABLE_PREFIX_CACHING=0 deploy/sglang/launch.sh` starts.

### `--chunked-prefill-size`

The token budget one scheduler step may spend on prefill, and therefore the mechanism that
keeps a long prompt from occupying a whole step: prefill is split into chunks that share
each step with the running decodes. vLLM expresses the same thing as
`--enable-chunked-prefill` plus `--max-num-batched-tokens`; SGLang has the single number,
and chunking is what having one implies. Larger keeps the GPU busier on prefill, smaller
keeps decode smoother.

### `--speculative-algorithm` (`EAGLE`, `NEXTN`) and `--speculative-draft-model-path`

A draft proposes `k` tokens; the target verifies all `k+1` positions in one forward pass and
accepts the longest correct prefix. Decode is memory-bandwidth bound — each step re-reads
the entire weight matrix to produce one token — so verifying several tokens in the pass that
would have produced one is nearly free, and the win is proportional to how often the draft
is right. EAGLE drafts from a small head trained on the target's hidden states rather than
from an independent model, which is why its acceptance is usually higher than a separate
draft checkpoint's at the same `k`; NEXTN uses the multi-token-prediction head a checkpoint
was trained with. The costs are the same as vLLM's and worth stating: the draft's weights
come out of the same 80 GB as the KV cache, and under heavy batching the target's forward
pass is already compute-saturated, so the technique helps least exactly when the server is
busiest.

### `--lora-paths` and `--max-loras-per-batch`

One set of base weights, many per-tenant adapters, selected per request by the `model`
field. `--lora-paths` takes every `name=path` pair in a single flag (unlike vLLM's repeated
`--lora-modules`), and `--max-loras-per-batch` is how many *distinct* adapters one batch may
mix — the knob that decides whether requests for many different tenants can share a step or
have to be split across steps. The gateway's tenant configuration maps a tenant's adapter
name to the `model` string SGLang expects, which is what lets a tenant ask for its own
fine-tune without knowing anything about the base model.

### `--quantization fp8` and `--kv-cache-dtype fp8_e5m2`

The FP8 launch option, and the one place the two engines' argv genuinely differ.
`--quantization fp8` stores the linear weights as W8A8-FP8 and runs the matmuls on the FP8
tensor cores Hopper has; `--kv-cache-dtype` names the KV format outright, and `fp8_e5m2` is
the scale-free one this deployment uses, where vLLM's `fp8` means E4M3 with per-tensor
scales. E5M2 spends a bit of mantissa on exponent range, so it needs no calibration pass —
the reason it is the default here rather than the more accurate format.

Both halve bytes, which is what makes it a capacity decision: the 7B checkpoint's weights go
from about 15 GiB to about 7.6, and a KV token from 57344 bytes to 28672, so
`--mem-fraction-static 0.90` covers roughly twice the tokens it did before.

```bash
QUANT=fp8 deploy/sglang/launch.sh                                    # quantize on the fly
QUANT=fp8 FP8_MODEL=Qwen/Qwen2.5-7B-Instruct-FP8 deploy/sglang/launch.sh   # pre-quantized
```

A pre-quantized checkpoint declares its scheme in its own `config.json`, so the launcher
passes only `--kv-cache-dtype` for it. Unlike vLLM, SGLang has no `--served-model-name`, so
swapping the checkpoint also swaps the `model` string clients send — which is why the
on-the-fly path is the one the benchmark arms use. In the chart both cases are
`engine.quantization: fp8`, with `engine.quantizedCheckpoint: true` selecting the second.

What it is worth on this hardware is a rendered row: the `SGLang (fp8)` arm under
`naive_vs_cb` in [docs/results.md](../../docs/results.md). Accuracy is not measured here and
no claim is made about it.

### `--mem-fraction-static`

The fraction of the GPU SGLang is allowed to claim for weights and KV cache; whatever is
left absorbs activations. It is the counterpart of vLLM's `--gpu-memory-utilization`, and it
sets concurrency the same way: the KV cache is what decides how many sequences can be
resident. Raising it buys more concurrent sequences and narrows the headroom that absorbs
activation spikes during a long prefill, which is why the values file stops at 0.90.

## What the server will tell you about itself

Beside the OpenAI-compatible routes under `/v1`, SGLang serves three native endpoints this
repository uses:

| Endpoint | Used for |
| --- | --- |
| `GET /health` | The chart's startup and readiness probes, and `OpenAICompatBackend.health()` |
| `GET /version` | `{"version": ...}`, on the builds that route it — this is vLLM's endpoint, and SGLang reports its version in `/get_server_info` regardless |
| `GET /get_server_info` | The version, model path, dtype, context length and scheduler settings the server is running with |

`OpenAICompatBackend.server_info()` reads the last two when they are reachable and records
what they said, in the result file of every benchmark arm this server served; nothing else in the gateway distinguishes SGLang from vLLM, because nothing
else has to — the request body, the SSE framing, `stream_options.include_usage` and
`ignore_eos` are identical on both.

## Metrics

SGLang exports its own `sglang:*` series, but only when it is started with
`--enable-metrics`, which neither the chart nor `launch.sh` passes. Were it on, they would
be scraped as a separate Prometheus job, as vLLM's are, so that an engine restart does not
fire the gateway's availability alerts — the gateway reports that as backend errors of its
own, which is the signal a client actually experiences. The Grafana dashboard in `deploy/grafana/dashboards/turboserve.json` is built
entirely on gateway metrics for the same reason: it is a view of what tenants see, not of
what the engine is doing internally.

## What has and has not been run here

The manifests in this repository are validated on every CI run with `helm lint`,
`helm template | kubeconform -strict`, and an end-to-end kind job that installs the chart and
runs load against it while deleting pods. That job uses `engine.mode: mock`, because
GitHub's hosted runners have no GPU.

The SGLang path — this directory — has therefore been validated as manifests and as a
command line, not executed: SGLang needs an Ampere-or-newer GPU, and no benchmark or GPU run
is published from anywhere but the measurement host. `launch.sh` prints the exact command
line it will run and exits when `DRY_RUN=1`, which is how the flag set is checked without a
GPU.
