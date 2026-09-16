# Gateway

`src/turboserve/gateway/` is the HTTP front door: an OpenAI-compatible API that authenticates
tenants, enforces their quotas, picks a backend replica, streams tokens back and accounts for
what was used. It is the only part of the system that the from-scratch reference engine and a
production fleet — vLLM or SGLang — sit behind, which is the point: auth, routing, canary
weighting, chaos and accounting are then identical for all of them, and a comparison between
two engines is a comparison of engines rather than of serving stacks.

The gateway is engine-agnostic, and that is a property worth stating rather than assuming.
Adding SGLang as a second production backend changed no code on the request path: it is
served by the same `OpenAICompatBackend` as vLLM, over the same protocol, and the router,
the quotas, the metrics and the canary gate cannot tell the two apart. The one thing that
was added is the ability to *record* which of them answered — see
[Backends](#backends).

## Purpose

- Speak the OpenAI wire protocol exactly, so the `openai` client, the benchmark harness and
  any existing tooling work unchanged.
- Attribute everything to a tenant: quotas, metrics, logs, cost.
- Route a model name to a *pool* of replicas split into `stable` and `canary` lanes, honouring
  the canary controller's weight and each replica's health.
- Retry a failed request only while retrying is still honest — before the first token.

## Request path

```mermaid
flowchart LR
  C[client] --> A["POST /v1/chat/completions"]
  A --> AU["auth.py<br/>bearer key to tenant"]
  AU -->|401 / 403| C
  AU --> MD["model allow-list<br/>adapter resolution"]
  MD -->|403 / 404| C
  MD --> CT["chat_template.py<br/>messages to prompt"]
  CT --> LM["limits.py<br/>rpm / tpm buckets"]
  LM -->|"429 + Retry-After"| C
  LM --> R["router.py<br/>lane, replica, retries"]
  R -->|"429 concurrency<br/>503 / 502 / 504"| C
  R --> B["backend<br/>local | openai | mock"]
  B --> S["SSE chunks + [DONE]"]
  S --> U["usage.py + metrics.py"]
  S --> C
```

Every gate refuses with a status the client can act on, and the cheap refusals come first: a
request from an unknown key never touches a token bucket, and a request over quota never
touches a GPU. The one expensive gate — routing — is entered *before* the response is opened,
because the router may still switch replicas at that point and the status code is still
available.

## Modules

| File | What it owns |
| --- | --- |
| `app.py` | `create_app()`, the routes, the exception handlers, and the `turboserve gateway` CLI |
| `auth.py` | `hash_api_key`, `Authenticator`, `Principal`; 401 vs 403 |
| `tenants.py` | `Tenant`, `TenantRegistry`; the parsed `configs/tenants.yaml` |
| `limits.py` | `TokenBucket`, `TenantLimiter`, `LimiterRegistry`, `RateLimitExceeded` |
| `router.py` | `Router`, `ModelPool`, `BackendEntry`, `ModelsFile`, `build_router` |
| `openai_types.py` | Every request/response/chunk model, plus the finish-reason mapping |
| `chat_template.py` | `ChatTemplate`, `ChatTemplateCache`, the fallback template |
| `metrics.py` | `GatewayMetrics` — one private Prometheus registry per app |
| `usage.py` | `UsageAccumulator`, `UsageRecord`, `UsageTracker`, `PriceTable` |
| `tracing.py` | `Tracing`, `GenerationSpan`, `configure_tracing`, `instrument_app`, `inject_trace_context` |
| `backends/openai_compat.py` | `OpenAICompatBackend` — streams from vLLM/SGLang/TGI over httpx |
| `backends/mock.py` | `MockBackend`, `build_mock_app`, `serve_mock` |

`backends/protocol.py` and `backends/__init__.py` (the `Backend` contract and the name
registry) are shared with the engine and documented in [`contracts.md`](contracts.md).

## Routes

| Route | Behaviour |
| --- | --- |
| `POST /v1/completions` | Text completion; streaming or buffered. One prompt per request. |
| `POST /v1/chat/completions` | Chat completion; messages rendered with the model's template. |
| `GET /v1/models` | Models this tenant may address — filtered, never the whole fleet. |
| `GET /healthz` | Liveness. Never consults a backend. |
| `GET /readyz` | Readiness: every served model has at least one healthy replica, else 503. |
| `GET /metrics` | Prometheus exposition of this app's registry. |

### Status codes

| Status | Cause |
| --- | --- |
| 400 | Malformed body, or a field this gateway does not implement (see below) |
| 401 | Missing, malformed or unknown API key (`WWW-Authenticate: Bearer`) |
| 403 | Tenant disabled, model outside its allow-list, adapter not its own |
| 404 | No pool serves the requested model |
| 429 | `rpm`, `tpm` or `max_concurrency` exceeded, with `Retry-After` |
| 502 | A backend failed in a way the router could not classify, or mid-stream on a buffered request |
| 503 | No healthy replica, or every attempt failed with a retryable error |
| 504 | The backend accepted the connection and then produced nothing in time |

Bodies are OpenAI-shaped — `{"error": {"message", "type", "param", "code"}}` — so a client
unwraps them into its own exception types with no turboserve-specific code.

`n > 1`, `best_of`, `echo`, `suffix`, `tools`, `functions`, `response_format` and `logit_bias`
are **refused with 400** rather than ignored. Silently dropping a field that changes what the
answer means is the dishonest option: a client that asked for a tool call and received prose
has been misled. Unknown fields with no semantic consequence (`frequency_penalty`, `user`,
client telemetry) are accepted and ignored, which is what keeps real clients working.

### Streaming format

Streaming responses are `text/event-stream` with exactly OpenAI's framing:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null,"logprobs":null}],"usage":null}

data: {"id":"chatcmpl-…",…,"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null,"logprobs":null}],"usage":null}

data: {"id":"chatcmpl-…",…,"choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}],"usage":null}

data: [DONE]

```

Details that clients depend on and that `tests/unit/test_gateway_app.py` asserts with a
deliberately strict parser:

- the separator is `\n\n`, not CRLF (`EventSourceResponse(..., sep="\n")`);
- one id and one `created` for every chunk of a stream;
- the first chat chunk carries `delta.role` only, content chunks carry `delta.content` only —
  clients concatenate `delta.content`, so repeating the role would corrupt the message;
- `finish_reason` is present as `null` on non-final chunks, because clients read the key as
  required;
- the stream always ends with `data: [DONE]\n\n`;
- `usage` arrives in one extra final chunk, only when the request set
  `stream_options.include_usage` — the same opt-in OpenAI uses.

A failure *after* the first chunk cannot be a status code any more, so it travels as a final
`data: {"error": …}` chunk followed by `[DONE]`, and the request is counted as an error.

`FinishReason.ABORT` is reported to clients as `"stop"`: the field is validated against a
closed set by strict clients, and an unknown value makes them raise instead of surfacing the
partial completion they already hold. Aborts remain visible in `requests_total{status="error"}`
and in the usage record.

## Tenants and authentication

`configs/tenants.yaml` holds the directory. Keys are stored **only** as SHA-256 hex digests,
because that file ends up in git, in a ConfigMap and in a Helm values file:

```bash
turboserve gateway hash-key 'sk-your-key-here'
printf %s 'sk-your-key-here' | sha256sum          # identical digest
```

A `Tenant` carries `rpm`, `tpm`, `max_concurrency`, `allowed_models` (fnmatch patterns; empty
means all), `adapters` (tenant-visible name → what the backend calls it), `priority` (who
the engine preempts first when the KV pool is full -- not an admission order) and `weight`
(the tenant's share under the `tenant_fair` admission policy). `enabled: false` authenticates the key and then refuses with 403, which is how an
account is suspended without deleting its credentials.

Two rules worth restating:

- **401 and 403 are kept apart.** A spike of 401s means a rotated key was not rolled out; a
  spike of 403s means a model or adapter left an allow-list. A client that conflates them
  loops.
- **Adapter names are tenant-private.** An unknown adapter is 403, not 404, so a caller cannot
  enumerate another tenant's adapters by probing names.

`GatewayOptions(require_auth=False)` attributes every request to the `default` tenant. That
mode exists for closed deployments — the kind end-to-end test and the chaos harness — where
the only client is the load generator.

## Quotas

`rpm` and `tpm` are token buckets: the per-minute figure is both the refill rate and the
default burst. `max_concurrency` is a counter, not an `asyncio.Semaphore`, because the gateway
*rejects* rather than queues — an immediate 429 is actionable, an unbounded queue of
connections that all time out together is not.

`tpm` is charged twice: the prompt at admission, the completion when it finishes. Completion
length is unknowable at admission, so the bucket is allowed to go negative and the deficit
pushes the tenant's next `Retry-After` out. Reserving `max_tokens` up front would instead
penalise every request for the longest one it might have been.

`Retry-After` is computed from the bucket's refill rate, rounded up to whole seconds (RFC 9110
allows nothing finer) with a floor of one second. For the concurrency limit there is no refill
rate to derive a wait from — it clears when another of that tenant's requests finishes — so a
fixed one-second hint is sent.

## Routing, lanes and retries

```mermaid
stateDiagram-v2
  [*] --> Select
  Select --> Attempt: healthy replica in the drawn lane
  Select --> Fail503: no healthy replica
  Attempt --> Streaming: first event yielded
  Attempt --> Select: retryable error, attempts left
  Attempt --> Fail: non-retryable error, or attempts exhausted
  Streaming --> Done: terminating event
  Streaming --> InBandError: error after the first event
  InBandError --> Done
```

The lane is drawn per request from `CanaryController.canary_weight` (a percentage), so a
rollback takes effect on the next request rather than the next connection. A lane with no
healthy replica is never drawn, so a canary whose only pod is down stops receiving traffic
instead of black-holing its share. Within a lane, replicas are chosen in proportion to their
configured `weight`.

Health is cached per replica for `health_ttl_s` (5 s by default) and refreshed before a
selection when stale. A request that fails on a replica marks it unhealthy immediately — the
strongest signal available is a request that just failed — *unless* the failure was a
`NonRetryableBackendError`, because a rejected request says nothing about the replica and one
malformed client could otherwise empty a pool.

**Retry is legal only before the first event.** After it the client holds part of a completion
and re-running it elsewhere would duplicate or contradict what they have, so the failure is
converted into a terminating error event. The route pulls the first event out of the router
*before* constructing the response, so "before the first event" and "before the first byte"
are the same moment.

The router only depends on the canary controller through a small `CanaryWeightSource` protocol
(`canary_weight` plus `observe(lane, ok, ttft_ms, e2e_ms, now)`), so the gateway does not
import the progressive-delivery machinery and a deployment without a canary passes `None`.
`tests/unit/test_gateway_router.py` imports the real controller and asserts the structural
match, so the two halves of that contract cannot drift apart unnoticed.

Finished requests are fed back through `observe`, which is what closes the SLO loop. A
request counts as a failure there whether the backend *raised* after the first byte or
*yielded* a terminating error event -- the second is the documented way for a backend to
fail mid-stream, so treating a normally-ending stream as a success would make the gate blind
to exactly the failures the client already started reading.

## Backends

`configs/models.yaml` maps each model to its pool:

```yaml
models:
  - name: Qwen/Qwen2.5-7B-Instruct
    backends:
      - {name: vllm-stable-a,   backend: openai, lane: stable, options: {base_url: "http://vllm-a:8000/v1"}}
      - {name: sglang-stable-a, backend: openai, lane: stable, options: {base_url: "http://sglang-a:30000/v1"}}
      - {name: vllm-canary,     backend: openai, lane: canary, options: {base_url: "http://vllm-c:8000/v1"}}
    price: {input_per_1m_usd: 0.20, output_per_1m_usd: 0.60}
```

The second replica in that pool runs a different engine, and nothing in the file says so
beyond its name and port: there is no per-engine backend type, because there is no per-engine
behaviour to implement. That is also how an engine migration is done here — put SGLang on the
canary lane and let `turboserve canary run` gate it on the same SLO a new build is gated on.

`backend:` names a class in the registry (`turboserve.gateway.backends`), and `options:` is its
constructor's keyword arguments, so a deployment adds a backend type by installing a module
that registers itself.

**`OpenAICompatBackend`** streams from any OpenAI-compatible server with httpx. Three details
of that protocol shape it:

- What the `openai` client calls `extra_body` is, on the wire, additional top-level members of
  the request object — that is how vLLM and SGLang receive `top_k`, `repetition_penalty` and
  `ignore_eos`. They are sent only when they differ from their neutral value, so a stricter
  upstream still works for ordinary requests.
- A LoRA adapter is addressed through the `model` field: both engines serve each loaded
  adapter under its adapter name (vLLM's `--lora-modules`, SGLang's `--lora-paths`), so
  `adapter_models` maps the tenant's name to that string.
- `stream_options.include_usage` is always requested, because it is the only way to learn the
  upstream's own tokenisation of the prompt instead of guessing at it.

Those three are the whole protocol surface, and they are the same on both production engines,
which is why `generate()` contains no branch on which server it is talking to. What is *not*
the same is what each server will say about itself, and `server_info()` is the whole of the
difference:

| Endpoint | vLLM | SGLang | Used for |
| --- | --- | --- | --- |
| `GET /health` | yes | yes | `health()`, and the chart's startup/readiness probes |
| `GET /version` | yes | some builds | `{"version": ...}` |
| `GET /get_server_info` | no | yes | the running server's version, model path, dtype, context length, scheduler settings |

`server_info()` asks for both, takes the version from whichever of them answered (`/version`
first, the second document's own `version` field otherwise, because settings recorded
without a version are settings nobody can look up), keeps a documented whitelist of scalar
settings out of the second, and returns `{}` when neither answers — it never raises and is never on a request
path. A server that answers `/get_server_info` is an SGLang server, which is how a benchmark
result file can record what a client otherwise cannot see: whether the radix cache was
disabled, what the context window was, how many adapters a batch could mix. Those flags are
most of what a number means, and the benchmark scenarios copy the block into the run's
`config["engine"]["server"]`.

Transport failures map onto the router's vocabulary: connect errors → `BackendUnavailableError`,
timeouts → `BackendTimeoutError`, 429/503 → `BackendOverloadedError`, 404 → `ModelNotFoundError`,
other 4xx → `BackendRequestError`. A failure after the first token is yielded as a terminating
error event rather than raised, per the `Backend` contract.

**`MockBackend`** behaves like a model server without a model. It exists because every
interesting behaviour in a gateway is a *timing* behaviour: a stub that returns instantly
would make routing, retry, canary and chaos tests meaningless, and a real engine would make
them slow and hardware-dependent. It is configurable along exactly the axes those tests need —
`ttft_ms`, `itl_ms`, `jitter`, `error_probability` (failure before the first token, the only
kind a router may retry), `drop_probability` (failure after tokens were sent, never
retryable), `healthy`, `models`, `adapters` — and every draw comes from an RNG seeded by the
request id, so a given request always gets the same latency, text and fate and a chaos run can
be replayed.

`MockConfig` is mutable (`validate_assignment=True`), which is how the chaos harness applies a
fault schedule to a live backend without restarting anything.

`build_mock_app()` / `serve_mock(port, **cfg)` put an OpenAI-compatible HTTP server in front of
one `MockBackend`. It is *the real gateway app* with authentication disabled, not a second
implementation, so a fault injected there travels the same auth, routing, accounting and SSE
code a production request does. The chaos harness launches it as a subprocess
(`python -m turboserve.gateway.backends.mock --port 9001 --error-probability 0.01`) so that
killing a replica is a real process kill and the gateway sees a real connection failure.

`LocalEngineBackend` (the in-process `AsyncLLMEngine`, registered as `local`) lives in
`backends/local.py` and is documented in [`engine.md`](engine.md).

## Chat templating

The tokenizer's own `chat_template` is always preferred: it ships with the checkpoint and is
by definition the one the model was tuned on. Getting this wrong does not raise — it silently
produces a model that rambles, ignores the system prompt or never stops.

When there is no tokenizer to ask (the gateway fronts a remote server, or the checkpoint is a
base model with no template), a neutral role-labelled fallback is used:

```
System: be brief
User: hello
Assistant:
```

The fallback does not imitate any model's real template, and `ChatTemplate.uses_tokenizer`
says which path ran. Tokenizers are loaded once per model with `local_files_only=True` by
default, so starting a server never turns into a multi-gigabyte download; misses are cached as
the fallback so they are not retried per request. A template that raises at render time also
degrades to the fallback — a slightly off prompt beats a 500.

## Metrics

One private `CollectorRegistry` per app (two gateways in one process must not collide), all
prefixed `turboserve_gateway_`:

| Metric | Type | Labels |
| --- | --- | --- |
| `ttft_seconds` | histogram | `tenant`, `model`, `backend`, `lane` |
| `tpot_seconds` | histogram | `tenant`, `model`, `backend`, `lane` |
| `e2e_seconds` | histogram | `tenant`, `model`, `backend`, `lane` |
| `tokens_total` | counter | + `kind` = `prompt`\|`completion` |
| `requests_total` | counter | + `status` = `ok`\|`error`\|`rate_limited`\|`unauthorized`\|`forbidden` |
| `rate_limited_total` | counter | `tenant`, `limit` = `rpm`\|`tpm`\|`concurrency` |
| `cost_usd_total` | counter | `tenant`, `model` |
| `inflight_requests` | gauge | `tenant`, `model` |
| `queue_depth` | gauge | `model` — the HPA custom metric |
| `backend_up` | gauge | `backend`, `model`, `lane` |
| `canary_weight` | gauge | `model` |

TTFT, TPOT and E2E use the same definitions as the benchmark harness
(`turboserve.bench.records`), so a Grafana panel and a benchmark report measure the same
quantities. Latencies are observed only when they exist: a request refused by auth has no
TTFT, and recording a zero would drag the percentile down and hide real regressions.

`cost_usd_total` attributes spend using the **configured list prices** in
`configs/models.yaml`. They are an input to that attribution, not a measurement of anything;
no number in `docs/results.md` comes from them.

## Usage accounting

`UsageAccumulator` takes token counts from, in order of preference: the backend's own `usage`
on the terminating event; the token ids on each event; a count of non-empty text deltas (for
an HTTP backend that streams text and reports no usage); and, for the prompt only, a
character-based estimate when the gateway has no tokenizer at all. Only the last is an
approximation, and a record produced that way sets `UsageRecord.estimated`, so nothing
downstream can mistake it for a measurement.

TTFT is measured from `arrival_ts` — the moment the request was admitted, before auth, quota
and routing — not from the moment the backend was called. That is the number a client would
measure; excluding the gateway's own overhead would flatter the engine. TPOT is `None` below
two output tokens, because a one-token completion has no inter-token gap.

`prompt_tokens_details.cached_tokens` carries the engine's prefix-cache hits, which is the
field OpenAI already uses for prompt caching, so a client that understands prompt caching
needs no turboserve-specific code.

## Tracing

Metrics say *how many* requests were slow; a trace says which one, and where the time went.
`tracing.py` adds one span per generation and propagates the trace to whatever answered it,
so a slow request at the gateway and the engine's own view of that same request are two
nodes of one trace rather than two systems to correlate by timestamp.

**Off unless asked for.** `TURBOSERVE_OTEL_ENDPOINT` — the OTLP/HTTP path of a collector,
e.g. `http://otelcol:4318/v1/traces` — is the whole switch. Unset, `configure_tracing`
returns a disabled `Tracing`, imports nothing from `opentelemetry` and hands the request
path a null span whose every method does nothing, so the feature costs an attribute lookup
per request rather than an exporter, a queue and four packages. The packages are an optional
extra (`uv sync --extra otel`); with the endpoint set and the extra missing, the gateway logs
one warning and serves exactly as before — observability must never be the reason a fleet
cannot boot.

| Setting | Environment variable | What it does |
| --- | --- | --- |
| `otel_endpoint` | `TURBOSERVE_OTEL_ENDPOINT` | OTLP/HTTP traces endpoint; empty disables tracing |
| `otel_service_name` | `TURBOSERVE_OTEL_SERVICE_NAME` | `service.name` on the exported resource |
| `otel_service_namespace` | `TURBOSERVE_OTEL_SERVICE_NAMESPACE` | `service.namespace`; the chart sets the release namespace |
| `otel_sample_ratio` | `TURBOSERVE_OTEL_SAMPLE_RATIO` | Ratio for traces this gateway *starts*; a sampled parent is always honoured |

The span:

```
turboserve.generate                        child of the FastAPI server span
  turboserve.tenant             "acme"
  turboserve.model              "Qwen/Qwen2.5-7B-Instruct"
  turboserve.backend            "vllm-a"   set once the router has chosen a replica
  turboserve.lane               "stable"
  turboserve.lora               "acme-support-r16"   absent when no adapter was resolved
  turboserve.prompt_tokens      <int>      the count the backend reported, or the estimate
  turboserve.completion_tokens  <int>
  event first_token  { turboserve.ttft_ms = <float>, milliseconds since arrival }
  event finished     { turboserve.finish_reason = "stop", turboserve.status = "ok" }
```

Attributes are namespaced because OpenTelemetry's semantic conventions own the unprefixed
names, and the suffixes are the label names `metrics.py` already uses, so a span and a metric
series join on the same words. `first_token` carries the same TTFT the histogram does —
measured from arrival, through auth, quotas and routing — so a trace and a dashboard cannot
disagree about what TTFT means.

**Ids, never content.** No prompt text, no completion text, no messages, no API key and
nothing derived from them ever reaches a span. A trace backend is usually a different trust
domain from the gateway and frequently retains data for months, and a prompt is the one thing
here that is unambiguously the tenant's. A test asserts it by sending a distinctive string
and searching the exported span for it.

**Propagation.** The FastAPI auto-instrumentation adopts an incoming `traceparent`, and
`OpenAICompatBackend` injects the current context into its outgoing request, so vLLM's or
SGLang's own spans hang under the gateway's. The header goes on the individual request and
never on the client's default headers, which are shared by every request on that connection
pool. `/healthz`, `/readyz` and `/metrics` are excluded: they are polled forever and tracing
them buries everything else.

**No global state.** Nothing calls `trace.set_tracer_provider`. The provider lives on
`GatewayState` and is handed to the instrumentation explicitly, so two gateways in one
process — the real one and the in-process mock upstream the chaos harness builds — get two
providers instead of fighting over a global only the first caller may set.

Running a collector is a prerequisite this repository does not ship. Anything that terminates
OTLP works: `otel/opentelemetry-collector`, Grafana Alloy, Tempo, or Jaeger with OTLP
enabled. `docker-compose.yml` passes `TURBOSERVE_OTEL_ENDPOINT` straight through, and the
Helm chart renders it from `gateway.tracing.endpoint` (with `serviceName` and `sampleRatio`
beside it, and a render-time check that the ratio is a fraction).

## Running it

```bash
# in-process mock backend, no configuration needed
uv run turboserve gateway serve --engine mock --model mock-model --port 8000 --no-require-auth

# model pools from configs/models.yaml, tenants from configs/tenants.yaml
uv run turboserve gateway serve --engine config --tenants configs/tenants.yaml \
    --models configs/models.yaml --port 8000

# front an existing OpenAI-compatible server (vLLM, SGLang, TGI) -- one flag, any engine
uv run turboserve gateway serve --engine http://127.0.0.1:8001/v1 --model Qwen/Qwen2.5-7B-Instruct
uv run turboserve gateway serve --engine http://127.0.0.1:30000/v1 --model Qwen/Qwen2.5-7B-Instruct

# validate configuration without starting anything
uv run turboserve gateway config-check

# hash a key for configs/tenants.yaml
uv run turboserve gateway hash-key 'sk-your-key-here'
```

Then, with the example configuration:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Authorization: Bearer sk-turboserve-acme-dev' \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

The keys in `configs/tenants.yaml` are throwaway development keys whose plaintext is written
beside their digests, so a fresh clone can issue a request without minting anything — which
also means they are public. Replace every one of them before deploying, and keep real keys in
a Secret. `turboserve gateway config-check` names any tenant that still accepts one, and the
gateway logs the same warning at startup whenever authentication is required.

In code:

```python
from turboserve.gateway import create_app

app = create_app(settings, router=my_router, tenants=my_tenants, metrics=my_metrics)
```

Every collaborator is injectable and nothing is read from module-level state, which is what
lets two gateways coexist in one process without sharing quotas or metrics.

## How it is tested

`tests/unit/test_gateway_*.py`, 194 tests, all CPU, no network, no model weights:

| File | Covers |
| --- | --- |
| `test_gateway_auth.py` | Digest hashing, bearer parsing, 401 vs 403, allow-lists, per-tenant adapter namespacing, the shipped `configs/tenants.yaml` |
| `test_gateway_limits.py` | Bucket refill arithmetic against an injected clock, overdraw and `Retry-After`, rpm/tpm/concurrency refusals, slot release on exceptions |
| `test_gateway_router.py` | Lane share vs canary weight and intra-lane weight share (both statistical, seeded), health caching, retry before the first event, no retry after it, non-retryable errors, the concurrency gate, `models.yaml` loading |
| `test_gateway_backends.py` | Mock determinism, latency, error and drop rates, model/adapter refusal; `OpenAICompatBackend` against `httpx.MockTransport` replaying synthetic transcripts — streaming shapes, usage, adapter-as-model, status-code mapping, malformed chunks, health fallback, and `server_info()` against an SGLang-shaped `/version` + `/get_server_info`, a vLLM-shaped `/version` alone, and a server that answers neither |
| `test_gateway_app.py` | Every status code, the SSE format through a strict parser including `[DONE]`, opt-in usage chunks, quota 429s with `Retry-After`, metric exposition and label values, mid-stream failures |
| `test_gateway_usage.py` | The three latency definitions, token-count precedence, the OpenAI usage object, prices, per-tenant totals |
| `test_gateway_chat_template.py` | Tokenizer template vs fallback on the cached tiny Qwen2 and tiny Llama checkpoints, content-part flattening, render failures, caching |
| `test_gateway_cli.py` | `hash-key`, `config-check`, `--engine` resolution, and the mock server in-process |

```bash
uv run pytest -q tests/unit/test_gateway_auth.py tests/unit/test_gateway_limits.py \
  tests/unit/test_gateway_router.py tests/unit/test_gateway_backends.py \
  tests/unit/test_gateway_app.py tests/unit/test_gateway_usage.py \
  tests/unit/test_gateway_chat_template.py tests/unit/test_gateway_cli.py
```

The statistical tests (lane share, replica weight share, drop rate) use seeded RNGs, so they
are deterministic reruns rather than flaky sampling.

## Limitations

These are deliberate, and each is enforced rather than assumed:

- **One completion per request.** `n`, `best_of` and batched `prompt` arrays are refused with
  400 so that a 429, a retry and a usage record each describe exactly one thing.
- **No tool calls, no structured output, no logit bias.** Refused with 400 rather than ignored.
- **`logprobs` is passed to the engine but not rendered** into the response body; the field is
  accepted and reaches `SamplingParams`, and the response carries `logprobs: null`.
- **No response caching and no request coalescing.** Prompt reuse is handled a layer down by
  the engine's prefix cache, where it can actually share KV blocks.
- **Quota state is per process.** Two gateway replicas each enforce the configured limits, so
  the fleet-wide limit is the configured one times the replica count. A shared store would be
  the fix; it is not built.
- **Tenants and model pools are read at startup.** Changing them means a new app object (or a
  rolling restart); there is no hot reload.
- **Prompt token counts may be estimates** when the gateway has no local tokenizer for a model
  and the backend reports no usage. Such records are flagged `estimated`.
- **Tracing is exported, never stored.** The gateway has no trace backend of its own and no
  buffer beyond the exporter's: with no collector reachable, spans are dropped after the
  OTLP client's own retries and the request path never notices. That is the intended
  failure mode — a serving request must not wait on an observability pipeline.
- **The chat template needs local files by default.** A gateway fronting a remote server with
  no local checkpoint uses the fallback template; set `GatewayOptions(local_files_only=False)`
  to allow a download.

Nothing on this page states a performance figure. Measured numbers live in
[`results.md`](results.md), rendered from `results/**/*.json`.

## References

- OpenAI API reference — chat completions, completions and streaming chunk shapes:
  <https://platform.openai.com/docs/api-reference/chat>
- vLLM OpenAI-compatible server, including `extra_body` sampling parameters and serving LoRA
  adapters under the `model` field:
  <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- SGLang's OpenAI-compatible API and its native endpoints (`/health`, `/version`,
  `/get_server_info`, `/generate`): <https://docs.sglang.ai/>
- Server-Sent Events, WHATWG HTML §9.2 (framing, `data:` fields, comments):
  <https://html.spec.whatwg.org/multipage/server-sent-events.html>
- RFC 9110 §10.2.3 `Retry-After`: <https://www.rfc-editor.org/rfc/rfc9110#field.retry-after>
- Token bucket: Tanenbaum & Wetherall, *Computer Networks*, §5.4 (traffic shaping).
- Prometheus metric and label naming conventions:
  <https://prometheus.io/docs/practices/naming/>
