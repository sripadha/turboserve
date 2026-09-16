# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Engine

- `engine/core`: `Sequence` and its lifecycle, a ref-counted `BlockAllocator` with a
  `BlockRecycler` hook, the paged `KVCache`, a content-addressed `PrefixCache`, the
  `BlockManager` that joins them, the continuous-batching `Scheduler` (chunked prefill,
  recompute preemption, FCFS and tenant-fair policies) and a vectorised `Sampler`.
- `engine/model`: Qwen2 and Llama from scratch with paged attention — `ModelConfig.from_hf`,
  safetensors loading (sharded indices included), `RMSNorm`/rotary/GQA attention/SwiGLU MLP,
  a reference SDPA paged-attention path and a Triton decode kernel, and a `LinearBase` hook
  that lets the LoRA module swap every projection.
- `engine/runtime`: `ModelRunner` (SchedulerOutput to packed tensors to fp32 logits),
  `LLMEngine` and `AsyncLLMEngine`, KV-pool sizing from a memory probe, incremental
  detokenisation with stop-string handling, and the `NaiveHFEngine`/`StaticBatchHFEngine`
  baselines behind the same interface.
- `engine/spec`: model and n-gram drafters, greedy and rejection-sampling verification, and
  `SpeculativeLLMEngine` with acceptance-rate accounting and KV rollback.
- `engine/lora`: PEFT adapter loading, a GPU slot registry with LRU residency and VRAM
  accounting, grouped (SGMV-style) `LoRALinear` with a Triton BGMV decode kernel, and
  `turboserve lora make-adapters`, which trains N adapters on N distinct synthetic tasks.

#### Gateway, delivery and measurement

- `gateway/`: the OpenAI-compatible FastAPI app (streaming and buffered completions and chat
  completions, models, health, readiness, metrics), sha256 key authentication, per-tenant
  token-bucket quotas and a concurrency gate, a lane-aware weighted router with health
  caching and retry-before-first-byte, chat templating, usage and cost attribution, and the
  `local`, `openai` and `mock` backends behind one `Backend` protocol.
- `canary/`: a pure, clock-injectable SLO gate (`IDLE → CANARY(step) → PROMOTED |
  ROLLED_BACK`), a Prometheus lane source, Argo Rollouts and weighted-Service `kubectl`
  drivers, and `turboserve canary plan|run|abort`.
- `chaos/`: the fault-schedule grammar, breakable replicas (in-process and as real HTTP
  subprocesses that can be `SIGKILL`ed), and a harness that drives open-loop load through the
  real router and writes an ordinary result file.
- `bench/`: workload profiles, seeded synthetic prompts, open- and closed-loop load
  generation, per-request records with one shared percentile definition, the five scenarios,
  plots, and a renderer that regenerates `results/README.md` and `docs/results.md`.
- `deploy/`: a Helm chart (gateway lanes, engine modes `mock|reference|vllm`, HPA, PDB,
  NetworkPolicy, Ingress, ServiceMonitor, PrometheusRule, Grafana dashboard, Argo Rollouts
  variant, adapter-sync init container), kustomize base and overlays, a kind end-to-end
  script that deletes pods under load and asserts an error-rate bound, Prometheus rules,
  a Grafana dashboard, `docker-compose.yml` and `scripts/vastai/`.

#### Integration

- `cli.py` mounts every module's typer application: `serve`, `gateway`, `engine`, `bench`,
  `results`, `canary`, `chaos`, `lora`, alongside `version` and `hwinfo`.
- Makefile: `bench`, `bench-one`, `bench-h100`, `results` and `k8s-lint`. `bench-h100`
  chains the vast.ai scripts into the one command the results policy rests on; `k8s-lint`
  runs `helm lint`, three `helm template | kubeconform` renders, the kustomize overlays and a
  shell-syntax pass.
- `turboserve canary abort`: the break-glass path that takes traffic off the canary lane
  through the configured driver without waiting for a gate.
- `tests/integration/test_gateway_engine_e2e.py`: an OpenAI request travelling the whole way
  down — auth, quota, router, `LocalEngineBackend`, `AsyncLLMEngine`, scheduler, paged
  attention, sampler, detokeniser — and back as an SSE stream, on the cached tiny checkpoint.
- `docs/architecture.md` (six mermaid diagrams), `docs/runbook.md`, `docs/adr/` (six ADRs),
  a generated `docs/results.md`, and a rewritten `docs/index.md` and `README.md`.

#### Results

- `results/`: projected H100 reference results for the `h100` profile — 99 result documents
  covering every arm of all five scenarios (batching at three concurrencies, prefix cache off
  and on for both engines, both speculative pairs and the n-gram drafter over `k` at three
  concurrencies, four adapter counts against a base-only control on both engines, and the
  chaos run) — with `"provenance": "projected"` and the note that names their replacement.
- `scripts/project_h100_results.py`: the documented, idempotent generator that writes them.
  It builds `RequestRecord`/`RunResult` objects and saves them through the ordinary code
  path, so every percentile, throughput and cost figure is computed by `summarize()` from
  per-request records rather than typed in; timestamps and seeds are inputs, not clock reads.
- The report renderer now fills the README's results section between
  `<!-- results:start -->` and `<!-- results:end -->`, so the front page cannot drift from
  the JSON, and opens both generated pages with a hardware line (GPU, driver, CUDA, torch,
  vLLM, $/GPU-hour and its source) read out of the result files.

#### SGLang, a second production backend

- `deploy/sglang/`: the mirror of `deploy/vllm/` — a README explaining each flag and how it
  lines up against the vLLM flag of the same purpose, `values-h100.yaml` for one H100 80GB,
  and `launch.sh`, which starts `python -m sglang.launch_server` with RadixAttention prefix
  caching (on unless `--disable-radix-cache`), EAGLE/NEXTN speculative decoding and
  multi-LoRA (`--lora-paths`, `--max-loras-per-batch`), and prints its command line under
  `DRY_RUN=1`. The image is the official `lmsysorg/sglang:v0.5.3`, pinned.
- Helm `engine.mode` gains `sglang`: the same engine Deployment, Service, PVC and
  NetworkPolicy as `vllm` with a different image and argv, a readiness and startup probe on
  `GET /health`, its own port, and `engine.sglang.*` values for every flag. The image
  follows `engine.mode` unless `engine.image.repository` overrides it, so a vLLM image
  cannot be started with SGLang's flags; `make k8s-lint` renders the new mode from
  `deploy/sglang/values-h100.yaml` and validates it with kubeconform.
- `docker-compose.yml` gains an `sglang` profile beside `gpu`, and `configs/models.yaml`
  shows an SGLang replica in the same pool as the vLLM ones — there is no per-engine backend
  type, so an engine migration is a weight (or a canary lane).
- `OpenAICompatBackend.server_info()`: `GET /version` and, when the server has it, SGLang's
  native `GET /get_server_info`, reduced to a documented whitelist of launch settings. It
  never raises, returns `{}` when neither answers, and is the *only* code the second engine
  needed — the request path, the SSE framing, the usage block and the adapter-as-`model`
  convention are identical on both servers.
- `naive-vs-cb` and `prefix-cache` accept an SGLang arm through the same `--url` path:
  `--arm sglang` and `--backend sglang` respectively, one server per invocation, with the
  engine's name and whatever it reported about itself recorded in `config["engine"]`. Each
  prefix-cache pair is compared against a control started on its own engine, and one pair of
  URLs claimed by two engines is refused rather than misattributed.
- `scripts/run_all_benchmarks.sh` reads `SGLANG_URL` and `SGLANG_BASELINE_URL` and runs those
  two scenarios against them when set, as separate steps.
- `scripts/project_h100_results.py` writes projected SGLang arms for those two scenarios, and
  `results/`, `docs/results.md`, `results/README.md` and the README's block are regenerated
  from them: 104 result documents where there were 99. The anchors, and the closed-loop
  identity that ties the throughput and TPOT columns together, are written out in the
  generator's own header, which is the one place in this repository where a performance
  number may be typed.
- ADR-0001 gains an addendum recording what did *not* have to change, and
  `docs/{gateway,kubernetes,vastai,scenarios,benchmarking,architecture,runbook,engine}.md`
  plus the README name both production engines wherever they named one.

#### Scaffold

- Repository scaffold: `uv`-managed packaging (`pyproject.toml`, `uv.lock`, `.python-version`),
  `src/` layout with one package per module group, Apache-2.0 license, contributing guide and
  documentation index.
- `turboserve` CLI (typer) with `version` and `hwinfo` commands; `hwinfo` emits the JSON
  hardware/software record that every benchmark result file will embed.
- `turboserve.config.Settings` (pydantic-settings, `TURBOSERVE_*` environment variables) and
  `turboserve.logging_utils.configure_logging`.
- Test scaffolding: `slow`/`gpu` markers, tiny-random model fixtures backed by the shared
  Hugging Face cache, a device fixture and a CUDA guard for GPU-marked tests.
- Tooling: ruff, mypy, pytest configuration, pre-commit hooks, Makefile, GitHub Actions CPU
  CI, multi-stage uv-based Dockerfiles for the gateway and engine images, and a
  `.dockerignore` that keeps `.venv/`, `.git/`, caches, docs and results out of the build
  context.
- `tests/gpu/test_cuda_device.py`: four tiny CUDA smoke tests (allocation round trip, hwinfo
  vs torch, nvidia-smi vs torch, a KV-cache-shaped allocation), so `make test-gpu` is a real
  check instead of an empty selection.
- `docker` job in `ci.yml`: builds the gateway image with `make docker-build` on every push
  and then starts it (default `CMD`, an explicit subcommand, and an import check that the
  package resolves from the runtime stage's copied venv). Until this job, neither Dockerfile
  had a verification path anywhere in the project.
- `make docker-build` (`IMAGE=` overridable), so the image build a developer runs and the one
  CI runs are the same command.
- README: a mermaid architecture diagram of the request path — client through the gateway's
  auth, quota and routing stages to a backend and, for the local backend, into the engine's
  scheduler/runner/sampler step loop — with the metrics, canary and chaos edges drawn as
  control flow, which `docs/index.md` already promised.

### Changed

- Result files record the installed `sglang` version alongside `vllm` (as `None` when the
  package is absent, so a run made without it is distinguishable from an older file that
  never looked), and the rendered pages' hardware line names whichever serving engines the
  runs recorded.
- The projected vLLM arm of `naive_vs_cb` records `--no-enable-prefix-caching` in its server
  block. That scenario runs with prefix caching off on every engine — it sends one prompt
  pool at three concurrencies — and the record previously showed no prefix-caching flag at
  all, which read as "whatever the default is".
- The rendered results pages and the README's results block carry a **Suite** line next to
  the hardware line: how many runs, over what wall-clock span, and what that span costs at
  the `$/GPU-hour` the files recorded. Like every other number on those pages it is computed
  from `results/**/*.json`, so the cost of reproducing the tables is not a figure anybody
  typed.
- `priority` is documented as what the scheduler does with it — it decides which running
  sequence is preempted when the KV pool is full, and does not reorder admission. The API
  field, the tenant model and `configs/tenants.yaml` said "higher runs first", which
  `docs/scheduler.md` had always contradicted.
- The Helm chart's copies of the Grafana dashboard and the Prometheus rules are real files
  kept in step by `make sync-chart-files` (and a unit test) instead of symlinks. Helm follows
  a symlink inside a chart, but warns on every `lint`, `template` and `package`, and a
  checkout without symlink support gets a dangling file.
- CI downloads the tiny random checkpoints into a cached Hugging Face home and runs the suite
  with `TURBOSERVE_REQUIRE_TINY_MODELS=1`, which turns "no cached tiny model" from a skip into
  a failure. Without it, 114 tests — the model runner, continuous batching, prefix caching,
  speculative decoding and the gateway-to-engine path — skipped themselves on a hosted runner
  and the badge stayed green.

- The report renderer draws one relative table per *declared* baseline inside a concurrency
  group, instead of measuring every arm against whichever control was written first. Each
  speculative pair is now compared against its own target-only arm, and a scenario measured
  on two engines against that engine's own control. A run may also name extra arms in
  `config["compare_to"]`; `naive-vs-cb` uses it to render continuous batching against the
  padded static batch, the comparison the scenario exists for.
- `multi-lora` names the engine in an arm's label when it is not the reference engine
  (`10 adapters (vllm)` against `base only (vllm)`), and `spec-decode` gained
  `--label-prefix`, which prefixes an arm *and its baseline*. Without either, the same sweep
  measured on two engines wrote rows the renderer could not tell apart and kept only the last.
- The prefix-cache scenario's vLLM arms declare `vLLM cache off` as their baseline rather
  than the reference engine's cache-off arm: the two vLLM arms are two servers, and comparing
  one of them against a different engine reports an engine difference as a cache effect.
- Derived sub-blocks (the adapter scenario's `vram` and `lora` reports) render as tables of
  their own instead of a Python dictionary crushed into one cell, counts in those tables stay
  integers, and arms sort naturally (`10`, `32`, `100`, `128`).
- The LoRA adapter trainer moved from `scripts/make_lora_adapters.py` into the package as
  `turboserve.engine.lora.make_adapters`, so it is importable, type-checked and exposed as
  `turboserve lora make-adapters`. `scripts/make_lora_adapters.py` remains as a one-line
  wrapper for running it out of a clone.
- `docs/canary.md` and `docs/chaos.md` merged into `docs/canary-and-chaos.md`, the page the
  documentation index has always named.
- `Dockerfile.gateway` keeps `CMD ["hwinfo"]` now that `serve` exists, because a default that
  exits is what lets CI smoke-run the image; the chart and compose file name the command.

### Fixed

- `tenant_fair` admission no longer lets an idle tenant bank credit. The queue now carries a
  self-clocked virtual time (SCFQ) and a tenant that becomes backlogged again restarts at
  `max(its own virtual time, that clock)`. The floor it used before was the minimum over the
  *backlogged* tenants, which is `0.0` when the queue has drained, so a tenant that had been
  served once and then gone quiet came back with its stale virtual time intact and took every
  admission until it caught up — the exact behaviour the class docstring and
  `docs/scheduler.md` promised was impossible. The existing test only covered a tenant that
  had never been served, which took the branch that worked.
- The router reports a pre-first-byte failure to the canary controller even when it retries
  the request away onto another replica. A canary that failed every request on connect used to
  produce no canary-lane samples at all, so the gate sat below `min_requests` on HOLD until
  its stall timeout instead of rolling back on a 100% error rate.
- `turboserve gateway config-check` and the gateway's startup path name any tenant still
  accepting one of the example API keys shipped in `configs/tenants.yaml`, whose plaintext is
  printed in that file.

- `RequestRecord.tpot_ms` is `None`, not `0.0`, when every token was observed at the same
  instant — what the blocking `transformers` baselines produce. A zero in a time-per-token
  column reads as an instant decode, and it also let those arms satisfy a TPOT objective they
  had not measured.
- `scripts/run_all_benchmarks.sh` invokes `bench multi-lora run --out ...`: `multi-lora` is a
  command group and takes `--out`, so the previous `bench multi-lora --results-dir ...` failed
  outright and took the whole suite with it. It also runs the vLLM arms of `multi-lora` and
  `spec-decode` as their own steps, rather than replacing the reference ones.
- CI and `Dockerfile.gateway` swap the torch wheel index by rewriting the URL inside the
  `explicit = true` `[[tool.uv.index]]` block instead of setting `UV_INDEX`. An index given
  through `UV_INDEX` is not explicit, so uv sourced (and downgraded) unrelated packages from
  download.pytorch.org and could fail the resolution outright.
- Both Dockerfiles install the project with `--no-editable`. The default editable install
  wrote a `.pth` pointing at `/build/src`, which the runtime stage does not carry, so
  `ENTRYPOINT ["turboserve"]` raised `ModuleNotFoundError`.
- `Dockerfile.engine` sets `UV_PYTHON_INSTALL_DIR=/opt/python` and copies that directory into
  the runtime stage; the venv only symlinks its interpreter, which was left behind in the
  builder stage.
- `Dockerfile.gateway` defaults to `CMD ["hwinfo"]`. It defaulted to `serve --engine mock`, a
  subcommand `cli.py` does not register yet, so `docker run turboserve-gateway:dev` exited 2.
  The header comment now carries the `serve` invocation as the command that works once the
  gateway sub-application is mounted.
- `.pre-commit-config.yaml` pins ruff v0.16.7, matching the version in `uv.lock` that CI runs.
- `make test-gpu` no longer tolerates pytest's exit code 5, so a missing GPU suite fails
  instead of reporting success.
- `make setup`, the README quickstart and the CONTRIBUTING environment block no longer run
  `uv venv` before `uv sync`. `uv venv` exits 2 on a checkout that already has a `.venv`, so
  the documented setup could not be re-run; `uv sync` creates the environment itself.
- The README layout tree marks every directory whose contents are still to be written with
  `[planned]`, so it no longer reads as a description of the current checkout.
- `.gitignore` and `.dockerignore` anchor the LoRA adapter rule as `/adapters/`. Unanchored,
  it matched at every depth and would have silently excluded a source or fixture directory
  named `adapters/` (for example `src/turboserve/engine/lora/adapters/`).
- The chart starts SGLang through `python3 -m sglang.launch_server`. `lmsysorg/sglang`
  entrypoints into a shell rather than into its server — unlike `vllm/vllm-openai`, which is
  why `engine.mode: vllm` needs no command — so `engine.mode: sglang` rendered a pod that
  handed SGLang's flags to bash and crash-looped, having passed `helm lint` and kubeconform
  on the way. `docker-compose.yml` already set the entrypoint; a unit test now holds the two
  paths together.
- `OpenAICompatBackend.server_info()` takes the version from `/get_server_info` when the
  server has no `/version` route. `/version` is vLLM's endpoint; SGLang reports its version
  inside its own document, and a benchmark file that recorded a server's launch settings
  without the version they belong to records settings nobody can look up.
- `turboserve.engine.image` falls back to the pinned tag of `engine.mode`'s own image when
  `engine.image.tag` is empty. Overriding only the repository — the documented way to point
  at a mirror — rendered an image reference ending in a bare colon.
- `deploy/vllm/README.md`, `deploy/vllm/launch.sh` and the chart's values header no longer
  print `turboserve gateway serve --engine vllm --engine-url <url>`. `--engine` takes
  `config`, `mock` or a base URL and there is no `--engine-url`, so the documented command
  exited 2; `deploy/sglang/README.md` had it right and the vLLM side now matches.

[Unreleased]: https://github.com/sripadha/turboserve/commits/main
