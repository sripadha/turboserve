# Contributing to turboserve

The rules below exist so that parallel changes to disjoint directories never collide, and so
that every number published from this repository can be traced back to a file that a
measurement wrote.

## Environment

`uv` manages the interpreter, the lock file and the repo-local `.venv`. Nothing is
installed into the system Python.

```bash
uv sync --all-groups     # creates .venv (CPython 3.12, pinned by .python-version) and installs deps
uv run turboserve version
uv run pytest -q
```

`uv sync` creates the environment itself, so there is no separate `uv venv` step; running
`uv venv` on a checkout that already has a `.venv` fails instead of being a no-op.

`torch` is pinned to `2.6.0` and resolved from the CUDA 12.4 wheel index declared as the
`pytorch-cu124` index in `pyproject.toml`. That index is `explicit = true`, so only `torch`
comes from it and everything else still comes from PyPI. To install a CPU-only build
instead (CI and `Dockerfile.gateway` do this), repoint that index in the file:

```bash
sed -i 's|download.pytorch.org/whl/cu124|download.pytorch.org/whl/cpu|' pyproject.toml
uv sync --all-groups
git checkout -- pyproject.toml   # keep the committed CUDA index
```

Do **not** reach for `UV_INDEX=pytorch-cu124=...` to do this. An index supplied through the
environment is not `explicit`, so uv's first-index strategy takes over for every package
that index happens to carry: `filelock`, `idna`, `jinja2`, `numpy`, `setuptools` and
`typing-extensions` are then sourced (and silently downgraded) from download.pytorch.org,
and the resolution fails outright when a pin cannot be satisfied there. Editing the URL in
place is the only swap that keeps `explicit = true`.

Never commit a lock file produced with that edit: the committed `uv.lock` is the CUDA one,
and CI re-resolves on the fly (it runs `uv lock --check` against the committed lock first,
then edits and re-syncs).

## Everyday commands

| Command | What it does |
| --- | --- |
| `make lint` | `ruff check .` and `ruff format --check .` |
| `make format` | applies ruff fixes and formatting |
| `make typecheck` | `mypy src` |
| `make test` | the default suite: CPU unit tests, `slow` and `gpu` deselected |
| `make test-slow` | tests marked `slow` (real models; minutes) |
| `make test-gpu` | tests marked `gpu` (need CUDA; run one process at a time) |
| `make bench` | every benchmark scenario on this host, then renders the pages (`PROFILE=h100\|dev-2060`) |
| `make bench-one` | one scenario (`SCENARIO=naive-vs-cb`, ...) |
| `make bench-h100` | rents a vast.ai H100, runs the suite there, pulls `results/` back |
| `make results` | regenerates `results/README.md`, `docs/results.md`, the README's results section and the plots from `results/*.json` |
| `make sync-chart-files` | copies the Grafana dashboard and Prometheus rules into the Helm chart (a unit test fails when they drift) |
| `make k8s-lint` | `helm lint`, four `helm template \| kubeconform` renders (defaults, vLLM, SGLang, Argo Rollouts), the kustomize overlays, shell syntax |
| `make docker-build` | builds the gateway image (`docker build -f Dockerfile.gateway`); needs Docker |

`make lint typecheck test` must be green before you push. The same three run in CI, in the
`check` job; the `docker` job then runs `make docker-build` and starts the image.

`make docker-build` is the only target that does not run inside the repo-local `.venv`, and
the only one that needs a Docker daemon. CI is therefore where the gateway image is always
built: see [What has never been executed here](#what-has-never-been-executed-here) for the
engine image, which is built on the GPU host instead.

### Commands the CLI exposes

`turboserve --help` lists the real surface; each sub-application lives in the module that
owns it and is mounted on the root app by `src/turboserve/cli.py`:

| Command | Owner module |
| --- | --- |
| `turboserve serve`, `turboserve gateway ...` | `gateway/app.py` |
| `turboserve engine generate\|kv-size` | `engine/runtime/engine.py` |
| `turboserve bench ...` | `bench/cli.py`, which registers each scenario module |
| `turboserve results render\|show` | `bench/report.py` |
| `turboserve canary plan\|run\|abort` | `canary/k8s.py` |
| `turboserve chaos run\|plan` | `chaos/harness.py` |
| `turboserve lora make-adapters` | `engine/lora/make_adapters.py` |
| `turboserve version`, `turboserve hwinfo` | `cli.py` itself |

A module owner adds a command by exporting a `typer.Typer` from their own package; the
integrator mounts it. `bench/cli.py` goes one step further and discovers the optional
scenario modules (`spec-decode`, `multi-lora`) by name, so a build without them still has a
working `bench loadgen`.

`Dockerfile.gateway` still defaults to `CMD ["hwinfo"]` rather than to `serve`: `hwinfo`
prints the record every result file embeds and then exits, so CI can smoke-run the image and
see it succeed, while a server as the default would have to be killed by a timeout. The
chart and `docker-compose.yml` both name the command explicitly.

## Test markers and fixtures

Three tiers, declared in `pyproject.toml` and registered in `tests/conftest.py`:

- **default** — CPU only, seconds, no network. Uses tiny random checkpoints resolved from
  the local Hugging Face cache by the `tiny_qwen2_path` and `tiny_llama_path` fixtures
  (`hf-tiny-v2/tiny-random-Qwen2ForCausalLM` with `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`
  as a fallback, and `hf-internal-testing/tiny-random-LlamaForCausalLM`). Both fixtures call
  `snapshot_download(..., local_files_only=True)` and `pytest.skip` when the model is not
  cached — a test must never download inside the default suite.
- **`@pytest.mark.slow`** — loads a real checkpoint (`Qwen/Qwen2.5-0.5B-Instruct` and larger)
  or otherwise takes minutes. Deselected by default.
- **`@pytest.mark.gpu`** — needs CUDA. Deselected by default, and skipped automatically by an
  autouse guard in `conftest.py` when `torch.cuda.is_available()` is false.

The default tier loads tiny checkpoints, so a machine with an empty Hugging Face cache
quietly skips every test that exercises the model runner, continuous batching, prefix
caching, speculative decoding and the gateway-to-engine path. Where that coverage is part of
the contract — CI is, after its own `hf download` step — set
`TURBOSERVE_REQUIRE_TINY_MODELS=1` and the skip becomes a failure.

Directories, which are about *scope* rather than about cost: `tests/unit/` tests one module
against fakes, `tests/integration/` runs a request through several real ones at once (the
gateway to the reference engine over HTTP, on the tiny checkpoint), and `tests/gpu/` holds
the CUDA-only kernels. `tests/integration/` is part of the default suite because it is
seconds, not minutes; a test that needs a real checkpoint belongs in `slow` wherever it
lives. `tests/` has no `__init__.py`, so **test file basenames must be unique across all
three directories**.

Other conventions: deterministic seeds (`torch.Generator().manual_seed(...)`), `tmp_path`
for anything written to disk, and the `device` fixture instead of hard-coding `"cuda"`
(`TURBOSERVE_TEST_DEVICE=cpu` forces CPU on a GPU machine).

## GPU etiquette

A development GPU is small and usually shared, so `gpu`-marked tests behave accordingly.

- Keep a `gpu`-marked test under ~1.5 GB of VRAM and a few seconds; `del` your tensors and
  call `torch.cuda.empty_cache()` at the end.
- Never load a checkpoint bigger than 0.5B parameters outside a measurement run, and
  never run two GPU test processes at once (no `pytest -n` on GPU tests).
- Benchmarks are not run on a development machine. The measurement target is a single
  H100 80GB; see `docs/runbook.md` for how the suite is launched there.

## Module ownership

Each top-level package under `src/turboserve/` has one owner at a time:

| Path | Scope |
| --- | --- |
| `engine/core/` | sequences, block allocator, KV cache, prefix cache, scheduler, sampler |
| `engine/model/` | Qwen2/Llama layers, paged attention (reference and Triton) |
| `engine/runtime/` | model runner, `LLMEngine`/`AsyncLLMEngine`, naive baselines, streaming |
| `engine/spec/` | drafters, verification, speculative engine |
| `engine/lora/` | adapter loading, GPU slot registry, grouped LoRA linears |
| `gateway/` | OpenAI-compatible API, auth, quotas, routing, metrics |
| `canary/` | SLO-gated progressive delivery |
| `chaos/` | fault injection and the chaos harness |
| `bench/` | load generator, metrics, scenarios, reports, `scripts/vastai/` and the `make bench-h100` target |
| `deploy/` | Helm chart, kustomize overlays, kind e2e, `docker-compose.yml`, the `.github/workflows/kind-e2e.yml` workflow and the `helm lint` + `kubeconform` CI job |

### What has never been executed here

Three things in this repository have never been *run* from a development checkout. Each is
written down on the page that claims it rather than left to be discovered:

| Not executed here | Why | Where it does run |
| --- | --- | --- |
| A build of `Dockerfile.engine` | It layers the CUDA 12.4 torch resolution on `nvidia/cuda:12.4.1-runtime`, which does not fit in a hosted GitHub runner's free disk | the GPU host: `docker build -f Dockerfile.engine -t turboserve-engine:dev .` |
| Anything against a real Kubernetes cluster — the chart, the kustomize overlays, `kubectl argo rollouts`, the Prometheus queries | A cluster needs Docker and a GPU-less runner cannot serve the GPU modes. They are validated statically by `make k8s-lint` and driven against fakes in the unit tests | `.github/workflows/kind-e2e.yml`, which creates a kind cluster, installs the chart and runs the chaos/loadgen job |
| `scripts/vastai/*` against the vast.ai API | Running them rents a GPU. They are shell-syntax checked in `make k8s-lint` and their embedded Python helpers are exercised against recorded response shapes | the measurement session: `make bench-h100` |

`Dockerfile.gateway` is not in that table: CI's `docker` job builds it on every push and
starts the resulting image, so the gateway build is verified.

No benchmark has been run from a development checkout either. The measurement target is one
H100 80GB rented on vast.ai; until it is rented, `results/` holds *projected* files written
by `scripts/project_h100_results.py` — see
[No numbers without a results JSON](#no-numbers-without-a-results-json). The `gpu`-marked
tests in `tests/gpu/` do run on a CUDA device: they prove the Triton kernels compile and
agree with the reference implementations, in seconds, on any card.

Work inside one package at a time, plus its tests under `tests/` and its page under `docs/`.
Shared files — `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, CI workflows,
`src/turboserve/{cli,config,logging_utils,hwinfo}.py` — belong to the integration pass
rather than to one package. If you need a dependency that is not installed, say so in the
pull request instead of editing `pyproject.toml` in the same change.

## No numbers without a results JSON

Any performance claim — in the README, in a docs page, in a docstring, in a commit message —
must cite a file under `results/` carrying the hardware, software versions, config and
timestamp in the shape `turboserve.hwinfo.collect()` writes. Concretely:

- Numbers in `docs/results.md`, `results/README.md` and the README's own results section
  (between its `<!-- results:start -->` / `<!-- results:end -->` markers) are **generated**
  from `results/*.json` by `make results`. Never hand-edit them; the next render wins.
- The README and the docs pages carry no hand-written numbers at all, aspirational ones
  included: a number appears only inside a table rendered from `results/*.json`.
- Every result file says whether it was `measured` or `projected`, and the renderer prints
  that under each table. A scenario nobody has run on the published hardware is either
  absent or projected — never quietly presented as a measurement.
- Test fixtures that imitate a recording must have `synthetic` in their filename or header.

### The projected reference results

`results/` currently holds *projected* files: `scripts/project_h100_results.py` builds one
result document per scenario arm from the hardware model documented in its own header
(H100 SXM bandwidth and FLOPs, model weight and KV sizes, adapter shapes, acceptance
arithmetic, the real expansion of the chaos fault schedule), using this repository's own
`RequestRecord`/`RunResult` classes so that every summary is computed by `summarize()` from
per-request records rather than typed in. Each file carries `"provenance": "projected"` and
the note that names its replacement. The script is idempotent — its timestamps are passed
in, its seeds come from the profile, and a re-run rewrites its own files and index rows:

```bash
uv run python scripts/project_h100_results.py   # rewrite the projected files
make results                                    # re-render the pages and the README
```

Replace them by measuring: `make bench-h100` writes files with `"provenance": "measured"`
for the same arms, and the renderer shows the newest run of each arm, so the measured ones
take over as soon as they exist.

## Code style

Python 3.12, `from __future__ import annotations` at the top of every module, type hints on
every public function, pydantic or dataclasses for data, `logging.getLogger(__name__)` for
output (no `print` in library code; the CLI may use `rich`/`typer.echo`), no bare `except`.
Docstrings on public classes and functions explain *why* and state the invariants. Line
length is 100. No `TODO`/`FIXME` markers and no stub bodies: if something cannot run here,
implement it fully against the documented API and say in the docs what was executed and what
was not.

## Commits

One logical change per commit, imperative subject line, body explaining why. Run
`uv run pre-commit install` once so ruff and the whitespace hooks run before each commit.
