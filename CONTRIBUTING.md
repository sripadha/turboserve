# Contributing to turboserve

This repository is built by several people (and agents) working on disjoint directories at
the same time. The rules below exist so that two parallel changes never collide and so that
every number published from this repository can be traced back to a file that a machine
wrote.

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
| `make docker-build` | builds the gateway image (`docker build -f Dockerfile.gateway`); needs Docker |

`make lint typecheck test` must be green before you push. The same three run in CI, in the
`check` job; the `docker` job then runs `make docker-build` and starts the image.

`make docker-build` is the only target that does not run inside the repo-local `.venv`, and
the only one that cannot run on a WSL checkout without Docker Desktop integration. CI is
therefore where the gateway image is actually built: see
[Deliverables not built yet](#deliverables-not-built-yet) for the engine image, which is
built on the GPU host instead.

### Targets that do not exist yet

`make` deliberately has no `bench`, `results` or `k8s-lint` target. They are named in the
spec's Makefile list, and they arrive with the code that makes them real rather than as
stubs that exit 0 on nothing:

| Target | Lands with | Will wrap |
| --- | --- | --- |
| `bench` | `bench/` (load generator, scenarios) | `turboserve bench <scenario>` over `configs/bench/` |
| `results` | `bench/report.py` | regenerating `docs/results.md`, `results/README.md` and the plots from `results/*.json` |
| `k8s-lint` | the Helm chart under `deploy/helm/turboserve/` | `helm lint` and `helm template ... \| kubeconform -strict -summary` |

`bench-h100`, which chains `scripts/vastai/` around `bench`, is tracked in
[Deliverables not built yet](#deliverables-not-built-yet) together with the CI job that will
run `k8s-lint`.

The same applies to the CLI: `cli.py` registers `version` and `hwinfo` and nothing else.
Module owners write their commands as a `typer.Typer` sub-application inside their own
package and the integrator mounts it on the root app. That is why `Dockerfile.gateway`
defaults to `CMD ["hwinfo"]` rather than to the spec's `serve --engine mock`:
`turboserve serve` exits 2 until `gateway/` mounts its sub-application.

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

Other conventions: deterministic seeds (`torch.Generator().manual_seed(...)`), `tmp_path`
for anything written to disk, and the `device` fixture instead of hard-coding `"cuda"`
(`TURBOSERVE_TEST_DEVICE=cpu` forces CPU on a GPU machine).

## GPU etiquette

There is one small development GPU, shared by everyone working in this repo.

- Keep a `gpu`-marked test under ~1.5 GB of VRAM and a few seconds; `del` your tensors and
  call `torch.cuda.empty_cache()` at the end.
- Never load a checkpoint bigger than 0.5B parameters outside the measurement phase, and
  never run two GPU test processes at once (no `pytest -n` on GPU tests).
- Benchmarks are not run on the development machine. The measurement target is a single
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

### Deliverables not built yet

These are in the spec (`specs/turboserve.md` §1, §7) and in the acceptance criteria, but are
deliberately absent from the scaffold because they would have nothing to act on: the Helm
chart under `deploy/helm/turboserve/templates/` and the Prometheus/Grafana assets under
`deploy/prometheus/` and `deploy/grafana/dashboards/` are still empty, and so is `bench/`.
The first three belong to the `deploy/` owner and land with the chart; `scripts/vastai/`
belongs to the `bench/` owner and is the single command the whole results policy rests on.
The last row is not blocked on a missing module — it is blocked on hardware, and it is here
so that no Dockerfile in this repo stays unverified without being written down:

| Missing | Why it is blocked | Lands with |
| --- | --- | --- |
| `docker-compose.yml` | would bind-mount `deploy/prometheus/*` and `deploy/grafana/dashboards/*`, which are `.gitkeep` only | the Prometheus/Grafana assets |
| `.github/workflows/kind-e2e.yml` | needs a deployable chart and the gateway image's `serve` command | the Helm chart |
| `helm lint` + `helm template \| kubeconform` job in `ci.yml` | `helm lint` fails on a chart with no `Chart.yaml` | the Helm chart |
| `scripts/vastai/{provision,sync,run_remote,pull_results,destroy}.sh` and the `make bench-h100` target that chains them | they provision a vast.ai H100, sync the repo, run `make bench PROFILE=h100` there and pull `results/` back — there is no bench scenario to run yet | the `bench/` scenarios and `bench/report.py` (PLAN.md §2, `specs/turboserve.md` §7) |
| a build of `Dockerfile.engine` anywhere | it layers the CUDA 12.4 torch resolution on `nvidia/cuda:12.4.1-runtime`, which does not fit in a hosted GitHub runner's free disk, and Docker is unavailable on the WSL dev machine | the first GPU-host run, which builds it with `docker build -f Dockerfile.engine -t turboserve-engine:dev .` |

`Dockerfile.gateway` is not in that table: CI's `docker` job builds it on every push and
starts the resulting image, so the gateway build is verified. `Dockerfile.engine` is the one
image in this repo that has never been built.

Work only inside your directory, plus your own tests under `tests/` and your own page under
`docs/`. Shared files — `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, CI workflows,
`src/turboserve/{cli,config,logging_utils,hwinfo}.py` — are changed by whoever is doing the
integration pass, not by module owners. If you need a dependency that is not installed, say
so in your change description instead of editing `pyproject.toml`.

## No numbers without a results JSON

Any performance claim — in the README, in a docs page, in a docstring, in a commit message —
must cite a file under `results/` that was produced by running the benchmark code, and that
file must carry the hardware, software versions, config and timestamp collected by
`turboserve.hwinfo.collect()`. Concretely:

- Numbers in `docs/results.md` and `results/README.md` are **generated** from
  `results/*.json`. Never hand-edit them.
- The README and the docs pages carry no hand-written numbers at all, aspirational ones
  included: a number appears only inside a table rendered from `results/*.json`.
- A scenario that has not been run on the published hardware says "not measured here" and
  gives the exact command that would measure it.
- Test fixtures that imitate a recording must have `synthetic` in their filename or header.

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
