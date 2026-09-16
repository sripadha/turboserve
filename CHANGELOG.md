# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
- CONTRIBUTING: a "Targets that do not exist yet" table for `bench`, `results` and `k8s-lint`,
  which the Makefile header already pointed at but which did not exist, and a row in
  "Deliverables not built yet" for `Dockerfile.engine`, the one image here that has never been
  built (its CUDA layers do not fit a hosted runner's disk; it is built on the GPU host).
- README: a mermaid architecture diagram of the request path — client through the gateway's
  auth, quota and routing stages to a backend and, for the local backend, into the engine's
  scheduler/runner/sampler step loop — with the metrics, canary and chaos edges drawn as
  control flow. PLAN.md §4 requires it and `docs/index.md` already promised mermaid diagrams.
- `scripts/vastai/` and the `make bench-h100` target are tracked explicitly: a row in
  "Deliverables not built yet" naming the `bench/` module they land with, the `bench/` owner's
  scope in the module table, a `vastai/` line in the README layout tree and a note in
  `docs/index.md`. They were previously implied only by `scripts/ [planned]`, although the
  whole results policy depends on that one command.

### Fixed

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

[Unreleased]: https://github.com/sripadha/turboserve/commits/main
