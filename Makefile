# turboserve developer entry points.
#
# Every target except `docker-build` runs inside the repo-local uv environment; none of them
# touch the system Python. Targets that depend on modules that do not exist yet (bench,
# results, k8s-lint) are intentionally absent rather than stubbed -- see the "Targets that do
# not exist yet" table in CONTRIBUTING.md for what each one lands with.

UV ?= uv
PYTEST_ARGS ?=
DOCKER ?= docker
# Overridden by CI, which tags the image it smoke-tests `turboserve-gateway:ci`.
IMAGE ?= turboserve-gateway:dev

.DEFAULT_GOAL := help
.PHONY: help setup lock lint format typecheck test test-slow test-gpu test-all hooks hwinfo \
        docker-build clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk -F':.*?## ' '{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

# No `uv venv` here on purpose: `uv sync` creates .venv itself, while `uv venv` exits 2 on a
# checkout that already has one, which made `make setup` impossible to re-run.
setup: ## Create .venv and install runtime + dev dependencies (torch from the cu124 index)
	$(UV) sync --all-groups

lock: ## Refresh uv.lock without installing
	$(UV) lock

lint: ## ruff check + ruff format --check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Apply ruff fixes and formatting in place
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## mypy over src/
	$(UV) run mypy src

test: ## Default suite: CPU unit tests (slow and gpu excluded)
	$(UV) run pytest -q $(PYTEST_ARGS)

# pytest exits 5 when a marker expression matches no tests. No `slow` suite exists yet, so an
# empty selection there is reported and ignored; remove this tolerance as soon as the first
# slow test lands, so a disappearing suite can never be mistaken for a passing one.
test-slow: ## Tests marked `slow` (load real models; minutes, not seconds)
	@$(UV) run pytest -m slow -ra $(PYTEST_ARGS) || ( [ $$? -eq 5 ] && echo "make: no tests marked 'slow' yet" )

# No exit-5 tolerance here: tests/gpu/ has a real suite, so an empty selection means the
# suite went missing and must fail. Without CUDA the tests skip (conftest guard), exit 0.
test-gpu: ## Tests marked `gpu` (require CUDA; run them one process at a time)
	$(UV) run pytest -m gpu -ra $(PYTEST_ARGS)

test-all: ## Every test, including slow and gpu
	$(UV) run pytest -m "" -ra $(PYTEST_ARGS)

hooks: ## Install the pre-commit hooks into .git/hooks
	$(UV) run pre-commit install

hwinfo: ## Print the hardware/software record embedded in benchmark results
	$(UV) run turboserve hwinfo

# Only the gateway image. Dockerfile.engine builds on nvidia/cuda:12.4.1-runtime and installs
# the CUDA 12.4 torch resolution, which does not fit in a hosted GitHub runner's free disk, so
# it is built on the GPU host instead:
#   docker build -f Dockerfile.engine -t turboserve-engine:dev .
# Docker is not available under WSL without Docker Desktop integration; the same command runs
# in the `docker` job of .github/workflows/ci.yml, which then smoke-runs the image.
docker-build: ## Build the gateway image (IMAGE=turboserve-gateway:dev); needs Docker
	$(DOCKER) build -f Dockerfile.gateway -t $(IMAGE) .

clean: ## Remove caches and build artifacts (keeps .venv and results/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
	find . -path ./.venv -prune -o -name '__pycache__' -type d -print0 | xargs -0 rm -rf
