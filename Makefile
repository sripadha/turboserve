# turboserve developer entry points.
#
# Every target except `docker-build` and the Kubernetes/vast.ai ones runs inside the
# repo-local uv environment; none of them touch the system Python.

UV ?= uv
PYTEST_ARGS ?=
DOCKER ?= docker
HELM ?= helm
KUBECTL ?= kubectl
KUBECONFORM ?= kubeconform
# Overridden by CI, which tags the image it smoke-tests `turboserve-gateway:ci`.
IMAGE ?= turboserve-gateway:dev
CHART ?= deploy/helm/turboserve

# Benchmarks. PROFILE selects a workload from configs/bench/profiles.yaml; `h100` is the
# published one and `dev-2060` is the small local shape.
PROFILE ?= h100
RESULTS_DIR ?= results
SCENARIO ?= naive-vs-cb

.DEFAULT_GOAL := help
.PHONY: help setup lock lint format typecheck test test-slow test-gpu test-all hooks hwinfo \
        bench bench-one bench-h100 results k8s-lint docker-build clean

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk -F':.*?## ' '{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

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

test: ## Default suite: CPU unit + integration tests (slow and gpu excluded)
	$(UV) run pytest -q $(PYTEST_ARGS)

test-slow: ## Tests marked `slow` (load real models; minutes, not seconds)
	$(UV) run pytest -m slow -ra $(PYTEST_ARGS)

# Without CUDA the tests skip (conftest guard) and the run still exits 0.
test-gpu: ## Tests marked `gpu` (require CUDA; run them one process at a time)
	$(UV) run pytest -m gpu -ra $(PYTEST_ARGS)

test-all: ## Every test, including slow and gpu
	$(UV) run pytest -m "" -ra $(PYTEST_ARGS)

hooks: ## Install the pre-commit hooks into .git/hooks
	$(UV) run pre-commit install

hwinfo: ## Print the hardware/software record embedded in benchmark results
	$(UV) run turboserve hwinfo

# ---------------------------------------------------------------------------------------
# Benchmarks
#
# `bench` is what runs ON the measurement host (scripts/vastai/run_remote.sh invokes
# exactly this line). `bench-h100` is what runs on a laptop: it rents the H100, ships this
# working tree to it, runs `make bench` there, and brings results/ home.
# ---------------------------------------------------------------------------------------
bench: ## Run every benchmark scenario here and render the results (PROFILE=h100|dev-2060)
	PROFILE=$(PROFILE) RESULTS_DIR=$(RESULTS_DIR) TURBOSERVE="$(UV) run turboserve" \
		scripts/run_all_benchmarks.sh

bench-one: ## Run one scenario (SCENARIO=naive-vs-cb|prefix-cache|spec-decode|multi-lora|chaos)
	$(UV) run turboserve bench $(SCENARIO) --profile $(PROFILE) --results-dir $(RESULTS_DIR)

# Needs the vastai CLI, an ssh key registered with vast.ai, and rsync. Every step is
# resumable on its own: if the run dies, re-run `scripts/vastai/run_remote.sh`. The
# instance is NOT destroyed automatically -- teardown deletes the instance's disk and any
# result file that was not pulled, so it stays an explicit `scripts/vastai/destroy.sh`
# (or `make bench-h100 DESTROY=1`).
DESTROY ?= 0
bench-h100: ## Rent an H100 on vast.ai, run the suite there, pull results/ back
	scripts/vastai/provision.sh
	scripts/vastai/sync.sh
	PROFILE=h100 scripts/vastai/run_remote.sh
	scripts/vastai/pull_results.sh
	$(MAKE) results RESULTS_DIR=$(RESULTS_DIR)
	@if [ "$(DESTROY)" = "1" ]; then FORCE=1 scripts/vastai/destroy.sh; \
	else echo "instance left running -- stop the meter with scripts/vastai/destroy.sh"; fi

results: ## Regenerate results/README.md, docs/results.md and the plots from results/*.json
	$(UV) run turboserve results render --results-dir $(RESULTS_DIR)

# ---------------------------------------------------------------------------------------
# Kubernetes assets. Static validation only -- nothing here talks to a cluster; the
# end-to-end run against a real API server is deploy/kind/e2e.sh (and the kind-e2e
# workflow). Mirrors the `k8s` job of .github/workflows/ci.yml.
#
# -ignore-missing-schemas is scoped to what it says: ServiceMonitor, PrometheusRule and
# Rollout are custom resources whose schemas are not part of Kubernetes. Every built-in
# object is still validated with -strict, which rejects unknown fields.
# ---------------------------------------------------------------------------------------
k8s-lint: ## helm lint + helm template | kubeconform + kustomize build (needs helm, kubectl, kubeconform)
	$(HELM) lint $(CHART)
	$(HELM) template turboserve $(CHART) \
		| $(KUBECONFORM) -strict -summary -ignore-missing-schemas
	$(HELM) template turboserve $(CHART) \
		-f deploy/vllm/values-h100.yaml \
		--set canary.enabled=true --set canary.weight=25 \
		--set ingress.enabled=true \
		--set engine.adapters.enabled=true \
		--set engine.adapters.sync.sourceUrl=s3://example-bucket/adapters \
		--set gateway.autoscaling.queueDepth.enabled=true \
		| $(KUBECONFORM) -strict -summary -ignore-missing-schemas
	$(HELM) template turboserve $(CHART) --set canary.argoRollouts.enabled=true \
		| $(KUBECONFORM) -strict -summary -ignore-missing-schemas
	@for overlay in deploy/kustomize/overlays/*/; do \
		echo "--- $$overlay"; \
		$(KUBECTL) kustomize "$$overlay" | $(KUBECONFORM) -strict -summary -ignore-missing-schemas; \
	done
	$(KUBECONFORM) -strict -summary -ignore-missing-schemas src/turboserve/chaos/k8s/podchaos.yaml
	@for script in deploy/kind/e2e.sh deploy/vllm/launch.sh scripts/*.sh scripts/vastai/*.sh; do \
		bash -n "$$script" && echo "ok  $$script"; \
	done

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
