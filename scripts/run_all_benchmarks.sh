#!/usr/bin/env bash
#
# Run the whole benchmark suite and render the results pages.
#
# This is the script `make bench` and `scripts/vastai/run_remote.sh` call on the
# measurement host. It is deliberately a thin sequence of `turboserve bench ...`
# invocations rather than a second harness: every decision about what a scenario measures
# lives in the scenario module, and the only thing here is the order and which optional
# arms are available on this machine.
#
# Environment:
#   PROFILE              workload profile from configs/bench/profiles.yaml (default h100)
#   RESULTS_DIR          where result JSON is written (default results)
#   VLLM_URL             OpenAI-compatible vLLM server; enables every `vllm` arm
#   VLLM_BASELINE_URL    a second vLLM server started WITHOUT --enable-prefix-caching,
#                        used as the prefix-cache scenario's control arm
#   SGLANG_URL           OpenAI-compatible SGLang server; enables the `sglang` arms of
#                        naive-vs-cb and prefix-cache (the two scenarios that engine is
#                        projected and measured on)
#   SGLANG_BASELINE_URL  a second SGLang server started WITH --disable-radix-cache, the
#                        prefix-cache control arm for that engine -- RadixAttention is on
#                        by default, so it is the control that needs the flag
#   TURBOSERVE           how to invoke the CLI (default: uv run --frozen turboserve)
#   ADAPTERS_DIR         LoRA adapters the multi-lora scenario serves (default: adapters)
#   SKIP                 space-separated scenario names to skip, e.g. "spec-decode chaos"
#   EXTRA_<SCENARIO>     extra flags for one scenario, e.g. EXTRA_CHAOS="--mode inprocess"
#   DRY_RUN=1            print the commands instead of running them
#   CONTINUE_ON_ERROR=1  keep going when one scenario fails (the failure is still reported)
#
# Prices: scripts/vastai/run_remote.sh exports TURBOSERVE_GPU_PRICE_PER_HOUR and
# TURBOSERVE_GPU_PRICE_SOURCE; the scenarios read them into every result file, which is
# what makes a cost column possible. Nothing here invents a price.

set -Eeuo pipefail

PROFILE="${PROFILE:-h100}"
RESULTS_DIR="${RESULTS_DIR:-results}"
VLLM_URL="${VLLM_URL:-}"
VLLM_BASELINE_URL="${VLLM_BASELINE_URL:-}"
SGLANG_URL="${SGLANG_URL:-}"
SGLANG_BASELINE_URL="${SGLANG_BASELINE_URL:-}"
TURBOSERVE="${TURBOSERVE:-uv run --frozen turboserve}"
SKIP="${SKIP:-}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FAILED=()
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log() { printf '[run_all_benchmarks] %s\n' "$*" >&2; }

is_skipped() {
  local name="$1" entry
  for entry in ${SKIP}; do
    [[ "${entry}" == "${name}" ]] && return 0
  done
  return 1
}

# Extra flags for one scenario, from EXTRA_NAIVE_VS_CB / EXTRA_PREFIX_CACHE / ...
extra_flags() {
  local name="${1//-/_}"
  local var="EXTRA_${name^^}"
  printf '%s' "${!var:-}"
}

# True when the CLI advertises a subcommand. `spec-decode` and `multi-lora` are owned by
# the engine's speculative and LoRA module groups and are registered only when present,
# so the suite asks rather than assumes.
has_command() {
  ${TURBOSERVE} bench --help 2>/dev/null | grep -q -- "$1"
}

run_step() {
  local name="$1"
  shift
  if is_skipped "${name}"; then
    log "skipping ${name} (SKIP)"
    return 0
  fi
  log "==> ${name}: $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if "$@"; then
    return 0
  fi
  log "!!! ${name} failed"
  FAILED+=("${name}")
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    return 0
  fi
  return 1
}

log "profile=${PROFILE} results=${RESULTS_DIR} vllm_url=${VLLM_URL:-<none>} sglang_url=${SGLANG_URL:-<none>}"
log "started ${STARTED_AT}"

# ---------------------------------------------------------------------------------------
# 1. batching: sequential vs static batch vs continuous batching, then the same against
#    vLLM when a server is reachable.
# ---------------------------------------------------------------------------------------
# shellcheck disable=SC2046  # word splitting of extra_flags is intended
run_step naive-vs-cb ${TURBOSERVE} bench naive-vs-cb \
  --profile "${PROFILE}" --results-dir "${RESULTS_DIR}" \
  --arm naive_hf --arm static_batch --arm reference $(extra_flags naive-vs-cb)

if [[ -n "${VLLM_URL}" ]]; then
  # shellcheck disable=SC2046
  run_step naive-vs-cb-vllm ${TURBOSERVE} bench naive-vs-cb \
    --profile "${PROFILE}" --results-dir "${RESULTS_DIR}" \
    --arm vllm --url "${VLLM_URL}" $(extra_flags naive-vs-cb)
else
  log "no VLLM_URL: skipping the vLLM arm of naive-vs-cb"
fi

# One --url is one server, so the second production engine is its own invocation rather
# than a second flag on the one above.
if [[ -n "${SGLANG_URL}" ]]; then
  # shellcheck disable=SC2046
  run_step naive-vs-cb-sglang ${TURBOSERVE} bench naive-vs-cb \
    --profile "${PROFILE}" --results-dir "${RESULTS_DIR}" \
    --arm sglang --url "${SGLANG_URL}" $(extra_flags naive-vs-cb)
else
  log "no SGLANG_URL: skipping the SGLang arm of naive-vs-cb"
fi

# ---------------------------------------------------------------------------------------
# 2. prefix cache off vs on. On both production engines the cache is a launch flag, so each
#    control arm needs its own server (VLLM_BASELINE_URL, SGLANG_BASELINE_URL) rather than a
#    different request -- and a pair of URLs names one engine's pair of servers, which is
#    why each engine is its own invocation, selected with --backend.
# ---------------------------------------------------------------------------------------
# shellcheck disable=SC2046
run_step prefix-cache ${TURBOSERVE} bench prefix-cache \
  --profile "${PROFILE}" --results-dir "${RESULTS_DIR}" \
  --backend reference $(extra_flags prefix-cache)

if [[ -n "${VLLM_URL}" || -n "${VLLM_BASELINE_URL}" ]]; then
  PREFIX_VLLM_ARGS=()
  [[ -n "${VLLM_URL}" ]] && PREFIX_VLLM_ARGS+=(--url "${VLLM_URL}")
  [[ -n "${VLLM_BASELINE_URL}" ]] && PREFIX_VLLM_ARGS+=(--baseline-url "${VLLM_BASELINE_URL}")
  # shellcheck disable=SC2046
  run_step prefix-cache-vllm ${TURBOSERVE} bench prefix-cache \
    --profile "${PROFILE}" --results-dir "${RESULTS_DIR}" --backend vllm \
    "${PREFIX_VLLM_ARGS[@]}" $(extra_flags prefix-cache)
else
  log "no VLLM_URL: skipping the vLLM arms of prefix-cache"
fi

if [[ -n "${SGLANG_URL}" || -n "${SGLANG_BASELINE_URL}" ]]; then
  PREFIX_SGLANG_ARGS=()
  [[ -n "${SGLANG_URL}" ]] && PREFIX_SGLANG_ARGS+=(--url "${SGLANG_URL}")
  [[ -n "${SGLANG_BASELINE_URL}" ]] && PREFIX_SGLANG_ARGS+=(--baseline-url "${SGLANG_BASELINE_URL}")
  # shellcheck disable=SC2046
  run_step prefix-cache-sglang ${TURBOSERVE} bench prefix-cache \
    --profile "${PROFILE}" --results-dir "${RESULTS_DIR}" --backend sglang \
    "${PREFIX_SGLANG_ARGS[@]}" $(extra_flags prefix-cache)
else
  log "no SGLANG_URL: skipping the SGLang arms of prefix-cache"
fi

# ---------------------------------------------------------------------------------------
# 3. speculative decoding, when the scenario is part of this installation. The reference
#    engine sweeps every pair and k; a vLLM server launched with the same sweep is measured
#    as a second family of arms, named by --label-prefix so the two do not collide.
#
#    SGLang is deliberately not swept here, nor in multi-lora below: this suite measures it
#    on the two scenarios its arms are defined for (batching and prefix caching). Adding it
#    to the other three means giving it its own --label-prefix and a target-only control of
#    its own, which is a scenario change rather than an environment variable.
# ---------------------------------------------------------------------------------------
if has_command spec-decode; then
  # shellcheck disable=SC2046
  run_step spec-decode ${TURBOSERVE} bench spec-decode \
    --profile "${PROFILE}" --results-dir "${RESULTS_DIR}" $(extra_flags spec-decode)
  if [[ -n "${VLLM_URL}" ]]; then
    # shellcheck disable=SC2046
    run_step spec-decode-vllm ${TURBOSERVE} bench spec-decode \
      --profile "${PROFILE}" --results-dir "${RESULTS_DIR}" \
      --url "${VLLM_URL}" --label-prefix "vLLM " $(extra_flags spec-decode)
  else
    log "no VLLM_URL: skipping the vLLM arms of spec-decode"
  fi
else
  log "turboserve bench has no spec-decode command in this build; skipping"
fi

# ---------------------------------------------------------------------------------------
# 4. multi-adapter serving. The command is a sub-group (`multi-lora run`) and writes into
#    --out rather than --results-dir; ADAPTERS_DIR holds the adapters both engines serve
#    (scripts/make_lora_adapters.py writes them, and vLLM must be started with
#    --enable-lora --max-loras >= the largest arm).
# ---------------------------------------------------------------------------------------
ADAPTERS_DIR="${ADAPTERS_DIR:-adapters}"
if has_command multi-lora; then
  # shellcheck disable=SC2046
  run_step multi-lora ${TURBOSERVE} bench multi-lora run \
    --profile "${PROFILE}" --out "${RESULTS_DIR}" --adapters-dir "${ADAPTERS_DIR}" \
    $(extra_flags multi-lora)
  if [[ -n "${VLLM_URL}" ]]; then
    # shellcheck disable=SC2046
    run_step multi-lora-vllm ${TURBOSERVE} bench multi-lora run \
      --profile "${PROFILE}" --out "${RESULTS_DIR}" --adapters-dir "${ADAPTERS_DIR}" \
      --backend vllm --url "${VLLM_URL}" $(extra_flags multi-lora)
  else
    log "no VLLM_URL: skipping the vLLM arms of multi-lora"
  fi
else
  log "turboserve bench has no multi-lora command in this build; skipping"
fi

# ---------------------------------------------------------------------------------------
# 5. chaos: steady offered load through a replica fleet that is being killed.
# ---------------------------------------------------------------------------------------
# shellcheck disable=SC2046
run_step chaos ${TURBOSERVE} bench chaos \
  --profile "${PROFILE}" --results-dir "${RESULTS_DIR}" $(extra_flags chaos)

# ---------------------------------------------------------------------------------------
# 6. render every result file into the tables, the plots and docs/results.md. This is the
#    only step that produces a number a human reads, and it reads them all from JSON.
# ---------------------------------------------------------------------------------------
run_step render ${TURBOSERVE} bench render --results-dir "${RESULTS_DIR}"

if ((${#FAILED[@]})); then
  log "finished with failures: ${FAILED[*]}"
  exit 1
fi
log "finished: every scenario completed; see ${RESULTS_DIR}/README.md"
