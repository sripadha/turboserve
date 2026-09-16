#!/usr/bin/env bash
#
# Start an SGLang OpenAI-compatible server natively, with the flags this project's
# Kubernetes deployment uses.
#
# Why a shell script exists next to a Helm chart: a vast.ai instance *is* a container, and
# there is no nested Docker inside one, so the Kubernetes path cannot be used there. On
# that host SGLang is pip-installed and started by this script, and the gateway is pointed
# at it with `turboserve gateway serve --engine http://127.0.0.1:30000/v1`.
# The flags are the same ones deploy/helm/turboserve/templates/engine-deployment.yaml
# renders in `sglang` mode, so a measurement taken on vast.ai is comparable with one taken
# in a cluster -- and with the vLLM measurement taken beside it by deploy/vllm/launch.sh.
#
# Usage:
#   deploy/sglang/launch.sh                                  # defaults below
#   PORT=30001 ENABLE_PREFIX_CACHING=0 deploy/sglang/launch.sh   # the prefix-cache control arm
#   SPEC_ALGORITHM=EAGLE SPEC_DRAFT_MODEL=Qwen/Qwen2.5-0.5B-Instruct deploy/sglang/launch.sh
#   ENABLE_LORA=1 LORA_PATHS="tenant-a=/adapters/tenant-a tenant-b=/adapters/tenant-b" deploy/sglang/launch.sh
#
# Every knob is an environment variable so the script can be driven from
# scripts/vastai/run_remote.sh without editing anything.

set -Eeuo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
# SGLang's equivalent of vLLM's --gpu-memory-utilization: the share of the device the
# server claims for weights plus KV cache. What is left absorbs activation spikes.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"

# RadixAttention prefix caching is ON by default in SGLang; there is no --enable flag, only
# the switch that turns it off. Setting this to 0 is how the prefix-cache scenario's control
# server is started (see docs/scenarios.md).
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# Speculative decoding: EAGLE or NEXTN, with a draft checkpoint that shares the target's
# tokenizer. Its weights come out of the same device budget MEM_FRACTION_STATIC governs.
SPEC_ALGORITHM="${SPEC_ALGORITHM:-}"
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"

# Multi-LoRA: LORA_PATHS is a space-separated list of name=path pairs, and
# MAX_LORAS_PER_BATCH is how many distinct adapters one batch may mix.
ENABLE_LORA="${ENABLE_LORA:-0}"
MAX_LORAS_PER_BATCH="${MAX_LORAS_PER_BATCH:-8}"
LORA_PATHS="${LORA_PATHS:-}"

PYTHON="${PYTHON:-python}"
LOG_FILE="${LOG_FILE:-}"
DRY_RUN="${DRY_RUN:-0}"

"${PYTHON}" -c 'import sglang' >/dev/null 2>&1 || {
  echo "sglang is not importable by ${PYTHON}. Install it into the environment first" >&2
  echo "(this repository ships no sglang extra; see deploy/sglang/README.md)." >&2
  exit 127
}

args=(
  -m sglang.launch_server
  --model-path "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --tp-size "$TP_SIZE"
)

# The only prefix-caching flag there is: RadixAttention is on unless it is disabled.
if [ "$ENABLE_PREFIX_CACHING" != "1" ]; then args+=(--disable-radix-cache); fi

if [ -n "$SPEC_ALGORITHM" ]; then
  args+=(--speculative-algorithm "$SPEC_ALGORITHM")
  if [ -n "$SPEC_DRAFT_MODEL" ]; then
    args+=(--speculative-draft-model-path "$SPEC_DRAFT_MODEL")
  fi
  args+=(--speculative-num-steps "$SPEC_NUM_STEPS")
  args+=(--speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS")
fi

if [ "$ENABLE_LORA" = "1" ]; then
  args+=(--max-loras-per-batch "$MAX_LORAS_PER_BATCH")
  # One --lora-paths flag takes every name=path pair, unlike vLLM's repeated
  # --lora-modules; passing them one flag at a time would keep only the last.
  if [ -n "$LORA_PATHS" ]; then
    args+=(--lora-paths)
    for pair in $LORA_PATHS; do
      args+=("$pair")
    done
  fi
fi

# Echo the exact command line before running it: every result file records the engine's
# flags, and this is where a human checks that they are the flags they meant. An `if`
# rather than `[ ... ] && exit 0`, because under `set -e` a false and-list is itself a
# failing command and would end the script without ever starting the server.
printf '%s' "$PYTHON"; printf ' %q' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

if [ -n "$LOG_FILE" ]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec "$PYTHON" "${args[@]}" 2>&1 | tee -a "$LOG_FILE"
fi
# exec so signals reach the server directly: this script is often PID 1 of a container, and
# a wrapper process would swallow the SIGTERM that should drain it.
exec "$PYTHON" "${args[@]}"
