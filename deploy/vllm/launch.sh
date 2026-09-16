#!/usr/bin/env bash
#
# Start a vLLM OpenAI-compatible server natively, with the flags this project's
# Kubernetes deployment uses.
#
# Why a shell script exists next to a Helm chart: a vast.ai instance *is* a container, and
# there is no nested Docker inside one, so the Kubernetes path cannot be used there. On
# that host vLLM is pip-installed and started by this script, and the gateway is pointed at
# it with `turboserve gateway serve --engine vllm --engine-url http://127.0.0.1:8000/v1`.
# The flags are the same ones deploy/helm/turboserve/templates/engine-deployment.yaml
# renders, so a measurement taken on vast.ai is comparable with one taken in a cluster.
#
# Usage:
#   deploy/vllm/launch.sh                                   # defaults below
#   MODEL=Qwen/Qwen2.5-3B-Instruct SPEC_MODEL=Qwen/Qwen2.5-0.5B-Instruct deploy/vllm/launch.sh
#   ENABLE_LORA=1 LORA_MODULES="tenant-a=/adapters/tenant-a tenant-b=/adapters/tenant-b" deploy/vllm/launch.sh
#
# Every knob is an environment variable so the script can be driven from
# scripts/vastai/run_remote.sh without editing anything.

set -Eeuo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"

ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"

# Speculative decoding: set SPEC_MODEL to a draft checkpoint that shares the target's
# tokenizer. Its weights come out of the same GPU memory budget as the KV cache.
SPEC_MODEL="${SPEC_MODEL:-}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-4}"

# Multi-LoRA: MAX_LORAS adapters resident on the GPU at once; others are swapped in from
# host memory on demand. LORA_MODULES is a space-separated list of name=path pairs.
ENABLE_LORA="${ENABLE_LORA:-0}"
MAX_LORAS="${MAX_LORAS:-8}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
LORA_MODULES="${LORA_MODULES:-}"

LOG_FILE="${LOG_FILE:-}"
DRY_RUN="${DRY_RUN:-0}"

command -v vllm >/dev/null 2>&1 || {
  echo "vllm is not on PATH. Install the extra first:  uv sync --extra vllm" >&2
  exit 127
}

args=(
  serve "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --dtype "$DTYPE"
)

# Shared prompt prefixes are prefilled once and reused by every later request that starts
# with the same blocks.
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then args+=(--enable-prefix-caching); fi
# Interleave a long prompt's prefill with other sequences' decode steps, so one big prompt
# cannot stall every stream in flight.
if [ "$ENABLE_CHUNKED_PREFILL" = "1" ]; then args+=(--enable-chunked-prefill); fi

if [ -n "$SPEC_MODEL" ]; then
  args+=(--speculative-model "$SPEC_MODEL" --num-speculative-tokens "$NUM_SPECULATIVE_TOKENS")
fi

if [ "$ENABLE_LORA" = "1" ]; then
  args+=(--enable-lora --max-loras "$MAX_LORAS" --max-lora-rank "$MAX_LORA_RANK")
  for module in $LORA_MODULES; do
    args+=(--lora-modules "$module")
  done
fi

# Echo the exact command line before running it: every result file records the engine's
# flags, and this is where a human checks that they are the flags they meant. An `if`
# rather than `[ ... ] && exit 0`, because under `set -e` a false and-list is itself a
# failing command and would end the script without ever starting the server.
printf 'vllm'; printf ' %q' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

if [ -n "$LOG_FILE" ]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec vllm "${args[@]}" 2>&1 | tee -a "$LOG_FILE"
fi
# exec so signals reach vLLM directly: this script is often PID 1 of a container, and a
# wrapper process would swallow the SIGTERM that should drain the server.
exec vllm "${args[@]}"
