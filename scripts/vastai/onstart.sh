#!/usr/bin/env bash
#
# vast.ai on-start script. Runs as root inside the instance the first time it boots, and
# again after every stop/start, so everything below is idempotent.
#
# It is uploaded by provision.sh with `vastai create instance --onstart`; its output lands
# in the instance log (`vastai logs <id>`) and in /var/log/onstart.log.
#
# What it does, in order: install uv and the handful of system packages the later steps
# need, clone the repository, create the virtual environment with the `vllm` extra, and
# pre-download the models the h100 benchmark profile uses. All of it happens while the
# laptop is doing nothing, which matters because the instance is billed by the hour from
# the moment it starts.
#
# It finishes by writing a marker file. provision.sh waits for that marker rather than for
# ssh alone, because ssh comes up long before a 15 GB checkpoint has finished downloading
# and a benchmark started too early would measure the download.
#
# Configuration arrives as environment variables set at creation time
# (`vastai create instance --env '-e KEY=value'`):
#
#   HF_TOKEN            Hugging Face token, required only for gated checkpoints
#   TURBOSERVE_REPO_URL  git URL to clone           (default: the public repository)
#   TURBOSERVE_REF       branch, tag or commit      (default: main)
#   TURBOSERVE_MODELS    space-separated repo ids   (default: the h100 profile's models)

set -Eeuo pipefail

exec > >(tee -a /var/log/onstart.log) 2>&1
echo "=== turboserve onstart $(date -Is) ==="

WORKDIR="${TURBOSERVE_WORKDIR:-/workspace/turboserve}"
REPO_URL="${TURBOSERVE_REPO_URL:-https://github.com/sripadha/turboserve.git}"
REF="${TURBOSERVE_REF:-main}"
MARKER="${TURBOSERVE_MARKER:-/workspace/.turboserve-onstart-done}"

# The models the h100 profile serves: a 7B target, a 3B target, and the 1.5B/0.5B drafts
# the speculative-decoding scenario pairs with them. Override with TURBOSERVE_MODELS.
DEFAULT_MODELS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct"
MODELS="${TURBOSERVE_MODELS:-$DEFAULT_MODELS}"

export DEBIAN_FRONTEND=noninteractive
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export UV_LINK_MODE=copy

rm -f "$MARKER"
mkdir -p /workspace "$HF_HOME"

# --- 1. system packages ----------------------------------------------------------------
# The pytorch image is Ubuntu with conda and CUDA already in it; these are the few things
# it does not ship that the rest of this script and sync.sh need.
echo "--- installing system packages"
apt-get update -qq
apt-get install -y -qq --no-install-recommends git rsync curl ca-certificates jq >/dev/null

# --- 2. uv ------------------------------------------------------------------------------
# The repository pins its Python and its lock file to uv; using the image's conda python
# would resolve a different dependency set from the one that was tested.
if ! command -v uv >/dev/null 2>&1; then
  echo "--- installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"
uv --version

# --- 3. repository ----------------------------------------------------------------------
# Cloned here so the instance is usable on its own; scripts/sync.sh later rsyncs the
# laptop's working tree over the top, which is how uncommitted changes get measured.
if [ -d "${WORKDIR}/.git" ]; then
  echo "--- updating existing checkout at ${WORKDIR}"
  git -C "$WORKDIR" fetch --depth 1 origin "$REF"
  git -C "$WORKDIR" checkout -f FETCH_HEAD
else
  echo "--- cloning ${REPO_URL} @ ${REF}"
  git clone --depth 1 --branch "$REF" "$REPO_URL" "$WORKDIR" \
    || git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

# --- 4. environment ----------------------------------------------------------------------
# --extra vllm pulls in the production engine. It is an extra rather than a dependency
# because vLLM needs a CUDA GPU and would make the package uninstallable on a laptop.
echo "--- creating the virtual environment (this pulls torch + vllm, several GB)"
uv sync --extra vllm

echo "--- torch / GPU check"
uv run python - <<'PY'
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    print(f"gpu: {properties.name} sm_{properties.major}{properties.minor} {total_gib:.0f} GiB")
PY

# --- 5. models ---------------------------------------------------------------------------
# snapshot_download rather than a CLI: it is the same API the library uses, it resumes a
# partial download, and its name does not change between huggingface-hub releases. Gated
# repositories need HF_TOKEN to be set at instance-creation time.
echo "--- downloading models into ${HF_HOME}"
for model in $MODELS; do
  echo "    ${model}"
  MODEL_ID="$model" uv run python - <<'PY'
import os

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
path = snapshot_download(
    model_id,
    token=os.environ.get("HF_TOKEN") or None,
    # Safetensors only: the .bin duplicates would double the download for nothing.
    ignore_patterns=["*.pth", "*.bin", "*.msgpack", "*.h5"],
)
print("      ->", path)
PY
done

# --- 6. marker ---------------------------------------------------------------------------
{
  echo "onstart_completed_at=$(date -Is)"
  echo "workdir=${WORKDIR}"
  echo "hf_home=${HF_HOME}"
  echo "models=${MODELS}"
} > "$MARKER"

echo "=== turboserve onstart complete: ${MARKER} ==="
