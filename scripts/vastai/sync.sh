#!/usr/bin/env bash
#
# Push this working tree to the rented instance over rsync.
#
#   scripts/vastai/sync.sh              # push src/, tests/, configs/, scripts/, deploy/, docs/
#   DELETE=0 scripts/vastai/sync.sh     # do not remove remote files that are gone locally
#   scripts/vastai/sync.sh --dry-run    # show what would change
#
# onstart.sh already cloned the repository, so this exists for the case that actually
# matters day to day: measuring code that is not committed yet. rsync over the instance's
# own ssh port, not the vast.ai proxy, because the proxy is slow enough to notice.
#
# What is deliberately NOT sent:
#   .venv/     built on the instance; the laptop's wheels are the wrong CUDA build anyway
#   .git/      the clone has its own
#   results/   flows the other way; see pull_results.sh
#   models     never rsynced, per the project's rules; downloaded on the instance
#
# Connection details come from the state file written by provision.sh, or from
# VAST_SSH_HOST/VAST_SSH_PORT in the environment.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_FILE="${VAST_STATE_FILE:-${XDG_STATE_HOME:-$HOME/.local/state}/turboserve/vastai.env}"
DELETE="${DELETE:-1}"

die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

if [ -f "$STATE_FILE" ]; then
  # shellcheck disable=SC1090
  . "$STATE_FILE"
fi
: "${VAST_SSH_HOST:?set VAST_SSH_HOST or run scripts/vastai/provision.sh first}"
: "${VAST_SSH_PORT:?set VAST_SSH_PORT or run scripts/vastai/provision.sh first}"
REMOTE_DIR="${VAST_WORKDIR:-/workspace/turboserve}"

command -v rsync >/dev/null 2>&1 || die "rsync not found on this machine"

SSH_OPTS=(-p "$VAST_SSH_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)

RSYNC_ARGS=(
  --archive --compress --human-readable --partial --info=stats1,progress2
  --exclude '.git/'
  --exclude '.venv/'
  --exclude 'results/'
  --exclude 'adapters/'
  --exclude '__pycache__/'
  --exclude '*.py[cod]'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
  --exclude '.hf_cache/'
  --exclude '*.nsys-rep'
  --exclude '*.ncu-rep'
)
# --delete makes the remote tree match the local one exactly, which is what makes a run
# reproducible: a file deleted locally must not keep being imported remotely. It is a flag
# because the remote tree also holds the venv and the model cache, and a mistake here is
# expensive in download time -- hence the excludes above are listed before it.
if [ "$DELETE" = "1" ]; then
  RSYNC_ARGS+=(--delete --delete-excluded --filter 'protect .venv' --filter 'protect results')
fi

echo "==> syncing ${REPO_ROOT}/ -> root@${VAST_SSH_HOST}:${REMOTE_DIR}/"
ssh "${SSH_OPTS[@]}" "root@${VAST_SSH_HOST}" "mkdir -p '${REMOTE_DIR}'"
rsync "${RSYNC_ARGS[@]}" "$@" \
  -e "ssh ${SSH_OPTS[*]}" \
  "${REPO_ROOT}/" "root@${VAST_SSH_HOST}:${REMOTE_DIR}/"

echo "==> done. Re-resolve dependencies on the instance if pyproject.toml changed:"
echo "    ssh -p ${VAST_SSH_PORT} root@${VAST_SSH_HOST} 'cd ${REMOTE_DIR} && uv sync --extra vllm'"
