#!/usr/bin/env bash
#
# Copy results/ back from the instance into this checkout.
#
#   scripts/vastai/pull_results.sh
#   scripts/vastai/pull_results.sh --dry-run
#
# One-directional on purpose: this pulls, and never deletes local files. A result file is
# the product of GPU time that has already been paid for, so the failure mode to avoid is
# a sync that removes one, not a stale copy.
#
# Result files are the only thing that comes back. Logs stay on the instance; fetch one
# explicitly if a run needs explaining.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_FILE="${VAST_STATE_FILE:-${XDG_STATE_HOME:-$HOME/.local/state}/turboserve/vastai.env}"

if [ -f "$STATE_FILE" ]; then
  # shellcheck disable=SC1090
  . "$STATE_FILE"
fi
: "${VAST_SSH_HOST:?set VAST_SSH_HOST or run scripts/vastai/provision.sh first}"
: "${VAST_SSH_PORT:?set VAST_SSH_PORT or run scripts/vastai/provision.sh first}"
REMOTE_DIR="${VAST_WORKDIR:-/workspace/turboserve}"

SSH_OPTS=(-p "$VAST_SSH_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)

mkdir -p "${REPO_ROOT}/results"
echo "==> pulling root@${VAST_SSH_HOST}:${REMOTE_DIR}/results/ -> ${REPO_ROOT}/results/"
rsync --archive --compress --human-readable --info=stats1,progress2 "$@" \
  -e "ssh ${SSH_OPTS[*]}" \
  "root@${VAST_SSH_HOST}:${REMOTE_DIR}/results/" "${REPO_ROOT}/results/"

echo
echo "==> local results/:"
find "${REPO_ROOT}/results" -name '*.json' -newermt '-1 day' -print | sort | head -40
echo
echo "Render the tables and plots from them with: make results"
