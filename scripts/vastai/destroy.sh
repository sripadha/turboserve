#!/usr/bin/env bash
#
# Destroy the rented instance and stop the meter.
#
#   scripts/vastai/destroy.sh            # asks first
#   FORCE=1 scripts/vastai/destroy.sh    # does not ask (for scripted teardown)
#
# Destroying is permanent: the instance's disk goes with it, including the model cache and
# any result file that has not been pulled. The confirmation prompt names both, because the
# expensive mistake here is not the $2 of GPU time, it is the run whose output was still
# only on that disk.

set -Eeuo pipefail

STATE_FILE="${VAST_STATE_FILE:-${XDG_STATE_HOME:-$HOME/.local/state}/turboserve/vastai.env}"
FORCE="${FORCE:-0}"

die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

if [ -f "$STATE_FILE" ]; then
  # shellcheck disable=SC1090
  . "$STATE_FILE"
fi
: "${VAST_INSTANCE_ID:?set VAST_INSTANCE_ID or run scripts/vastai/provision.sh first}"

command -v vastai >/dev/null 2>&1 || die "vastai CLI not found"

echo "instance:  ${VAST_INSTANCE_ID}"
echo "ssh:       root@${VAST_SSH_HOST:-?}:${VAST_SSH_PORT:-?}"
echo "price:     \$${VAST_PRICE_PER_HOUR:-?}/hr"

if [ "${VAST_SSH_HOST:-}" != "" ]; then
  # Best effort: the point is to notice un-pulled results before the disk is gone, so a
  # box that is already unreachable must not block the destroy.
  echo
  echo "result files still on the instance:"
  ssh -p "${VAST_SSH_PORT}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
    -o BatchMode=yes "root@${VAST_SSH_HOST}" \
    "find '${VAST_WORKDIR:-/workspace/turboserve}/results' -name '*.json' | head -20" 2>/dev/null \
    || echo "  (could not reach the instance)"
fi

if [ "$FORCE" != "1" ]; then
  echo
  read -r -p "Destroy instance ${VAST_INSTANCE_ID}? This deletes its disk. [y/N] " reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "aborted"; exit 1 ;;
  esac
fi

echo "==> destroying ${VAST_INSTANCE_ID}"
vastai destroy instance "$VAST_INSTANCE_ID"

rm -f "$STATE_FILE"
echo "==> destroyed; state file ${STATE_FILE} removed"
echo "    confirm with: vastai show instances"
