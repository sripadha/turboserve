#!/usr/bin/env bash
#
# Run the benchmark suite on the rented instance.
#
#   scripts/vastai/run_remote.sh                       # make bench PROFILE=h100
#   scripts/vastai/run_remote.sh make bench-one SCENARIO=prefix_cache
#   scripts/vastai/run_remote.sh bash                  # interactive shell on the instance
#
# The command runs under `nohup` with its output tee'd to a log on the instance, and this
# script tails that log. A dropped laptop connection therefore does not kill a run that is
# costing $2-3/hour, and reconnecting means re-running this script.
#
# The instance's price is exported into the run so that every result file records the
# $/GPU-hour it was produced at, as `gpu_price_per_hour` with `price_source` naming where
# the figure came from. A cost-per-million-tokens number without that is not reproducible.

set -Eeuo pipefail

STATE_FILE="${VAST_STATE_FILE:-${XDG_STATE_HOME:-$HOME/.local/state}/turboserve/vastai.env}"
PROFILE="${PROFILE:-h100}"

if [ -f "$STATE_FILE" ]; then
  # shellcheck disable=SC1090
  . "$STATE_FILE"
fi
: "${VAST_SSH_HOST:?set VAST_SSH_HOST or run scripts/vastai/provision.sh first}"
: "${VAST_SSH_PORT:?set VAST_SSH_PORT or run scripts/vastai/provision.sh first}"
REMOTE_DIR="${VAST_WORKDIR:-/workspace/turboserve}"
PRICE="${VAST_PRICE_PER_HOUR:-}"

# Everything after the script name is the remote command; with none, run the whole suite.
if [ "$#" -gt 0 ]; then
  REMOTE_CMD="$*"
else
  REMOTE_CMD="make bench PROFILE=${PROFILE}"
fi

LOG="/workspace/turboserve-run-$(date +%Y%m%d-%H%M%S).log"
SSH_OPTS=(-p "$VAST_SSH_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

echo "==> ${REMOTE_CMD}"
echo "    on root@${VAST_SSH_HOST}:${REMOTE_DIR}  (\$${PRICE:-unknown}/hr)"
echo "    log: ${LOG}"

# The script is fed to the remote shell on stdin (`bash -s`) rather than passed as an
# argument: ssh concatenates its command arguments and lets the remote login shell re-parse
# them, which mangles anything multi-line or quoted. stdin has neither problem.
#
# `nohup` and not `setsid`: nohup execs into the command, so $! is the real pid and its exit
# status can be waited on, while the ignored SIGHUP means a dropped laptop connection does
# not kill a run that is being billed by the hour. Reconnect by re-running this script --
# the log is still there and still growing.
ssh "${SSH_OPTS[@]}" "root@${VAST_SSH_HOST}" bash -s <<REMOTE
set -Eeuo pipefail
export PATH="/root/.local/bin:\$PATH"
export HF_HOME="\${HF_HOME:-/workspace/hf-cache}"
# Recorded into every result file, so a cost-per-token figure can be traced to the price
# actually paid rather than to a price someone remembered.
export TURBOSERVE_GPU_PRICE_PER_HOUR='${PRICE}'
export TURBOSERVE_GPU_PRICE_SOURCE='vast.ai instance price at run time'
export TURBOSERVE_PROFILE='${PROFILE}'
cd '${REMOTE_DIR}'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "--- ${REMOTE_CMD}"
nohup bash -lc '${REMOTE_CMD}' > '${LOG}' 2>&1 &
pid=\$!
echo "started pid \$pid, streaming ${LOG}"
tail -n +1 -f --pid=\$pid '${LOG}' &
tail_pid=\$!
# `wait || status=\$?` and not a bare wait: under `set -e` a failing run would end the
# script here and the exit status would never be reported back through ssh.
status=0
wait \$pid || status=\$?
wait \$tail_pid 2>/dev/null || true
echo "--- exited with status \$status"
exit \$status
REMOTE

echo
echo "==> finished. Bring the results back with: scripts/vastai/pull_results.sh"
