#!/usr/bin/env bash
#
# Rent one H100 80GB on vast.ai and get it ready to run the benchmark suite.
#
#   scripts/vastai/provision.sh                     # cheapest H100 SXM matching the filters
#   MAX_PRICE=2.00 scripts/vastai/provision.sh      # refuse anything above $2.00/hr
#   DRY_RUN=1 scripts/vastai/provision.sh           # print the offers and stop
#
# A vast.ai instance *is* a Docker container: you pick the image at creation time and there
# is no nested Docker inside it. That single fact shapes everything here -- the image is a
# pinned CUDA 12.4 PyTorch one, vLLM is pip-installed into it rather than run as its own
# container, and the gateway runs as a process next to it.
#
# The instance's price is captured into the state file and passed to the benchmark run,
# because every result file records the $/GPU-hour it was produced at; a cost-per-million
# -tokens figure is meaningless without it.
#
# State is written to $VAST_STATE_FILE (default under XDG_STATE_HOME, i.e. outside the
# repository) so that a rented instance is never accidentally committed or wiped by a
# `git clean`.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${VAST_STATE_FILE:-${XDG_STATE_HOME:-$HOME/.local/state}/turboserve/vastai.env}"
PYTHON="${PYTHON:-python3}"

# --- what to rent ------------------------------------------------------------------------
GPU_NAME="${GPU_NAME:-H100_SXM}"
NUM_GPUS="${NUM_GPUS:-1}"
DISK_GB="${DISK_GB:-200}"
CUDA_MIN="${CUDA_MIN:-12.4}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.98}"
MIN_DOWNLOAD_MBPS="${MIN_DOWNLOAD_MBPS:-300}"
# Guard against a price spike renting something absurd while nobody is watching.
MAX_PRICE="${MAX_PRICE:-4.00}"

# Pinned CUDA 12.4 image: the repository's torch is a cu124 wheel, and an image with a
# different CUDA minor would either refuse to load it or silently fall back.
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"
LABEL="${LABEL:-turboserve}"

WAIT_ONSTART="${WAIT_ONSTART:-1}"
ONSTART_TIMEOUT="${ONSTART_TIMEOUT:-3600}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v vastai >/dev/null 2>&1 || die "vastai CLI not found. Install it with: uv tool install vastai  (or pip install vastai)"
command -v "$PYTHON" >/dev/null 2>&1 || die "python3 not found; it is used to parse the CLI's JSON"
vastai show user --raw >/dev/null 2>&1 || die "vastai is not authenticated. Run: vastai set api-key <key>"

# --- 1. search ---------------------------------------------------------------------------
# The query is vast.ai's own filter language. rentable=true excludes offers already taken;
# reliability and inet_down matter more than they look, because a machine that drops mid-run
# costs the whole run and a slow link turns a 15 GB model download into an hour of billing.
QUERY="gpu_name=${GPU_NAME} num_gpus=${NUM_GPUS} disk_space>=${DISK_GB} cuda_vers>=${CUDA_MIN} reliability>${MIN_RELIABILITY} inet_down>=${MIN_DOWNLOAD_MBPS} rentable=true"

log "searching offers: ${QUERY}"
OFFERS_JSON="$(vastai search offers "$QUERY" --order 'dph_total' --raw)" \
  || die "offer search failed"

# The selection program is kept in a variable and passed with -c: `python -` would need
# stdin for the program, and stdin is already carrying the CLI's JSON.
PICK_OFFER_PY=$(cat <<'PY'
import json
import sys

max_price = float(sys.argv[1])
offers = json.load(sys.stdin)
if not offers:
    sys.exit("no offers matched the filters")

rows = sorted(offers, key=lambda offer: offer.get("dph_total", 1e9))
header = "    {:>10}  {:>8}  {:>7}  {:>6}  {:>8}  {}".format(
    "id", "$/hr", "disk GB", "rel", "down", "location"
)
print(header, file=sys.stderr)
for offer in rows[:5]:
    print(
        "    {:>10}  {:>8.3f}  {:>7.0f}  {:>6.3f}  {:>8.0f}  {}".format(
            offer.get("id", 0),
            offer.get("dph_total", 0.0),
            offer.get("disk_space", 0.0),
            offer.get("reliability2", 0.0),
            offer.get("inet_down", 0.0),
            offer.get("geolocation", "?"),
        ),
        file=sys.stderr,
    )

best = rows[0]
price = best.get("dph_total", 1e9)
if price > max_price:
    sys.exit(f"cheapest matching offer is ${price:.3f}/hr, above MAX_PRICE=${max_price:.2f}")
print(f"{best['id']} {price:.4f}")
PY
)

OFFER="$("$PYTHON" -c "$PICK_OFFER_PY" "$MAX_PRICE" <<<"$OFFERS_JSON")" || die "no usable offer"

OFFER_ID="${OFFER%% *}"
OFFER_PRICE="${OFFER##* }"
log "selected offer ${OFFER_ID} at \$${OFFER_PRICE}/hr"

if [ "$DRY_RUN" = "1" ]; then
  echo "    DRY_RUN=1, not creating an instance"
  exit 0
fi

# --- 2. create ---------------------------------------------------------------------------
# --ssh --direct asks for a direct SSH endpoint rather than the proxy, which is what makes
# rsync fast enough to be worth using. --onstart uploads the script that prepares the box.
log "creating instance from ${IMAGE}"
CREATE_JSON="$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --ssh --direct \
  --label "$LABEL" \
  --onstart "${SCRIPT_DIR}/onstart.sh" \
  --env "-e HF_TOKEN=${HF_TOKEN:-} -e TURBOSERVE_REPO_URL=${TURBOSERVE_REPO_URL:-https://github.com/sripadha/turboserve.git} -e TURBOSERVE_REF=${TURBOSERVE_REF:-main} -p 8000:8000 -p 8001:8001" \
  --raw)" || die "instance creation failed"

INSTANCE_ID="$("$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("new_contract") or d.get("id") or "")' <<<"$CREATE_JSON")"
[ -n "$INSTANCE_ID" ] || die "could not read the new instance id from: ${CREATE_JSON}"
log "instance ${INSTANCE_ID} created"

# --- 3. wait for ssh ---------------------------------------------------------------------
log "waiting for the instance to start"
SHOW_PY=$(cat <<'PY'
import json
import sys

try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    data = {}
if isinstance(data, list):
    data = data[0] if data else {}
print(
    data.get("actual_status") or data.get("cur_state") or "unknown",
    data.get("ssh_host") or "-",
    data.get("ssh_port") or "-",
    data.get("dph_total") or 0.0,
)
PY
)

SSH_HOST="-"
SSH_PORT="-"
PRICE="$OFFER_PRICE"
for _ in $(seq 1 120); do
  SHOW_JSON="$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || echo '{}')"
  read -r STATUS SSH_HOST SSH_PORT PRICE <<<"$("$PYTHON" -c "$SHOW_PY" <<<"$SHOW_JSON")"
  if [ "$STATUS" = "running" ] && [ "$SSH_HOST" != "-" ] && [ "$SSH_PORT" != "-" ]; then
    break
  fi
  printf '.'
  sleep 10
done
echo
[ "$SSH_HOST" != "-" ] || die "instance ${INSTANCE_ID} never reported an ssh endpoint; check: vastai show instance ${INSTANCE_ID}"
log "ssh endpoint: root@${SSH_HOST}:${SSH_PORT}  (\$${PRICE}/hr)"

# --- 4. state ----------------------------------------------------------------------------
mkdir -p "$(dirname "$STATE_FILE")"
cat > "$STATE_FILE" <<EOF
# Written by scripts/vastai/provision.sh at $(date -Is). Sourced by the other vastai scripts.
VAST_INSTANCE_ID=${INSTANCE_ID}
VAST_SSH_HOST=${SSH_HOST}
VAST_SSH_PORT=${SSH_PORT}
VAST_PRICE_PER_HOUR=${PRICE}
VAST_IMAGE=${IMAGE}
VAST_WORKDIR=${TURBOSERVE_WORKDIR:-/workspace/turboserve}
EOF
chmod 600 "$STATE_FILE"
echo "    state written to ${STATE_FILE}"

# --- 5. wait for the ssh daemon, then for onstart ----------------------------------------
log "waiting for sshd"
for _ in $(seq 1 60); do
  if ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
      -o BatchMode=yes "root@${SSH_HOST}" true 2>/dev/null; then
    break
  fi
  printf '.'
  sleep 10
done
echo

if [ "$WAIT_ONSTART" = "1" ]; then
  # ssh answers minutes before the box is usable: the onstart script is still installing
  # torch and downloading checkpoints. Starting a benchmark now would measure the download.
  log "waiting for the onstart script to finish (up to ${ONSTART_TIMEOUT}s; follow it with: vastai logs ${INSTANCE_ID})"
  deadline=$(( SECONDS + ONSTART_TIMEOUT ))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
        "root@${SSH_HOST}" 'test -f /workspace/.turboserve-onstart-done' 2>/dev/null; then
      log "instance ready"
      ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new "root@${SSH_HOST}" \
        'cat /workspace/.turboserve-onstart-done; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv'
      break
    fi
    printf '.'
    sleep 20
  done
  echo
fi

cat <<EOF

Next:
  scripts/vastai/sync.sh          push this working tree over the clone
  scripts/vastai/run_remote.sh    run: make bench PROFILE=h100
  scripts/vastai/pull_results.sh  bring results/ back
  scripts/vastai/destroy.sh       stop paying for it

The instance bills by the hour from now. \$${PRICE}/hr.
EOF
