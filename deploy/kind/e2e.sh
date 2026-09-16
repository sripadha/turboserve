#!/usr/bin/env bash
#
# End-to-end chaos test of the turboserve deployment on a throwaway kind cluster.
#
# What it proves, in one run:
#
#   1. the gateway image built from this checkout starts under Kubernetes,
#   2. the Helm chart's objects are accepted by a real API server (not just kubeconform),
#   3. the Service, probes, PodDisruptionBudget and preStop drain are configured well
#      enough that deleting a gateway pod every CHAOS_INTERVAL seconds, for the whole
#      duration of a steady RPS load, keeps the failed-request rate under MAX_ERROR_RATE.
#
# (3) is the point. Every individual piece -- readiness gating, maxUnavailable: 0, the
# preStop sleep that covers endpoint propagation, three replicas spread over two workers --
# exists to make pod loss invisible to a client, and this is the only place where that
# claim is actually tested rather than asserted.
#
# Usage:
#   deploy/kind/e2e.sh                    # create a cluster, run, tear it down
#   KEEP_CLUSTER=1 deploy/kind/e2e.sh     # leave the cluster up for debugging
#   SKIP_BUILD=1 deploy/kind/e2e.sh       # reuse an already-built IMAGE
#   SKIP_CLUSTER=1 deploy/kind/e2e.sh     # use the current kubectl context as-is
#
# Requires docker, kind, kubectl, helm and python3 on PATH. Docker is not available inside
# WSL without Docker Desktop integration, so on this project's development machine the
# script is run by .github/workflows/kind-e2e.yml rather than locally; every object it
# installs is validated locally with `helm lint` and `kubeconform` instead.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CLUSTER_NAME="${CLUSTER_NAME:-turboserve-e2e}"
NAMESPACE="${NAMESPACE:-turboserve}"
RELEASE="${RELEASE:-turboserve}"
# Object names are pinned with --set fullnameOverride below so they do not depend on how
# the release happens to be named: the chart drops the chart name when the release already
# contains it, so composing names out of the release name here would be right for one
# release name and wrong for another. Everything this script addresses derives from
# FULLNAME instead.
FULLNAME="${FULLNAME:-turboserve}"
GATEWAY_SVC="${FULLNAME}-gateway"
IMAGE="${IMAGE:-turboserve-gateway:e2e}"

RPS="${RPS:-20}"
DURATION="${DURATION:-60}"
CHAOS_INTERVAL="${CHAOS_INTERVAL:-10}"
MAX_ERROR_RATE="${MAX_ERROR_RATE:-0.005}"
REPLICAS="${REPLICAS:-3}"
# The gateway is installed with --no-require-auth (gateway.requireAuth=false), so the load
# generator needs no credential. The variable under test here is pod lifecycle, and a
# bearer token would only add a way for the run to fail for an unrelated reason.
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

KEEP_CLUSTER="${KEEP_CLUSTER:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_CLUSTER="${SKIP_CLUSTER:-0}"

LOADGEN_JOB="turboserve-loadgen"
RESULT_FILE="${RESULT_FILE:-${REPO_ROOT}/results/kind-e2e/run.json}"
BEGIN_MARKER="-----BEGIN TURBOSERVE RESULT JSON-----"
END_MARKER="-----END TURBOSERVE RESULT JSON-----"

CHAOS_PID=""
DELETIONS=0

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

require() {
  local missing=0
  for binary in "$@"; do
    command -v "$binary" >/dev/null 2>&1 || { echo "missing required binary: $binary" >&2; missing=1; }
  done
  [ "$missing" -eq 0 ] || die "install the missing tools and re-run"
}

dump_diagnostics() {
  log "diagnostics"
  kubectl get pods -n "$NAMESPACE" -o wide || true
  kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -40 || true
  kubectl logs -n "$NAMESPACE" "job/${LOADGEN_JOB}" --tail=60 2>/dev/null || true
  kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/component=gateway --tail=60 --prefix 2>/dev/null || true
}

cleanup() {
  local status=$?
  if [ -n "$CHAOS_PID" ] && kill -0 "$CHAOS_PID" 2>/dev/null; then
    kill "$CHAOS_PID" 2>/dev/null || true
    wait "$CHAOS_PID" 2>/dev/null || true
  fi
  if [ "$status" -ne 0 ]; then
    dump_diagnostics
  fi
  if [ "$SKIP_CLUSTER" != "1" ] && [ "$KEEP_CLUSTER" != "1" ]; then
    log "deleting kind cluster ${CLUSTER_NAME}"
    kind delete cluster --name "$CLUSTER_NAME" >/dev/null 2>&1 || true
  elif [ "$KEEP_CLUSTER" = "1" ]; then
    info "cluster ${CLUSTER_NAME} left running (KEEP_CLUSTER=1); delete it with: kind delete cluster --name ${CLUSTER_NAME}"
  fi
  exit "$status"
}
trap cleanup EXIT

# --------------------------------------------------------------------------------------
# 1. cluster and image
# --------------------------------------------------------------------------------------
create_cluster() {
  if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    info "reusing existing kind cluster ${CLUSTER_NAME}"
  else
    log "creating kind cluster ${CLUSTER_NAME}"
    kind create cluster --name "$CLUSTER_NAME" --config "${REPO_ROOT}/deploy/kind/kind-config.yaml" --wait 180s
  fi
  kubectl cluster-info --context "kind-${CLUSTER_NAME}" >/dev/null
  kubectl config use-context "kind-${CLUSTER_NAME}" >/dev/null
}

build_and_load_image() {
  if [ "$SKIP_BUILD" != "1" ]; then
    log "building ${IMAGE} from Dockerfile.gateway"
    docker build -f "${REPO_ROOT}/Dockerfile.gateway" -t "$IMAGE" "$REPO_ROOT"
  fi
  log "side-loading ${IMAGE} into the kind nodes"
  # The overlay and the chart both set imagePullPolicy so the node never reaches for a
  # registry; a missing load would otherwise surface as ImagePullBackOff.
  kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"
}

# --------------------------------------------------------------------------------------
# 2. install
# --------------------------------------------------------------------------------------
install_chart() {
  log "installing the chart (engine.mode=mock, ${REPLICAS} replicas)"
  # Mock engine: this cluster has no GPU and no model. The mock backend emits synthetic
  # tokens with the same streaming shape, which is exactly what a resilience test needs --
  # the variable under test is pod lifecycle, not token quality.
  helm upgrade --install "$RELEASE" "${REPO_ROOT}/deploy/helm/turboserve" \
    --namespace "$NAMESPACE" --create-namespace \
    --set "fullnameOverride=${FULLNAME}" \
    --set "image.repository=${IMAGE%:*}" \
    --set "image.tag=${IMAGE##*:}" \
    --set image.pullPolicy=Never \
    --set engine.mode=mock \
    --set gateway.requireAuth=false \
    --set "engine.model=${MODEL}" \
    --set "gateway.replicaCount=${REPLICAS}" \
    --set gateway.service.type=NodePort \
    --set gateway.service.nodePort=30080 \
    --set gateway.pdb.enabled=true \
    --wait --timeout 5m

  kubectl rollout status -n "$NAMESPACE" "deploy/${GATEWAY_SVC}-stable" --timeout=3m
  kubectl get pods -n "$NAMESPACE" -o wide
}

smoke_test() {
  log "smoke-testing the gateway through the Service"
  kubectl run turboserve-smoke -n "$NAMESPACE" --rm -i --restart=Never \
    --image="$IMAGE" --image-pull-policy=Never \
    --command -- python -c "
import json, urllib.request
base = 'http://${GATEWAY_SVC}.${NAMESPACE}.svc.cluster.local'
for path in ('/healthz', '/readyz'):
    with urllib.request.urlopen(base + path, timeout=10) as response:
        print(path, response.status)
with urllib.request.urlopen(base + '/v1/models', timeout=10) as response:
    payload = json.loads(response.read())
print('models', [entry['id'] for entry in payload['data']])
assert payload['data'], 'the gateway serves no models'
"
}

# --------------------------------------------------------------------------------------
# 3. load + chaos
# --------------------------------------------------------------------------------------
chaos_loop() {
  # Delete one gateway pod every CHAOS_INTERVAL seconds for the length of the run. One at
  # a time and never more: killing a majority would test the client's retry budget rather
  # than the deployment's resilience. In mock mode the engine runs inside the gateway
  # process, so these pods *are* the engine pods; with engine.mode=vllm the same loop is
  # pointed at app.kubernetes.io/component=engine.
  local deadline=$(( SECONDS + DURATION ))
  local alive victim
  while [ "$SECONDS" -lt "$deadline" ]; do
    sleep "$CHAOS_INTERVAL"
    # Candidates are Running pods that are not already terminating: a pod with a
    # deletionTimestamp is on its way out, and deleting it again would spend a chaos tick
    # doing nothing.
    alive="$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/component=gateway" \
      -o go-template='{{range .items}}{{if not .metadata.deletionTimestamp}}{{if eq .status.phase "Running"}}{{.metadata.name}}{{"\n"}}{{end}}{{end}}{{end}}' \
      2>/dev/null || true)"
    # Never take the last two: with fewer than three live pods the Service can end up with
    # a single endpoint, and the run would then measure how a client copes with an outage
    # rather than whether the deployment avoids one.
    if [ "$(printf '%s\n' "$alive" | grep -c .)" -lt 3 ]; then
      echo "    chaos: skipping this tick, fewer than 3 live gateway pods"
      continue
    fi
    victim="$(printf '%s\n' "$alive" | head -n1)"
    echo "    chaos: deleting pod ${victim}"
    kubectl delete pod -n "$NAMESPACE" "$victim" --wait=false >/dev/null 2>&1 || true
  done
}

run_load() {
  log "running load: ${RPS} rps for ${DURATION}s, deleting a gateway pod every ${CHAOS_INTERVAL}s"
  kubectl delete job -n "$NAMESPACE" "$LOADGEN_JOB" --ignore-not-found >/dev/null

  # The job writes the run file to an emptyDir and then prints it between markers, because
  # a completed pod's filesystem is gone by the time anything could copy it out. The
  # markers keep the JSON separable from the load generator's own log lines.
  kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${LOADGEN_JOB}
  labels:
    app.kubernetes.io/name: turboserve
    app.kubernetes.io/component: loadgen
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 600
  template:
    metadata:
      labels:
        app.kubernetes.io/name: turboserve
        app.kubernetes.io/component: loadgen
    spec:
      restartPolicy: Never
      containers:
        - name: loadgen
          image: ${IMAGE}
          imagePullPolicy: Never
          command: ["/bin/sh", "-c"]
          args:
            - |
              GATEWAY_URL="http://${GATEWAY_SVC}.${NAMESPACE}.svc.cluster.local/v1"
              turboserve bench loadgen --url "\$GATEWAY_URL" --rps ${RPS} --duration ${DURATION} --out /results/run.json
              status=\$?
              echo "${BEGIN_MARKER}"
              cat /results/run.json
              echo "${END_MARKER}"
              exit \$status
          volumeMounts:
            - name: results
              mountPath: /results
      volumes:
        - name: results
          emptyDir: {}
EOF

  # Best effort: on a fast cluster the pod can go Running and finish before the watch is
  # established, and a timeout here is not a failure of the thing under test.
  kubectl wait -n "$NAMESPACE" --for=condition=ready pod \
    -l app.kubernetes.io/component=loadgen --timeout=120s || true

  chaos_loop &
  CHAOS_PID=$!

  local deadline=$(( SECONDS + DURATION + 180 ))
  local phase=""
  while [ "$SECONDS" -lt "$deadline" ]; do
    phase="$(kubectl get job -n "$NAMESPACE" "$LOADGEN_JOB" -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || true)"
    case "$phase" in
      *Complete*) break ;;
      *Failed*)   die "the loadgen job failed; see the diagnostics above" ;;
    esac
    sleep 5
  done
  [ -n "$phase" ] || die "the loadgen job did not finish within $(( DURATION + 180 ))s"

  wait "$CHAOS_PID" 2>/dev/null || true
  CHAOS_PID=""
}

collect_and_assert() {
  log "collecting the run file"
  mkdir -p "$(dirname "$RESULT_FILE")"
  kubectl logs -n "$NAMESPACE" "job/${LOADGEN_JOB}" \
    | awk -v begin="$BEGIN_MARKER" -v end="$END_MARKER" \
        'index($0, begin) { capture = 1; next } index($0, end) { capture = 0 } capture' \
    > "$RESULT_FILE"
  [ -s "$RESULT_FILE" ] || die "no result JSON found between the markers in the job log"
  info "wrote ${RESULT_FILE}"

  log "asserting the error rate"
  # A run that produced far fewer requests than it was asked for is a failure even when
  # every request it did make succeeded; half the requested count is a generous floor that
  # still catches a gateway that was unreachable for most of the window.
  python3 "${REPO_ROOT}/deploy/kind/assert_error_rate.py" "$RESULT_FILE" \
    --max-error-rate "$MAX_ERROR_RATE" \
    --min-requests "$(( RPS * DURATION / 2 ))"
}

main() {
  require docker kind kubectl helm python3 awk
  [ "$SKIP_CLUSTER" = "1" ] || create_cluster
  [ "$SKIP_CLUSTER" = "1" ] || build_and_load_image
  install_chart
  smoke_test
  run_load
  collect_and_assert
  log "end-to-end chaos test passed"
}

main "$@"
