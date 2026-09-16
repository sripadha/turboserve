{{/*
Naming, labels and the few pieces of logic that more than one template needs.

Two rules keep this file small:
  * anything a single template uses stays in that template;
  * anything that decides *shape* (which workloads exist, what the gateway's argv is, how
    replicas are split between lanes) lives here, so that the Deployment templates stay
    readable and the decision is made in exactly one place.
*/}}

{{- define "turboserve.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name, capped at 63 characters (the label-value limit; a Deployment
name longer than that produces pod names Kubernetes rejects). If the release name already
contains the chart name -- `helm install turboserve ./turboserve` -- it is not repeated.
*/}}
{{- define "turboserve.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "turboserve.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Labels on every object. */}}
{{- define "turboserve.labels" -}}
helm.sh/chart: {{ include "turboserve.chart" . }}
{{ include "turboserve.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: turboserve
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/*
Selector labels. Deliberately just name+instance: a Service or a NetworkPolicy that also
matched on the chart version would stop selecting its own pods during an upgrade.
*/}}
{{- define "turboserve.selectorLabels" -}}
app.kubernetes.io/name: {{ include "turboserve.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "turboserve.gateway.fullname" -}}
{{- printf "%s-gateway" (include "turboserve.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "turboserve.engine.fullname" -}}
{{- printf "%s-engine" (include "turboserve.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "turboserve.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "turboserve.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "turboserve.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}

{{/*
Whether a separate engine workload exists. The two production modes have one -- `vllm` and
`sglang` -- while `mock` runs the mock backend inside the gateway process and `reference`
runs the from-scratch engine there, so in both of those the gateway pod *is* the engine
pod. Everything that has to know where the engine lives -- the chaos loop's pod selector,
the NetworkPolicy, the Grafana dashboard's backend panel -- derives from this one
predicate, which is why adding a second production engine changed no template but the
engine Deployment's own argv.
*/}}
{{- define "turboserve.engine.standalone" -}}
{{- if has .Values.engine.mode (list "vllm" "sglang") -}}true{{- end -}}
{{- end -}}

{{/*
The engine container image.

`engine.image.repository` overrides everything, for a mirror or a digest pin. Left empty
(the default) the image follows `engine.mode`, so switching engines is one value rather
than three that have to be kept consistent -- a vLLM image started with SGLang's argv fails
in a way that looks like a bad flag rather than like the wrong container.

The two halves are overridden independently: a mirror is usually the same image at the same
version under a different host, so an empty `engine.image.tag` keeps the pinned tag of the
mode's own image rather than rendering a reference that ends in a bare colon.
*/}}
{{- define "turboserve.engine.image" -}}
{{- $engine := ternary .Values.engine.sglang.image .Values.engine.vllm.image (eq .Values.engine.mode "sglang") -}}
{{- $repository := default $engine.repository .Values.engine.image.repository -}}
{{- $tag := default $engine.tag .Values.engine.image.tag -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end -}}

{{/* In-cluster base URL of the engine's OpenAI-compatible API. */}}
{{- define "turboserve.engine.url" -}}
{{- printf "http://%s.%s.svc.cluster.local:%d/v1" (include "turboserve.engine.fullname" .) .Release.Namespace (int .Values.engine.service.port) -}}
{{- end -}}

{{/*
The gateway's argv.

`turboserve gateway serve` (src/turboserve/gateway/app.py) takes `--engine` as one of
`config`, `mock`, or the base URL of an OpenAI-compatible server -- there is no `vllm`
literal and no separate `--engine-url`. The mapping from this chart's `engine.mode` to that
flag is therefore:

  mock       --engine mock            the in-process synthetic backend
  reference  --engine config          pools from models.yaml; a `local` backend there is
                                      the from-scratch engine running in this pod
  vllm       --engine <engine URL>    the in-cluster vLLM Service
  sglang     --engine <engine URL>    the in-cluster SGLang Service -- the same flag with
                                      the same value, because the gateway speaks to both
                                      over the same OpenAI-compatible API

Host, port, tenants and models are passed as flags rather than through TURBOSERVE_* env:
`serve` constructs its Settings with those four values explicitly, so an environment
variable for them would be read and then overridden, which is worse than not setting it.
Override the whole line with `gateway.args`.
*/}}
{{- define "turboserve.gateway.engineArg" -}}
{{- if include "turboserve.engine.standalone" . -}}
{{- include "turboserve.engine.url" . -}}
{{- else if eq .Values.engine.mode "reference" -}}
config
{{- else -}}
mock
{{- end -}}
{{- end -}}

{{- define "turboserve.gateway.args" -}}
{{- if .Values.gateway.args -}}
{{- toYaml .Values.gateway.args -}}
{{- else -}}
{{- $args := list "gateway" "serve"
      "--engine" (include "turboserve.gateway.engineArg" .)
      "--model" .Values.engine.model
      "--tenants" (printf "%s/%s" .Values.tenants.mountPath .Values.tenants.key)
      "--models" (printf "%s/%s" .Values.models.mountPath .Values.models.key)
      "--host" "0.0.0.0"
      "--port" (printf "%d" (int .Values.gateway.service.targetPort))
      "--log-level" .Values.gateway.logLevel -}}
{{- if .Values.gateway.requireAuth -}}
{{- $args = append $args "--require-auth" -}}
{{- else -}}
{{- $args = append $args "--no-require-auth" -}}
{{- end -}}
{{- $args = concat $args .Values.gateway.extraArgs -}}
{{- toYaml $args -}}
{{- end -}}
{{- end -}}

{{/*
Environment shared by both gateway lanes.

Deliberately small: everything the server actually needs arrives as a flag (see
turboserve.gateway.args). What is left is context -- the pod's own identity for log
correlation, and anything the operator adds through gateway.env/extraEnv/envFrom.

There is no TURBOSERVE_LANE here. The `lane` label on the gateway's metrics comes from the
backend pool in models.yaml, not from the pod, because a lane is a property of the build
being routed to rather than of the process doing the routing. The pods still carry a
`turboserve.io/lane` label so that kubectl, the PodDisruptionBudgets and the chaos loop can
address one lane at a time.
*/}}
{{- define "turboserve.gateway.env" -}}
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
{{- if and .Values.engine.adapters.enabled (not (include "turboserve.engine.standalone" .)) }}
- name: TURBOSERVE_ENABLE_LORA
  value: "true"
- name: TURBOSERVE_ADAPTERS_DIR
  value: {{ .Values.engine.adapters.mountPath | quote }}
{{- end }}
{{- if .Values.gateway.tracing.endpoint }}
{{/*
Tracing is the one thing here that is environment rather than a flag: `gateway serve` takes
no --otel-* options, because whether a process is traced is a property of where it runs and
not of what it was asked to do. An empty endpoint renders nothing at all, and the gateway
then builds no tracer.
*/}}
- name: TURBOSERVE_OTEL_ENDPOINT
  value: {{ .Values.gateway.tracing.endpoint | quote }}
- name: TURBOSERVE_OTEL_SERVICE_NAME
  value: {{ default (include "turboserve.gateway.fullname" .) .Values.gateway.tracing.serviceName | quote }}
- name: TURBOSERVE_OTEL_SERVICE_NAMESPACE
  value: {{ .Release.Namespace | quote }}
- name: TURBOSERVE_OTEL_SAMPLE_RATIO
  value: {{ .Values.gateway.tracing.sampleRatio | quote }}
{{- end }}
{{- range $name, $value := .Values.gateway.env }}
- name: {{ $name }}
  value: {{ $value | quote }}
{{- end }}
{{- with .Values.gateway.extraEnv }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/*
Replicas for one lane. With the canary disabled the stable lane takes every replica.
With it enabled the weight is turned into a replica split, because two Deployments behind
one ClusterIP Service can only approximate a weight by replica count -- see the note on
canary.weight in values.yaml. Both lanes are floored at 1: a lane with zero pods is not a
canary, it is a rollback, and the controller has its own way of expressing that.
*/}}
{{- define "turboserve.gateway.replicas" -}}
{{- $lane := .lane -}}
{{- $values := .root.Values -}}
{{- $total := int $values.gateway.replicaCount -}}
{{- if not $values.canary.enabled -}}
{{- $total -}}
{{- else -}}
{{- $canary := max 1 (div (mul (int $values.canary.weight) $total) 100) | int -}}
{{- if eq $lane "canary" -}}
{{- $canary -}}
{{- else -}}
{{- max 1 (sub $total $canary) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Volumes mounted into every gateway pod. */}}
{{- define "turboserve.gateway.volumes" -}}
- name: tenants
  secret:
    secretName: {{ default (printf "%s-tenants" (include "turboserve.fullname" .)) .Values.tenants.existingSecret }}
- name: models
  configMap:
    name: {{ default (printf "%s-models" (include "turboserve.fullname" .)) .Values.models.existingConfigMap }}
- name: tmp  {{/* readOnlyRootFilesystem is on: anything that writes needs a real volume */}}
  emptyDir:
    medium: Memory
    sizeLimit: 64Mi
{{- if and .Values.engine.adapters.enabled (not (include "turboserve.engine.standalone" .)) }}
- name: adapters
  {{- if .Values.engine.adapters.persistence.enabled }}
  persistentVolumeClaim:
    claimName: {{ default (printf "%s-adapters" (include "turboserve.fullname" .)) .Values.engine.adapters.persistence.existingClaim }}
  {{- else }}
  emptyDir: {}
  {{- end }}
{{- end }}
{{- with .Values.extraVolumes }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{- define "turboserve.gateway.volumeMounts" -}}
- name: tenants
  mountPath: {{ .Values.tenants.mountPath }}
  readOnly: true
- name: models
  mountPath: {{ .Values.models.mountPath }}
  readOnly: true
- name: tmp
  mountPath: /tmp
{{- if and .Values.engine.adapters.enabled (not (include "turboserve.engine.standalone" .)) }}
- name: adapters
  mountPath: {{ .Values.engine.adapters.mountPath }}
  readOnly: true
{{- end }}
{{- with .Values.extraVolumeMounts }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/*
The adapter-sync init container, shared by the gateway (reference mode) and the engine
(either production mode -- vLLM reads the volume through --lora-modules and SGLang through
--lora-paths). It runs on every pod start rather than as a CronJob so that a pod which comes
up after an adapter was published cannot serve a stale set.
*/}}
{{- define "turboserve.adapters.initContainer" -}}
- name: adapter-sync
  image: {{ .Values.engine.adapters.sync.image | quote }}
  imagePullPolicy: IfNotPresent
  command: {{ toYaml .Values.engine.adapters.sync.command | nindent 4 }}
  args: {{ toYaml .Values.engine.adapters.sync.args | nindent 4 }}
  env:
    - name: ADAPTERS_SOURCE_URL
      value: {{ .Values.engine.adapters.sync.sourceUrl | quote }}
    - name: ADAPTERS_DIR
      value: {{ .Values.engine.adapters.mountPath | quote }}
    {{- with .Values.engine.adapters.sync.env }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
  {{- with .Values.engine.adapters.sync.envFrom }}
  envFrom: {{ toYaml . | nindent 4 }}
  {{- end }}
  resources: {{ toYaml .Values.engine.adapters.sync.resources | nindent 4 }}
  volumeMounts:
    - name: adapters
      mountPath: {{ .Values.engine.adapters.mountPath }}
{{- end -}}

{{/*
Validation that fails `helm template`/`helm lint` instead of producing a cluster that
half-works. Each check is one a reviewer would otherwise have to make by reading values.
*/}}
{{- define "turboserve.validateValues" -}}
{{- $mode := .Values.engine.mode -}}
{{- if not (has $mode (list "mock" "reference" "vllm" "sglang")) -}}
{{- fail (printf "engine.mode must be one of mock|reference|vllm|sglang, got %q" $mode) -}}
{{- end -}}
{{- $quantization := .Values.engine.quantization -}}
{{- if not (has $quantization (list "none" "fp8")) -}}
{{- fail (printf "engine.quantization must be one of none|fp8, got %q" $quantization) -}}
{{- end -}}
{{- if and (eq $quantization "fp8") (not (include "turboserve.engine.standalone" .)) -}}
{{- fail (printf "engine.quantization=fp8 needs a production engine (engine.mode vllm or sglang), not %q: the reference engine runs bf16/fp16 only and the mock backend has no weights at all" $mode) -}}
{{- end -}}
{{- if and .Values.engine.quantizedCheckpoint (ne $quantization "fp8") -}}
{{- fail "engine.quantizedCheckpoint is true but engine.quantization is not fp8: an already-quantized checkpoint is served by asking for the scheme it carries" -}}
{{- end -}}
{{- if and .Values.canary.enabled .Values.canary.argoRollouts.enabled -}}
{{- fail "canary.enabled and canary.argoRollouts.enabled are mutually exclusive: a Rollout owns its own pods, so the two would fight over the same lane labels" -}}
{{- end -}}
{{- if and .Values.canary.enabled (or (lt (int .Values.canary.weight) 1) (gt (int .Values.canary.weight) 99)) -}}
{{- fail (printf "canary.weight must be between 1 and 99, got %v" .Values.canary.weight) -}}
{{- end -}}
{{- if and .Values.engine.adapters.enabled .Values.engine.adapters.sync.enabled (not .Values.engine.adapters.sync.sourceUrl) -}}
{{- fail "engine.adapters.sync.enabled is true but engine.adapters.sync.sourceUrl is empty: set the object-store prefix, or set sync.enabled=false to use a pre-populated volume" -}}
{{- end -}}
{{- if and (eq $mode "vllm") .Values.engine.vllm.speculative.enabled (not .Values.engine.vllm.speculative.model) -}}
{{- fail "engine.vllm.speculative.enabled requires engine.vllm.speculative.model" -}}
{{- end -}}
{{- if and (eq $mode "sglang") .Values.engine.sglang.speculative.enabled -}}
{{- if not .Values.engine.sglang.speculative.algorithm -}}
{{- fail "engine.sglang.speculative.enabled requires engine.sglang.speculative.algorithm (EAGLE or NEXTN)" -}}
{{- end -}}
{{- if and (eq .Values.engine.sglang.speculative.algorithm "EAGLE") (not .Values.engine.sglang.speculative.draftModel) -}}
{{- fail "engine.sglang.speculative.algorithm=EAGLE requires engine.sglang.speculative.draftModel: EAGLE drafts from a checkpoint, unlike NEXTN which uses the target's own head" -}}
{{- end -}}
{{- end -}}
{{- if .Values.gateway.tracing.endpoint -}}
{{- $ratio := float64 .Values.gateway.tracing.sampleRatio -}}
{{- if or (lt $ratio 0.0) (gt $ratio 1.0) -}}
{{- fail (printf "gateway.tracing.sampleRatio must be between 0.0 and 1.0, got %v" .Values.gateway.tracing.sampleRatio) -}}
{{- end -}}
{{- end -}}
{{- if and .Values.gateway.autoscaling.enabled .Values.gateway.pdb.enabled .Values.gateway.pdb.maxUnavailable (eq (int .Values.gateway.autoscaling.minReplicas) 1) -}}
{{- fail "a PodDisruptionBudget with maxUnavailable and autoscaling.minReplicas=1 blocks every voluntary eviction; raise minReplicas or use pdb.minAvailable" -}}
{{- end -}}
{{- end -}}
