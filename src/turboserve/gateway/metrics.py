"""Prometheus instrumentation for the gateway.

Every series carries the same four dimensions -- ``tenant``, ``model``, ``backend``,
``lane`` -- because every operational question the gateway must answer is a slice along
them: *is the canary lane slower than stable for this model*, *which tenant is spending the
budget*, *did that one backend replica go bad*. Adding a label later is cheap; discovering
after an incident that the data was never split is not.

The instrument set mirrors the metric definitions the benchmark harness uses
(:mod:`turboserve.bench.records`), so a Grafana panel and a benchmark report are measuring
the same quantities: ``ttft_seconds`` from arrival to first output token, ``tpot_seconds``
as the mean inter-token gap after the first token, ``e2e_seconds`` from arrival to finish.

Each :class:`GatewayMetrics` owns a private :class:`~prometheus_client.CollectorRegistry`
rather than the process-global default. Two gateways (or two tests) in one process would
otherwise collide on duplicate time-series names, and a per-app registry makes ``/metrics``
render exactly this app's numbers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from turboserve.gateway.limits import LimitName

logger = logging.getLogger(__name__)

__all__ = [
    "LATENCY_BUCKETS",
    "REQUEST_LABELS",
    "TPOT_BUCKETS",
    "GatewayMetrics",
    "RequestStatus",
]

#: Buckets for arrival-to-first-token and end-to-end latency, in seconds. Spread
#: logarithmically from a few milliseconds (a cache hit on a short prompt) to a minute (a
#: long generation under load) so that a percentile read off the histogram keeps roughly
#: constant relative resolution across that range.
LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    60.0,
)

#: Buckets for the mean inter-token gap, in seconds. A decode step is orders of magnitude
#: shorter than a whole request, so it needs its own, finer scale.
TPOT_BUCKETS: Final[tuple[float, ...]] = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
)

#: The four dimensions shared by every request-scoped series.
REQUEST_LABELS: Final[tuple[str, ...]] = ("tenant", "model", "backend", "lane")

#: Terminal state of a request, the ``status`` label of ``requests_total``.
#: ``ok`` -- completed; ``error`` -- backend or stream failure; ``rate_limited`` -- a quota
#: refused it; ``unauthorized``/``forbidden`` -- auth refused it; ``bad_request`` -- the
#: body was invalid; ``cancelled`` -- the client disconnected mid-stream.
RequestStatus = str


class GatewayMetrics:
    """The gateway's Prometheus instruments, bound to one registry.

    All methods are cheap and non-raising: instrumentation must never be the reason a
    request fails, so label values are coerced to strings and unknown ones are accepted.
    """

    __slots__ = (
        "_namespace",
        "backend_up",
        "canary_weight",
        "cost_usd_total",
        "e2e_seconds",
        "inflight_requests",
        "queue_depth",
        "rate_limited_total",
        "registry",
        "requests_total",
        "tokens_total",
        "tpot_seconds",
        "ttft_seconds",
    )

    def __init__(
        self,
        *,
        registry: CollectorRegistry | None = None,
        namespace: str = "turboserve",
        include_process_metrics: bool = False,
    ) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self._namespace = namespace
        labels = list(REQUEST_LABELS)

        self.ttft_seconds = Histogram(
            "gateway_ttft_seconds",
            "Time from request arrival to the first output token.",
            labelnames=labels,
            buckets=LATENCY_BUCKETS,
            namespace=namespace,
            registry=self.registry,
        )
        self.tpot_seconds = Histogram(
            "gateway_tpot_seconds",
            "Mean time per output token after the first one.",
            labelnames=labels,
            buckets=TPOT_BUCKETS,
            namespace=namespace,
            registry=self.registry,
        )
        self.e2e_seconds = Histogram(
            "gateway_e2e_seconds",
            "Time from request arrival to the final token.",
            labelnames=labels,
            buckets=LATENCY_BUCKETS,
            namespace=namespace,
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "gateway_tokens_total",
            "Tokens accounted to a tenant, split into prompt and completion.",
            labelnames=[*labels, "kind"],
            namespace=namespace,
            registry=self.registry,
        )
        self.requests_total = Counter(
            "gateway_requests_total",
            "Requests by terminal status.",
            labelnames=[*labels, "status"],
            namespace=namespace,
            registry=self.registry,
        )
        self.rate_limited_total = Counter(
            "gateway_rate_limited_total",
            "Requests refused by a tenant quota, by which quota refused them.",
            labelnames=["tenant", "limit"],
            namespace=namespace,
            registry=self.registry,
        )
        self.cost_usd_total = Counter(
            "gateway_cost_usd_total",
            "Attributed spend from the configured price table in configs/models.yaml. "
            "Configured list prices, not a measurement.",
            labelnames=["tenant", "model"],
            namespace=namespace,
            registry=self.registry,
        )
        self.inflight_requests = Gauge(
            "gateway_inflight_requests",
            "Requests currently being streamed.",
            labelnames=["tenant", "model"],
            namespace=namespace,
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "gateway_queue_depth",
            "Requests admitted and waiting on a backend; the Kubernetes HPA scales on this.",
            labelnames=["model"],
            namespace=namespace,
            registry=self.registry,
        )
        self.backend_up = Gauge(
            "gateway_backend_up",
            "Last health probe result for a backend in a model pool (1 healthy, 0 not).",
            labelnames=["backend", "model", "lane"],
            namespace=namespace,
            registry=self.registry,
        )
        self.canary_weight = Gauge(
            "gateway_canary_weight",
            "Share of traffic the canary lane is currently taking, in percent.",
            labelnames=["model"],
            namespace=namespace,
            registry=self.registry,
        )
        if include_process_metrics:
            self._add_process_collectors()

    def _add_process_collectors(self) -> None:
        """Attach the standard process/platform collectors to our private registry.

        Optional because the collectors are process-wide: registering them in two apps in
        one process would double-count, so only the real server switches them on.
        """
        from prometheus_client import PlatformCollector, ProcessCollector

        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)

    @property
    def namespace(self) -> str:
        """Metric-name prefix, ``turboserve`` by default."""
        return self._namespace

    # -- recording --------------------------------------------------------------------

    def record_request(
        self,
        *,
        tenant: str,
        model: str,
        backend: str = "",
        lane: str = "stable",
        status: RequestStatus = "ok",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttft_s: float | None = None,
        tpot_s: float | None = None,
        e2e_s: float | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Record one finished request across every instrument it touches.

        Latencies are only observed when they exist: a request refused by auth has no TTFT,
        and writing a zero would drag the percentile down and hide real regressions.
        """
        keys = (tenant, model, backend, lane)
        self.requests_total.labels(*keys, status).inc()
        if prompt_tokens:
            self.tokens_total.labels(*keys, "prompt").inc(prompt_tokens)
        if completion_tokens:
            self.tokens_total.labels(*keys, "completion").inc(completion_tokens)
        if ttft_s is not None and ttft_s >= 0.0:
            self.ttft_seconds.labels(*keys).observe(ttft_s)
        if tpot_s is not None and tpot_s >= 0.0:
            self.tpot_seconds.labels(*keys).observe(tpot_s)
        if e2e_s is not None and e2e_s >= 0.0:
            self.e2e_seconds.labels(*keys).observe(e2e_s)
        if cost_usd:
            self.cost_usd_total.labels(tenant, model).inc(cost_usd)

    def record_rate_limited(self, *, tenant: str, limit: LimitName | str) -> None:
        """Count a 429, labelled by the quota that produced it."""
        self.rate_limited_total.labels(tenant, str(limit)).inc()

    def inc_inflight(self, *, tenant: str, model: str) -> None:
        """A request started streaming."""
        self.inflight_requests.labels(tenant, model).inc()

    def dec_inflight(self, *, tenant: str, model: str) -> None:
        """A request stopped streaming, successfully or not."""
        self.inflight_requests.labels(tenant, model).dec()

    def set_queue_depth(self, *, model: str, value: float) -> None:
        """Publish the pool's queue depth; the HPA custom metric reads this series."""
        self.queue_depth.labels(model).set(value)

    def set_backend_up(self, *, backend: str, model: str, lane: str, up: bool) -> None:
        """Publish the outcome of a backend health probe."""
        self.backend_up.labels(backend, model, lane).set(1.0 if up else 0.0)

    def set_canary_weight(self, *, model: str, weight: float) -> None:
        """Publish the canary lane's current traffic share, in percent."""
        self.canary_weight.labels(model).set(weight)

    # -- exposition -------------------------------------------------------------------

    def render(self) -> tuple[bytes, str]:
        """Return ``(body, content_type)`` for the ``GET /metrics`` route."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    def sample_value(self, name: str, **labels: str) -> float | None:
        """Read one sample back, by full metric name and labels.

        Exists for tests and for the config-check command: asserting on a rendered text
        exposition is brittle, while this reads the value the collector holds.
        """
        return self.registry.get_sample_value(name, labels or None)
