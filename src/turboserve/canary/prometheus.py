"""Read lane health out of Prometheus so the gate sees the fleet, not one process.

In the in-process path the controller counts requests itself. In Kubernetes it cannot:
the canary is a set of pods behind a Service, and the only place that knows how *all* of
them are doing is the Prometheus that scrapes the gateway's ``/metrics``. This module turns
four PromQL queries into the same :class:`~turboserve.canary.controller.LaneSummary` the
in-process windows produce, so the gate code downstream is byte-for-byte the same.

The queries, over the metrics the gateway exports:

======================  ================================================================
requests in the window  ``sum(increase(turboserve_requests_total{lane=…}[W]))``
errors in the window    the same, restricted to the failing ``status`` values
p95 TTFT                ``histogram_quantile(0.95, sum by (le) (rate(B{lane=…}[W])))``
                        with ``B`` = ``turboserve_ttft_seconds_bucket``
p95 end-to-end          the same over ``turboserve_e2e_seconds_bucket``
======================  ================================================================

Three deliberate choices:

* ``increase`` over a range, not the raw counter, because a pod restart resets the counter
  and ``increase`` is the function that accounts for that.
* ``sum by (le)`` *inside* ``histogram_quantile``, because the quantile must be taken over
  the merged bucket counts of every replica; taking it per pod and averaging is the classic
  way to produce a p95 that is not a p95.
* Prometheus histograms are in seconds, the controller's gates are in milliseconds, so the
  conversion happens here, once.

A ``NaN`` from ``histogram_quantile`` (no observations in the range) becomes ``None``
rather than zero: an absent latency must not read as a fast one.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field

from turboserve.canary.controller import LANES, Lane, LaneSummary

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "PrometheusClient",
    "PrometheusError",
    "PrometheusLaneSource",
    "PrometheusSettings",
    "VectorSample",
    "escape_label_value",
    "format_selector",
]


class PrometheusError(RuntimeError):
    """Prometheus was unreachable, unhappy, or answered something unusable."""


@dataclass(frozen=True, slots=True)
class VectorSample:
    """One element of an instant-vector result: its labels, its value, its timestamp."""

    metric: dict[str, str]
    value: float
    timestamp: float


def escape_label_value(value: str) -> str:
    """Escape a PromQL label value (backslash, double quote, newline).

    Interpolating an unescaped tenant or service name into a query would at best break the
    query and at worst change which series it selects, so every selector goes through here.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def format_selector(labels: Mapping[str, str], *, regex: Mapping[str, str] | None = None) -> str:
    """Render ``{a="1",b=~"x|y"}`` from exact-match and regex-match label maps.

    Labels are emitted in sorted order so that two equal selectors are the same string,
    which makes the queries comparable in logs and assertable in tests.
    """
    parts = [f'{name}="{escape_label_value(value)}"' for name, value in sorted(labels.items())]
    parts += [
        f'{name}=~"{escape_label_value(value)}"' for name, value in sorted((regex or {}).items())
    ]
    return "{" + ",".join(parts) + "}" if parts else ""


class PrometheusSettings(BaseModel):
    """Where Prometheus is and what the gateway's metrics are called.

    Metric names are configurable rather than hard-coded because the same controller is
    meant to gate a vLLM deployment fronted by this gateway *and* a deployment whose
    metrics were relabelled by an operator's recording rules.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = "http://prometheus-operated.monitoring.svc:9090"
    lookback_s: float | None = Field(default=None, gt=0.0)
    """Range width for every query; ``None`` means "use the controller's ``window_s``"."""

    timeout_s: float = Field(default=10.0, gt=0.0)
    request_metric: str = "turboserve_requests_total"
    ttft_histogram: str = "turboserve_ttft_seconds"
    e2e_histogram: str = "turboserve_e2e_seconds"
    lane_label: str = "lane"
    status_label: str = "status"
    error_statuses: tuple[str, ...] = ("error", "timeout", "5xx")
    """``status`` label values counted as failures; joined into one regex selector."""

    extra_labels: dict[str, str] = Field(default_factory=dict)
    """Additional exact-match labels, e.g. ``{"service": "turboserve-gateway"}``."""

    quantile: float = Field(default=0.95, gt=0.0, lt=1.0)

    @classmethod
    def from_yaml(cls, path: str | Path, *, key: str | None = "prometheus") -> PrometheusSettings:
        """Load the ``prometheus:`` section of ``configs/canary.yaml``."""
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(document, dict):
            raise ValueError(f"{path}: expected a YAML mapping, got {type(document).__name__}")
        section: Any = document
        if key is not None:
            section = document.get(key, {})
        if not isinstance(section, dict):
            raise ValueError(f"{path}: section {key!r} must be a mapping")
        return cls.model_validate(section)


class PrometheusClient:
    """A minimal, synchronous ``/api/v1/query`` client.

    Synchronous on purpose: the Kubernetes driver's loop is a sequence of blocking
    ``kubectl`` subprocesses, and an event loop around four queries every fifteen seconds
    would buy nothing. An ``httpx.Client`` may be injected, which is how the tests serve the
    whole API from a :class:`httpx.MockTransport` with no socket involved.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 10.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_s, headers=dict(headers or {}))

    def __enter__(self) -> PrometheusClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client if this object created it."""
        if self._owns_client:
            self._client.close()

    def query(self, expr: str, *, at: float | None = None) -> list[VectorSample]:
        """Run an instant query and return its instant vector.

        Raises :class:`PrometheusError` for transport failures, non-2xx responses, a
        ``status != "success"`` body, or a result type other than ``vector``/``scalar``.
        A gate that cannot read its metrics must fail loudly; silently returning "no data"
        would make the controller hold for ever on a broken monitoring stack.
        """
        params: dict[str, str] = {"query": expr}
        if at is not None:
            params["time"] = repr(float(at))
        url = f"{self.base_url}/api/v1/query"
        logger.debug("prometheus query %s", expr)
        try:
            response = self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise PrometheusError(f"querying {url}: {exc}") from exc
        if response.status_code >= 400:
            raise PrometheusError(
                f"{url} returned HTTP {response.status_code} for {expr!r}: {response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PrometheusError(f"{url} returned a non-JSON body for {expr!r}") from exc
        if not isinstance(payload, dict) or payload.get("status") != "success":
            detail = payload.get("error") if isinstance(payload, dict) else None
            raise PrometheusError(f"prometheus rejected {expr!r}: {detail or payload}")
        data = payload.get("data") or {}
        result_type = data.get("resultType")
        result = data.get("result") or []
        if result_type == "scalar":
            return [_parse_sample({"metric": {}, "value": result})]
        if result_type != "vector":
            raise PrometheusError(f"expected an instant vector for {expr!r}, got {result_type!r}")
        return [_parse_sample(item) for item in result]

    def scalar(self, expr: str, *, at: float | None = None) -> float | None:
        """Run a query expected to yield at most one sample and return its value.

        ``None`` for an empty vector (the series does not exist yet) or a ``NaN`` value
        (a quantile over an empty range). More than one sample is an error: it means the
        query forgot an aggregation and the caller would have silently used an arbitrary
        one of them.
        """
        samples = self.query(expr, at=at)
        if not samples:
            return None
        if len(samples) > 1:
            raise PrometheusError(
                f"expected at most one sample for {expr!r}, got {len(samples)}; "
                "the query is missing an aggregation"
            )
        value = samples[0].value
        return None if math.isnan(value) else value


def _parse_sample(item: Any) -> VectorSample:
    """Turn one ``{"metric": {...}, "value": [ts, "1.5"]}`` object into a sample."""
    if not isinstance(item, dict):
        raise PrometheusError(f"malformed result element: {item!r}")
    pair = item.get("value")
    if not isinstance(pair, list | tuple) or len(pair) != 2:
        raise PrometheusError(f"malformed sample value: {item!r}")
    try:
        timestamp = float(pair[0])
        value = float(pair[1])
    except (TypeError, ValueError) as exc:
        raise PrometheusError(f"non-numeric sample value: {item!r}") from exc
    metric = item.get("metric") or {}
    if not isinstance(metric, dict):
        raise PrometheusError(f"malformed sample labels: {item!r}")
    return VectorSample(
        metric={str(k): str(v) for k, v in metric.items()},
        value=value,
        timestamp=timestamp,
    )


class PrometheusLaneSource:
    """Builds the four queries for a lane and assembles a :class:`LaneSummary`."""

    def __init__(
        self,
        client: PrometheusClient,
        settings: PrometheusSettings | None = None,
        *,
        default_lookback_s: float = 300.0,
    ) -> None:
        self.client = client
        self.settings = settings or PrometheusSettings()
        lookback = self.settings.lookback_s
        self.lookback_s = float(lookback if lookback is not None else default_lookback_s)
        if self.lookback_s <= 0:
            raise ValueError(f"lookback_s must be > 0, got {self.lookback_s}")

    # -- query construction -------------------------------------------------------------

    @property
    def range_selector(self) -> str:
        """The ``[300s]`` suffix every query shares, rendered from the lookback."""
        return f"[{self.lookback_s:g}s]"

    def selector(self, lane: Lane, *, errors_only: bool = False) -> str:
        """Label selector for one lane, optionally restricted to failing statuses."""
        labels = dict(self.settings.extra_labels)
        labels[self.settings.lane_label] = lane
        regex: dict[str, str] = {}
        if errors_only:
            regex[self.settings.status_label] = "|".join(self.settings.error_statuses)
        return format_selector(labels, regex=regex)

    def requests_query(self, lane: Lane, *, errors_only: bool = False) -> str:
        """``sum(increase(<counter><selector><range>))``."""
        selector = self.selector(lane, errors_only=errors_only)
        return f"sum(increase({self.settings.request_metric}{selector}{self.range_selector}))"

    def quantile_query(self, lane: Lane, histogram: str) -> str:
        """``histogram_quantile(q, sum by (le) (rate(<histogram>_bucket…)))``."""
        selector = self.selector(lane)
        inner = f"sum by (le) (rate({histogram}_bucket{selector}{self.range_selector}))"
        return f"histogram_quantile({self.settings.quantile:g}, {inner})"

    # -- fetching -----------------------------------------------------------------------

    def fetch(self, lane: Lane, *, at: float | None = None) -> LaneSummary:
        """Query the four series for ``lane`` and return the summary the gates consume.

        Counts are rounded because ``increase()`` extrapolates to fractional values at the
        range boundaries, and the gate compares them against an integer ``min_requests``.
        Errors are clamped to the request count so that boundary extrapolation can never
        manufacture an error rate above one.
        """
        if lane not in LANES:
            raise ValueError(f"lane must be one of {LANES}, got {lane!r}")
        total = self.client.scalar(self.requests_query(lane), at=at) or 0.0
        errors = self.client.scalar(self.requests_query(lane, errors_only=True), at=at) or 0.0
        p95_ttft_s = self.client.scalar(
            self.quantile_query(lane, self.settings.ttft_histogram), at=at
        )
        p95_e2e_s = self.client.scalar(
            self.quantile_query(lane, self.settings.e2e_histogram), at=at
        )
        requests = max(int(round(total)), 0)
        return LaneSummary(
            lane=lane,
            requests=requests,
            errors=min(max(int(round(errors)), 0), requests),
            p95_ttft_ms=None if p95_ttft_s is None else p95_ttft_s * 1000.0,
            p95_e2e_ms=None if p95_e2e_s is None else p95_e2e_s * 1000.0,
            window_s=self.lookback_s,
        )

    def fetch_all(self, *, at: float | None = None) -> dict[Lane, LaneSummary]:
        """Both lanes, fetched at the same evaluation time for a fair comparison."""
        return {lane: self.fetch(lane, at=at) for lane in LANES}
