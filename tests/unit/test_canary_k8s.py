"""Unit tests for the cluster driver, the Prometheus reader and the canary CLI.

No cluster and no network are involved: ``kubectl`` is a recording fake and Prometheus is
served by an :class:`httpx.MockTransport`. Every payload below is synthetic -- the counts
and latencies exist to sit on one side of a gate, and none of them was measured.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from turboserve.canary.controller import (
    CanaryConfig,
    CanaryController,
    CanaryState,
    DecisionKind,
    Lane,
    LaneSummary,
)
from turboserve.canary.k8s import (
    ArgoRolloutsDriver,
    CanaryRunner,
    CommandResult,
    Kubectl,
    KubectlError,
    KubeSettings,
    RolloutOutcome,
    WeightedServiceDriver,
    canary_app,
    load_outcomes,
    simulate,
)
from turboserve.canary.prometheus import (
    PrometheusClient,
    PrometheusError,
    PrometheusLaneSource,
    PrometheusSettings,
    escape_label_value,
    format_selector,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "canary.yaml"
CANARY_IMAGE = "ghcr.io/example/turboserve-gateway:v2"

runner = CliRunner()


# ---------------------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------------------


class FakeKubectl:
    """A stand-in for the ``kubectl`` subprocess: records argv, replays canned output."""

    def __init__(self, *, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.calls: list[tuple[str, ...]] = []
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, args: Any, timeout_s: float) -> CommandResult:
        argv = tuple(args)
        self.calls.append(argv)
        return CommandResult(
            args=argv,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )

    def argv_strings(self) -> list[str]:
        """Each recorded command joined back into a single readable string."""
        return [" ".join(call) for call in self.calls]


def _vector(value: float | None) -> dict[str, Any]:
    """A Prometheus instant-vector body holding zero or one sample."""
    if value is None:
        return {"status": "success", "data": {"resultType": "vector", "result": []}}
    rendered = "NaN" if isinstance(value, float) and math.isnan(value) else repr(float(value))
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [{"metric": {}, "value": [1700000000.0, rendered]}],
        },
    }


def prometheus_transport(
    lanes: dict[Lane, dict[str, float | None]],
    *,
    recorder: list[str] | None = None,
) -> httpx.MockTransport:
    """Serve ``/api/v1/query`` from a per-lane table of ``total/errors/ttft/e2e``.

    The handler classifies the query the way a reader would: an ``increase`` of the request
    counter with a status regex is the error count, without it the total, and the two
    ``histogram_quantile`` queries are told apart by the histogram's name. That keeps the
    fixture honest about the queries the source actually emits.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        expr = request.url.params["query"]
        if recorder is not None:
            recorder.append(expr)
        lane: Lane = "canary" if 'lane="canary"' in expr else "stable"
        table = lanes[lane]
        if "increase(" in expr:
            key = "errors" if "status=~" in expr else "total"
        elif "ttft" in expr:
            key = "ttft_s"
        else:
            key = "e2e_s"
        return httpx.Response(200, json=_vector(table.get(key)))

    return httpx.MockTransport(handler)


def make_source(
    lanes: dict[Lane, dict[str, float | None]],
    *,
    settings: PrometheusSettings | None = None,
    recorder: list[str] | None = None,
) -> PrometheusLaneSource:
    """A lane source wired to an in-memory Prometheus."""
    client = PrometheusClient(
        "http://prom.test",
        client=httpx.Client(transport=prometheus_transport(lanes, recorder=recorder)),
    )
    return PrometheusLaneSource(client, settings or PrometheusSettings(), default_lookback_s=60.0)


HEALTHY_LANES: dict[Lane, dict[str, float | None]] = {
    "stable": {"total": 1000.0, "errors": 0.0, "ttft_s": 0.1, "e2e_s": 1.0},
    "canary": {"total": 1000.0, "errors": 0.0, "ttft_s": 0.1, "e2e_s": 1.0},
}
BROKEN_LANES: dict[Lane, dict[str, float | None]] = {
    "stable": {"total": 1000.0, "errors": 0.0, "ttft_s": 0.1, "e2e_s": 1.0},
    "canary": {"total": 1000.0, "errors": 400.0, "ttft_s": 0.1, "e2e_s": 1.0},
}


def fast_policy(**overrides: Any) -> CanaryConfig:
    base: dict[str, Any] = {
        "steps": (10, 100),
        "step_hold_s": 10.0,
        "window_s": 60.0,
        "min_requests": 5,
        "stall_timeout_s": None,
    }
    base.update(overrides)
    return CanaryConfig(**base)


# ---------------------------------------------------------------------------------------
# Kubectl
# ---------------------------------------------------------------------------------------


def patch_payload(argv: tuple[str, ...]) -> dict[str, Any]:
    """The JSON document of a ``kubectl patch`` command, wherever ``-p`` landed in argv."""
    return json.loads(argv[argv.index("-p") + 1])


def test_namespace_is_appended_to_every_namespaced_command() -> None:
    fake = FakeKubectl()
    kubectl = Kubectl(namespace="turboserve", runner=fake)
    kubectl.run(["get", "pods"], mutating=False)
    assert fake.calls[0] == ("kubectl", "get", "pods", "-n", "turboserve")
    assert kubectl.argv(["version"], namespaced=False) == ("kubectl", "version")


def test_a_failing_command_raises_with_its_stderr() -> None:
    fake = FakeKubectl(returncode=1, stderr="rollouts.argoproj.io not found")
    kubectl = Kubectl(runner=fake)
    with pytest.raises(KubectlError, match="not found"):
        kubectl.run(["argo", "rollouts", "abort", "x"], mutating=True)


def test_commands_are_recorded_for_the_audit_trail() -> None:
    fake = FakeKubectl()
    kubectl = Kubectl(namespace="ts", runner=fake)
    kubectl.run(["get", "svc"], mutating=False)
    kubectl.patch("service", "s", {"metadata": {"annotations": {"a": "1"}}})
    assert len(kubectl.commands) == 2
    assert kubectl.commands[0].ok
    assert [c.to_dict()["args"][1] for c in kubectl.commands] == ["get", "patch"]


def test_patch_sends_a_deterministic_merge_document() -> None:
    fake = FakeKubectl()
    kubectl = Kubectl(runner=fake)
    kubectl.patch("service", "svc", {"metadata": {"annotations": {"b": "2", "a": "1"}}})
    argv = fake.calls[0]
    assert argv[:5] == ("kubectl", "patch", "service", "svc", "--type")
    assert patch_payload(argv) == {"metadata": {"annotations": {"a": "1", "b": "2"}}}


# ---------------------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------------------


def test_argo_driver_emits_the_plugin_commands() -> None:
    fake = FakeKubectl()
    driver = ArgoRolloutsDriver(Kubectl(namespace="ts", runner=fake), "gw")
    driver.set_weight(25)
    driver.promote()
    driver.abort()
    assert fake.argv_strings() == [
        "kubectl argo rollouts set weight gw 25 -n ts",
        "kubectl argo rollouts promote gw --full -n ts",
        "kubectl argo rollouts abort gw -n ts",
    ]
    assert driver.describe()["mode"] == "argo"


@pytest.mark.parametrize("percent", [-1, 101, 1000])
def test_weights_outside_the_percentage_range_are_refused(percent: int) -> None:
    driver = ArgoRolloutsDriver(Kubectl(runner=FakeKubectl()), "gw")
    with pytest.raises(ValueError):
        driver.set_weight(percent)


def test_argo_driver_requires_a_rollout_name() -> None:
    with pytest.raises(ValueError):
        ArgoRolloutsDriver(Kubectl(runner=FakeKubectl()), "")


def service_driver(fake: FakeKubectl, **overrides: Any) -> WeightedServiceDriver:
    kwargs: dict[str, Any] = {
        "stable_deployment": "gw-stable",
        "canary_deployment": "gw-canary",
        "stable_service": "gw-stable",
        "canary_service": "gw-canary",
        "container": "gateway",
    }
    kwargs.update(overrides)
    return WeightedServiceDriver(Kubectl(namespace="ts", runner=fake), **kwargs)


def test_service_driver_splits_the_weight_across_both_services() -> None:
    fake = FakeKubectl()
    service_driver(fake).set_weight(25)

    assert len(fake.calls) == 2
    canary_patch = patch_payload(fake.calls[0])["metadata"]["annotations"]
    stable_patch = patch_payload(fake.calls[1])["metadata"]["annotations"]
    assert canary_patch["nginx.ingress.kubernetes.io/canary-weight"] == "25"
    assert canary_patch["turboserve.io/lane-weight"] == "25"
    assert stable_patch["turboserve.io/lane-weight"] == "75"


def test_service_driver_promotion_moves_the_image_then_drains_the_canary() -> None:
    fake = FakeKubectl(stdout=CANARY_IMAGE)
    service_driver(fake).promote()

    joined = fake.argv_strings()
    assert "get deployment gw-canary" in joined[0]
    assert joined[1] == f"kubectl set image deployment/gw-stable gateway={CANARY_IMAGE} -n ts"
    assert joined[2].startswith("kubectl rollout status deployment/gw-stable --timeout=")
    canary_weight = patch_payload(fake.calls[3])["metadata"]["annotations"]
    assert canary_weight["nginx.ingress.kubernetes.io/canary-weight"] == "0"
    assert joined[-1] == "kubectl scale deployment/gw-canary --replicas=0 -n ts"


def test_service_driver_promotion_refuses_a_missing_container() -> None:
    fake = FakeKubectl(stdout="")
    with pytest.raises(KubectlError, match="no container named"):
        service_driver(fake).promote()


def test_service_driver_abort_zeroes_the_weight_and_scales_down() -> None:
    fake = FakeKubectl()
    service_driver(fake).abort()
    annotations = patch_payload(fake.calls[0])["metadata"]["annotations"]
    assert annotations["turboserve.io/lane-weight"] == "0"
    assert fake.argv_strings()[-1] == "kubectl scale deployment/gw-canary --replicas=0 -n ts"


# ---------------------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------------------


def test_dry_run_skips_mutations_but_still_reads_the_cluster() -> None:
    fake = FakeKubectl(stdout=CANARY_IMAGE)
    kubectl = Kubectl(namespace="ts", dry_run=True, runner=fake)
    driver = WeightedServiceDriver(
        kubectl,
        stable_deployment="gw-stable",
        canary_deployment="gw-canary",
        stable_service="gw-stable",
        canary_service="gw-canary",
    )
    driver.promote()

    # The image read executed; nothing that changes the cluster did.
    assert len(fake.calls) == 1
    assert fake.calls[0][1] == "get"
    skipped = [command for command in kubectl.commands if command.skipped]
    assert len(skipped) == len(kubectl.commands) - 1
    assert all(command.ok for command in kubectl.commands)
    # The rehearsal still shows the operator the exact argv of every suppressed command.
    assert any("set image" in " ".join(command.args) for command in skipped)
    assert any("scale" in " ".join(command.args) for command in skipped)


def test_dry_run_argo_rollout_executes_nothing() -> None:
    fake = FakeKubectl()
    kubectl = Kubectl(namespace="ts", dry_run=True, runner=fake)
    driver = ArgoRolloutsDriver(kubectl, "gw")
    driver.set_weight(5)
    driver.promote()
    assert fake.calls == []
    assert [c.skipped for c in kubectl.commands] == [True, True]


# ---------------------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------------------


def test_label_selectors_are_escaped_and_ordered() -> None:
    assert escape_label_value('a"b\\c') == 'a\\"b\\\\c'
    assert format_selector({"b": "2", "a": "1"}) == '{a="1",b="2"}'
    assert format_selector({}, regex={"status": "error|timeout"}) == '{status=~"error|timeout"}'
    assert format_selector({}) == ""


def test_queries_have_the_shape_a_reviewer_expects() -> None:
    source = make_source(HEALTHY_LANES)
    total = source.requests_query("canary")
    errors = source.requests_query("canary", errors_only=True)
    quantile = source.quantile_query("canary", "turboserve_ttft_seconds")

    assert total == 'sum(increase(turboserve_requests_total{lane="canary"}[60s]))'
    assert 'status=~"error|timeout|5xx"' in errors
    assert quantile.startswith("histogram_quantile(0.95, sum by (le) (rate(")
    assert "turboserve_ttft_seconds_bucket" in quantile
    assert quantile.endswith("[60s])))")


def test_extra_labels_are_added_to_every_selector() -> None:
    settings = PrometheusSettings(extra_labels={"service": "turboserve-gateway"})
    source = make_source(HEALTHY_LANES, settings=settings)
    assert 'service="turboserve-gateway"' in source.requests_query("stable")


def test_fetch_converts_seconds_to_milliseconds_and_rounds_counts() -> None:
    lanes: dict[Lane, dict[str, float | None]] = {
        "stable": {"total": 10.4, "errors": 0.0, "ttft_s": 0.25, "e2e_s": 2.5},
        "canary": {"total": 10.4, "errors": 0.6, "ttft_s": 0.25, "e2e_s": 2.5},
    }
    summary = make_source(lanes).fetch("canary")
    assert summary == LaneSummary(
        lane="canary",
        requests=10,
        errors=1,
        p95_ttft_ms=250.0,
        p95_e2e_ms=2500.0,
        window_s=60.0,
    )


def test_fetch_clamps_extrapolated_errors_to_the_request_count() -> None:
    lanes: dict[Lane, dict[str, float | None]] = {
        "stable": {"total": 5.0, "errors": 0.0, "ttft_s": None, "e2e_s": None},
        "canary": {"total": 5.0, "errors": 7.0, "ttft_s": None, "e2e_s": None},
    }
    summary = make_source(lanes).fetch("canary")
    assert summary.errors == summary.requests == 5
    assert summary.error_rate == 1.0


def test_absent_series_and_nan_quantiles_become_none_not_zero() -> None:
    lanes: dict[Lane, dict[str, float | None]] = {
        "stable": {"total": None, "errors": None, "ttft_s": None, "e2e_s": None},
        "canary": {"total": None, "errors": None, "ttft_s": float("nan"), "e2e_s": None},
    }
    summary = make_source(lanes).fetch("canary")
    assert summary.requests == 0
    assert summary.errors == 0
    assert summary.p95_ttft_ms is None
    assert summary.p95_e2e_ms is None


def test_fetch_all_reads_both_lanes() -> None:
    recorded: list[str] = []
    summaries = make_source(HEALTHY_LANES, recorder=recorded).fetch_all()
    assert set(summaries) == {"stable", "canary"}
    assert len(recorded) == 8  # four queries per lane
    assert summaries["canary"].requests == 1000


def test_fetch_rejects_an_unknown_lane() -> None:
    with pytest.raises(ValueError):
        make_source(HEALTHY_LANES).fetch("shadow")  # type: ignore[arg-type]


def _client_returning(response: httpx.Response) -> PrometheusClient:
    return PrometheusClient(
        "http://prom.test",
        client=httpx.Client(transport=httpx.MockTransport(lambda request: response)),
    )


def test_http_errors_and_rejected_queries_raise() -> None:
    with pytest.raises(PrometheusError, match="HTTP 500"):
        _client_returning(httpx.Response(500, text="boom")).query("up")
    body = {"status": "error", "error": "parse error"}
    with pytest.raises(PrometheusError, match="parse error"):
        _client_returning(httpx.Response(200, json=body)).query("up{")


def test_transport_failures_raise_prometheus_error() -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    client = PrometheusClient(
        "http://prom.test", client=httpx.Client(transport=httpx.MockTransport(explode))
    )
    with pytest.raises(PrometheusError, match="no route to host"):
        client.query("up")


def test_a_query_missing_its_aggregation_is_an_error() -> None:
    body = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {"pod": "a"}, "value": [1.0, "1"]},
                {"metric": {"pod": "b"}, "value": [1.0, "2"]},
            ],
        },
    }
    with pytest.raises(PrometheusError, match="aggregation"):
        _client_returning(httpx.Response(200, json=body)).scalar("up")


def test_a_matrix_result_is_refused() -> None:
    body = {"status": "success", "data": {"resultType": "matrix", "result": []}}
    with pytest.raises(PrometheusError, match="instant vector"):
        _client_returning(httpx.Response(200, json=body)).query("up[5m]")


def test_malformed_samples_are_refused() -> None:
    body = {
        "status": "success",
        "data": {"resultType": "vector", "result": [{"metric": {}, "value": [1.0, "abc"]}]},
    }
    with pytest.raises(PrometheusError, match="non-numeric"):
        _client_returning(httpx.Response(200, json=body)).query("up")


def test_settings_load_from_the_repo_config() -> None:
    settings = PrometheusSettings.from_yaml(CONFIG_PATH)
    assert settings.request_metric.startswith("turboserve_")
    assert settings.quantile == 0.95


def test_lookback_defaults_to_the_controller_window() -> None:
    client = PrometheusClient(
        "http://prom.test", client=httpx.Client(transport=prometheus_transport(HEALTHY_LANES))
    )
    source = PrometheusLaneSource(client, PrometheusSettings(), default_lookback_s=123.0)
    assert source.range_selector == "[123s]"
    explicit = PrometheusLaneSource(
        client, PrometheusSettings(lookback_s=7.0), default_lookback_s=123.0
    )
    assert explicit.range_selector == "[7s]"


# ---------------------------------------------------------------------------------------
# The control loop
# ---------------------------------------------------------------------------------------


class StepClock:
    """A clock that advances by a fixed amount every time it is read."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def build_runner(
    lanes: dict[Lane, dict[str, float | None]],
    *,
    fake: FakeKubectl,
    policy: CanaryConfig | None = None,
    **overrides: Any,
) -> tuple[CanaryRunner, CanaryController]:
    controller = CanaryController(policy or fast_policy(), clock=lambda: 0.0)
    driver = ArgoRolloutsDriver(Kubectl(namespace="ts", runner=fake), "gw")
    kwargs: dict[str, Any] = {
        "poll_interval_s": 1.0,
        "sleep": lambda _seconds: None,
        "clock": StepClock(11.0),
    }
    kwargs.update(overrides)
    return CanaryRunner(controller, driver, make_source(lanes), **kwargs), controller


def test_runner_promotes_a_healthy_canary_and_logs_every_command() -> None:
    fake = FakeKubectl()
    loop, controller = build_runner(HEALTHY_LANES, fake=fake)
    report = loop.run("v2")

    assert report.outcome is RolloutOutcome.PROMOTED
    assert report.final_weight == 100
    assert controller.state is CanaryState.PROMOTED
    assert fake.argv_strings() == [
        "kubectl argo rollouts set weight gw 10 -n ts",
        "kubectl argo rollouts set weight gw 100 -n ts",
        "kubectl argo rollouts promote gw --full -n ts",
    ]
    assert len(report.commands) == len(fake.calls)
    assert report.to_dict()["outcome"] == "promoted"


def test_runner_aborts_a_breaching_canary() -> None:
    fake = FakeKubectl()
    loop, controller = build_runner(BROKEN_LANES, fake=fake)
    report = loop.run("v2")

    assert report.outcome is RolloutOutcome.ROLLED_BACK
    assert report.final_weight == 0
    assert controller.state is CanaryState.ROLLED_BACK
    assert fake.argv_strings()[-1] == "kubectl argo rollouts abort gw -n ts"
    assert "error rate" in report.decisions[-1].reason


def test_runner_rolls_back_when_prometheus_keeps_failing() -> None:
    fake = FakeKubectl()
    client = PrometheusClient(
        "http://prom.test",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(503, text="down"))
        ),
    )
    source = PrometheusLaneSource(client, PrometheusSettings(), default_lookback_s=60.0)
    controller = CanaryController(fast_policy(), clock=lambda: 0.0)
    driver = ArgoRolloutsDriver(Kubectl(namespace="ts", runner=fake), "gw")
    loop = CanaryRunner(
        controller,
        driver,
        source,
        poll_interval_s=1.0,
        max_metric_failures=2,
        sleep=lambda _seconds: None,
        clock=StepClock(1.0),
    )
    report = loop.run("v2")

    assert report.outcome is RolloutOutcome.ROLLED_BACK
    assert "metrics unavailable" in report.decisions[-1].reason
    assert fake.argv_strings()[-1] == "kubectl argo rollouts abort gw -n ts"


def test_runner_gives_up_at_its_deadline() -> None:
    fake = FakeKubectl()
    lanes: dict[Lane, dict[str, float | None]] = {
        "stable": {"total": 1000.0, "errors": 0.0, "ttft_s": 0.1, "e2e_s": 1.0},
        "canary": {"total": 0.0, "errors": 0.0, "ttft_s": None, "e2e_s": None},
    }
    loop, controller = build_runner(
        lanes, fake=fake, clock=StepClock(5.0), deadline_s=20.0, poll_interval_s=1.0
    )
    report = loop.run("v2")

    assert report.outcome is RolloutOutcome.TIMED_OUT
    assert "deadline" in report.decisions[-1].reason
    assert controller.state is CanaryState.ROLLED_BACK
    assert fake.argv_strings()[-1] == "kubectl argo rollouts abort gw -n ts"


def test_runner_reports_dry_run_and_executes_nothing() -> None:
    fake = FakeKubectl()
    controller = CanaryController(fast_policy(), clock=lambda: 0.0)
    kubectl = Kubectl(namespace="ts", dry_run=True, runner=fake)
    loop = CanaryRunner(
        controller,
        ArgoRolloutsDriver(kubectl, "gw"),
        make_source(HEALTHY_LANES),
        poll_interval_s=1.0,
        sleep=lambda _seconds: None,
        clock=StepClock(11.0),
    )
    report = loop.run("v2")

    assert report.dry_run is True
    assert report.outcome is RolloutOutcome.PROMOTED
    assert fake.calls == []
    assert all(command.skipped for command in report.commands)


def test_runner_rejects_a_non_positive_poll_interval() -> None:
    with pytest.raises(ValueError):
        CanaryRunner(
            CanaryController(fast_policy()),
            ArgoRolloutsDriver(Kubectl(runner=FakeKubectl()), "gw"),
            poll_interval_s=0.0,
        )


def test_runner_without_a_source_uses_the_controllers_own_windows() -> None:
    """With no metrics backend the loop gates on what the gateway observed in-process."""
    fake = FakeKubectl()
    controller = CanaryController(fast_policy(min_requests=1), clock=lambda: 0.0)
    clock = StepClock(11.0)

    def serve_traffic(_seconds: float) -> None:
        # Stands in for the gateway calling observe() while the loop waits for its poll.
        for lane in ("stable", "canary"):
            controller.observe(lane, True, 100.0, 1000.0, now=clock.now)

    loop = CanaryRunner(
        controller,
        ArgoRolloutsDriver(Kubectl(namespace="ts", runner=fake), "gw"),
        None,
        poll_interval_s=1.0,
        sleep=serve_traffic,
        clock=clock,
    )
    report = loop.run("v2")
    assert report.outcome is RolloutOutcome.PROMOTED
    assert fake.argv_strings()[-1] == "kubectl argo rollouts promote gw --full -n ts"


def test_a_canary_that_never_gets_traffic_is_rolled_back_by_the_stall_timeout() -> None:
    """The safety net that stops a source-less loop from holding for ever."""
    fake = FakeKubectl()
    controller = CanaryController(
        fast_policy(min_requests=5, stall_timeout_s=30.0), clock=lambda: 0.0
    )
    loop = CanaryRunner(
        controller,
        ArgoRolloutsDriver(Kubectl(namespace="ts", runner=fake), "gw"),
        None,
        poll_interval_s=1.0,
        sleep=lambda _seconds: None,
        clock=StepClock(11.0),
    )
    report = loop.run("v2")
    assert report.outcome is RolloutOutcome.ROLLED_BACK
    assert "stalled" in report.decisions[-1].reason


# ---------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------


def test_kube_settings_load_from_the_repo_config_and_build_a_driver() -> None:
    settings = KubeSettings.from_yaml(CONFIG_PATH)
    kubectl = settings.build_kubectl(dry_run=True, runner=FakeKubectl())
    driver = settings.build_driver(kubectl)
    assert isinstance(driver, ArgoRolloutsDriver)
    assert kubectl.namespace == settings.namespace
    assert kubectl.dry_run is True


def test_service_mode_builds_the_deployment_driver() -> None:
    settings = KubeSettings(mode="services")
    driver = settings.build_driver(settings.build_kubectl(runner=FakeKubectl()))
    assert isinstance(driver, WeightedServiceDriver)
    assert driver.describe()["mode"] == "services"


def test_unknown_kubernetes_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        KubeSettings.model_validate({"mode": "argo", "namespce": "typo"})


# ---------------------------------------------------------------------------------------
# Outcome streams and the simulator
# ---------------------------------------------------------------------------------------


def healthy_stream(count: int = 40, *, span: float = 40.0) -> list[dict[str, Any]]:
    """A synthetic stream where both lanes behave identically."""
    events: list[dict[str, Any]] = []
    for index in range(count):
        at = index * span / count
        for lane in ("stable", "canary"):
            events.append({"t": at, "lane": lane, "ok": True, "ttft_ms": 100.0, "e2e_ms": 1000.0})
    return events


def test_load_outcomes_reads_json_jsonl_and_csv(tmp_path: Path) -> None:
    events = healthy_stream(count=2, span=2.0)
    as_json = tmp_path / "a.json"
    as_json.write_text(json.dumps({"events": events}), encoding="utf-8")
    as_jsonl = tmp_path / "a.jsonl"
    as_jsonl.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    as_csv = tmp_path / "a.csv"
    as_csv.write_text(
        "t,lane,ok,ttft_ms,e2e_ms\n"
        + "\n".join(
            f"{event['t']},{event['lane']},{'true' if event['ok'] else 'false'},"
            f"{event['ttft_ms']},{event['e2e_ms']}"
            for event in events
        ),
        encoding="utf-8",
    )
    parsed = [load_outcomes(path) for path in (as_json, as_jsonl, as_csv)]
    assert parsed[0] == parsed[1] == parsed[2]
    assert parsed[0][0].lane in ("stable", "canary")
    assert parsed[0][0].ttft_ms == 100.0


def test_load_outcomes_sorts_by_time_and_defaults_missing_latencies(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(
        '{"t": 5, "lane": "canary", "ok": false}\n{"t": 1, "lane": "stable"}\n',
        encoding="utf-8",
    )
    events = load_outcomes(path)
    assert [event.t for event in events] == [1.0, 5.0]
    assert events[0].ok is True
    assert events[1].ttft_ms is None


def test_load_outcomes_rejects_an_unknown_lane(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"t": 1, "lane": "shadow"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="lane"):
        load_outcomes(path)


def test_simulation_promotes_a_healthy_stream(tmp_path: Path) -> None:
    from turboserve.canary.k8s import OutcomeEvent

    events = [
        OutcomeEvent(t=event["t"], lane=event["lane"], ok=True, ttft_ms=100.0, e2e_ms=1000.0)
        for event in healthy_stream(count=200, span=200.0)
    ]
    controller = CanaryController(
        fast_policy(steps=(10, 50, 100), step_hold_s=20.0, min_requests=5), clock=lambda: 0.0
    )
    report = simulate(controller, events, version="v2", tick_interval_s=5.0)
    assert report.outcome is RolloutOutcome.PROMOTED
    assert report.final_weight == 100
    weights = [decision.weight for decision in report.decisions]
    assert weights == sorted(weights)


def test_simulation_rolls_back_a_failing_stream() -> None:
    from turboserve.canary.k8s import OutcomeEvent

    events = [
        OutcomeEvent(t=float(index), lane="canary", ok=index % 3 != 0, ttft_ms=100.0)
        for index in range(30)
    ]
    controller = CanaryController(fast_policy(min_requests=5), clock=lambda: 0.0)
    report = simulate(controller, events, version="v2", tick_interval_s=5.0)
    assert report.outcome is RolloutOutcome.ROLLED_BACK
    assert report.decisions[-1].kind is DecisionKind.ROLLBACK


def test_simulation_rejects_a_non_positive_tick_interval() -> None:
    with pytest.raises(ValueError):
        simulate(CanaryController(fast_policy()), [], tick_interval_s=0.0)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def write_policy(tmp_path: Path, **overrides: Any) -> Path:
    body = {
        "canary": {
            "steps": [10, 100],
            "step_hold_s": 5,
            "window_s": 600,
            "min_requests": 5,
            "stall_timeout_s": None,
        }
    }
    body["canary"].update(overrides)
    path = tmp_path / "canary.yaml"
    path.write_text(json.dumps(body), encoding="utf-8")  # JSON is valid YAML
    return path


def test_cli_plan_prints_the_policy() -> None:
    result = runner.invoke(canary_app, ["plan", "--config", str(CONFIG_PATH)])
    assert result.exit_code == 0, result.output
    assert "max_error_rate" in result.output


def test_cli_run_replays_a_stream_and_writes_a_report(tmp_path: Path) -> None:
    stream = tmp_path / "outcomes.json"
    stream.write_text(json.dumps(healthy_stream(count=200, span=200.0)), encoding="utf-8")
    out = tmp_path / "reports" / "rollout.json"
    result = runner.invoke(
        canary_app,
        [
            "run",
            "--config",
            str(write_policy(tmp_path)),
            "--outcomes",
            str(stream),
            "--tick-interval",
            "5",
            "--version",
            "v2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["outcome"] == "promoted"
    assert report["final_weight"] == 100
    assert report["decisions"][0]["kind"] == "advance"


def test_cli_run_exits_non_zero_when_the_gate_rejects(tmp_path: Path) -> None:
    events = [
        {"t": float(index), "lane": "canary", "ok": index % 2 == 0, "ttft_ms": 100.0}
        for index in range(30)
    ]
    stream = tmp_path / "outcomes.jsonl"
    stream.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    result = runner.invoke(
        canary_app,
        ["run", "--config", str(write_policy(tmp_path)), "--outcomes", str(stream)],
    )
    assert result.exit_code == 1
    assert "rolled_back" in result.output


def test_cli_run_requires_outcomes_without_kube(tmp_path: Path) -> None:
    result = runner.invoke(canary_app, ["run", "--config", str(write_policy(tmp_path))])
    assert result.exit_code != 0
    assert "--outcomes" in result.output


def test_cli_run_reports_an_unreadable_stream(tmp_path: Path) -> None:
    bad = tmp_path / "outcomes.json"
    bad.write_text("{not json", encoding="utf-8")
    result = runner.invoke(
        canary_app,
        ["run", "--config", str(write_policy(tmp_path)), "--outcomes", str(bad)],
    )
    assert result.exit_code != 0
    assert "cannot read" in result.output


def test_cli_kube_dry_run_drives_the_rollout_without_touching_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run --kube --dry-run`` rehearses a real rollout: fake kubectl, mock Prometheus."""
    import turboserve.canary.k8s as k8s_module

    fake = FakeKubectl()
    monkeypatch.setattr(k8s_module, "_subprocess_runner", fake)

    def fake_client(base_url: str, *, timeout_s: float = 10.0) -> PrometheusClient:
        return PrometheusClient(
            base_url, client=httpx.Client(transport=prometheus_transport(HEALTHY_LANES))
        )

    monkeypatch.setattr(k8s_module, "PrometheusClient", fake_client)

    config = tmp_path / "canary.yaml"
    config.write_text(
        json.dumps(
            {
                "canary": {
                    "steps": [100],
                    "step_hold_s": 0.001,
                    "window_s": 60,
                    "min_requests": 5,
                    "stall_timeout_s": None,
                },
                "kubernetes": {"mode": "argo", "rollout": "gw", "poll_interval_s": 0.001},
                "prometheus": {"base_url": "http://prom.test"},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "rollout.json"
    result = runner.invoke(
        canary_app,
        ["run", "--kube", "--dry-run", "--config", str(config), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert fake.calls == []  # dry run executed no cluster-changing command
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["outcome"] == "promoted"
    assert report["dry_run"] is True
    assert [command["args"][:5] for command in report["commands"]][0] == [
        "kubectl",
        "argo",
        "rollouts",
        "set",
        "weight",
    ]
    assert all(command["skipped"] for command in report["commands"])


def test_cli_abort_is_the_break_glass_path() -> None:
    """``canary abort --dry-run`` names the abort command without running it.

    The shipped policy uses the Argo driver, whose abort is a single mutating command, so
    a dry run touches the cluster not at all -- which is exactly what makes this safe to
    assert here, on a machine with no cluster behind its kubectl.
    """
    result = runner.invoke(
        canary_app, ["abort", "--config", str(CONFIG_PATH), "--dry-run", "--namespace", "demo"]
    )
    assert result.exit_code == 0, result.output
    assert "skipped" in result.output
    assert "abort" in result.output
    assert "demo" in result.output
