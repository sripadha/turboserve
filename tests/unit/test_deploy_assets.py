"""Tests for the deployment assets under ``deploy/`` and ``scripts/vastai/``.

These files are not Python, so nothing else in the suite would notice if one of them
stopped parsing, lost the metric name an alert matches on, or drifted away from the
contract the rest of the repository relies on. `helm lint`, `kubeconform` and the kind
end-to-end job cover the Kubernetes objects themselves and run in CI; what is checked here
is everything those tools cannot see:

* the Grafana dashboard and the Prometheus rules only ever query metric names the gateway
  actually exports (``docs/kubernetes.md`` is the contract),
* the chart's copies of the dashboard and the rules are byte-identical to the canonical
  files, so a fix in one place cannot leave the other behind,
* the error-rate gate of the kind job fails for both of the reasons it must, and
* every shell script is executable and has a shebang.

No helm, kubectl or docker binary is needed, so this runs anywhere the rest of the unit
suite runs.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = REPO_ROOT / "deploy"
CHART = DEPLOY / "helm" / "turboserve"
DASHBOARD = DEPLOY / "grafana" / "dashboards" / "turboserve.json"
RULES = DEPLOY / "prometheus" / "rules.yaml"
ASSERT_SCRIPT = DEPLOY / "kind" / "assert_error_rate.py"

#: The series the gateway exports, taken from ``turboserve.gateway.metrics``: the test
#: below imports them rather than hard-coding the names, so renaming an instrument breaks
#: this test at the source instead of leaving a dashboard quietly querying nothing.
GATEWAY_METRIC_ATTRS = (
    "ttft_seconds",
    "tpot_seconds",
    "e2e_seconds",
    "tokens_total",
    "requests_total",
    "rate_limited_total",
    "cost_usd_total",
    "inflight_requests",
    "queue_depth",
    "backend_up",
    "canary_weight",
)


def _gateway_metric_names() -> set[str]:
    """Exported metric names, read off a live ``GatewayMetrics`` instance.

    prometheus_client mangles a name on its way to the wire -- the namespace and subsystem
    are prefixed and a counter grows a ``_total`` suffix -- so the only trustworthy source
    for what a query must say is the collector itself.
    """
    from prometheus_client import CollectorRegistry

    from turboserve.gateway.metrics import GatewayMetrics

    metrics = GatewayMetrics(registry=CollectorRegistry())
    names: set[str] = set()
    for attribute in GATEWAY_METRIC_ATTRS:
        collector = getattr(metrics, attribute)
        # `_name` is the base name; counters expose it with a `_total` suffix, histograms
        # with `_bucket`/`_sum`/`_count`. `_base_metric` below strips those again.
        names.add(collector._name)
        names.add(f"{collector._name}_total")
    return names


#: Metric-like identifiers in a PromQL expression: a bare name at the start of a selector.
#: Histogram suffixes are stripped before the comparison because `_bucket` and `_count` are
#: exposition artefacts of the base histogram, not separate metrics.
_METRIC_RE = re.compile(r"\b(turboserve[A-Za-z0-9_:]*)")
_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")


def _base_metric(name: str) -> str:
    for suffix in _HISTOGRAM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _dashboard() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _rules() -> dict:
    return yaml.safe_load(RULES.read_text(encoding="utf-8"))


def _known_names() -> set[str]:
    """Everything a PromQL expression here is allowed to name."""
    return _gateway_metric_names() | _recorded_names()


def _recorded_names() -> set[str]:
    return {
        rule["record"]
        for group in _rules()["groups"]
        for rule in group["rules"]
        if "record" in rule
    }


# --- the Grafana dashboard ------------------------------------------------------------


def test_dashboard_is_valid_json_with_a_stable_uid() -> None:
    """The uid is referenced by links and by provisioning, so it must not drift."""
    dashboard = _dashboard()
    assert dashboard["uid"] == "turboserve-gateway"
    assert dashboard["title"]
    assert dashboard["schemaVersion"] >= 39


def test_every_dashboard_panel_has_at_least_one_query() -> None:
    """A panel with no target renders an empty box that looks like an outage."""
    panels = [panel for panel in _dashboard()["panels"] if panel["type"] != "row"]
    assert panels, "the dashboard has no panels"
    for panel in panels:
        targets = panel.get("targets") or []
        assert targets, f"panel {panel['title']!r} has no query"
        for target in targets:
            assert target["expr"].strip(), f"panel {panel['title']!r} has an empty expr"


def test_dashboard_panels_do_not_overlap_or_leave_the_grid() -> None:
    """Grafana's grid is 24 columns wide; a panel past the edge is silently clipped."""
    for panel in _dashboard()["panels"]:
        position = panel["gridPos"]
        assert position["x"] >= 0
        assert position["x"] + position["w"] <= 24, f"{panel['title']!r} runs off the grid"
        assert position["h"] > 0


def test_dashboard_queries_only_use_metrics_the_gateway_exports() -> None:
    known = _known_names()
    for panel in _dashboard()["panels"]:
        for target in panel.get("targets") or []:
            for raw in _METRIC_RE.findall(target["expr"]):
                assert _base_metric(raw) in known, (
                    f"panel {panel['title']!r} queries {raw!r}, which the gateway does not "
                    "export; see the metrics contract in docs/kubernetes.md"
                )


def test_dashboard_has_a_template_variable_per_metric_label() -> None:
    """The panels filter on $tenant/$model/$lane, so those variables have to exist."""
    names = {variable["name"] for variable in _dashboard()["templating"]["list"]}
    assert {"datasource", "tenant", "model", "lane"} <= names


# --- the Prometheus rules ---------------------------------------------------------------


def test_rules_file_holds_nothing_but_groups() -> None:
    """The chart embeds this file as a PrometheusRule's ``spec``, which is exactly
    ``groups``; any other top-level key would render an invalid custom resource."""
    assert set(_rules()) == {"groups"}


def test_every_rule_is_a_record_or_an_alert_with_a_body() -> None:
    for group in _rules()["groups"]:
        assert group["name"].startswith("turboserve.")
        assert group["rules"], f"group {group['name']!r} is empty"
        for rule in group["rules"]:
            assert ("record" in rule) ^ ("alert" in rule)
            assert rule["expr"].strip()


def test_every_alert_names_a_severity_and_a_runbook() -> None:
    """An alert without a severity cannot be routed and one without a runbook link makes
    whoever is paged start from nothing."""
    for group in _rules()["groups"]:
        for rule in group["rules"]:
            if "alert" not in rule:
                continue
            assert rule["labels"]["severity"] in {"critical", "warning", "info"}
            annotations = rule["annotations"]
            assert annotations["summary"]
            assert annotations["description"]
            assert annotations["runbook_url"].endswith("docs/runbook.md")


def test_rule_expressions_only_use_known_metrics() -> None:
    known = _known_names() | {"turboserve"}
    for group in _rules()["groups"]:
        for rule in group["rules"]:
            for raw in _METRIC_RE.findall(rule["expr"]):
                assert _base_metric(raw) in known, f"{rule} references unknown {raw!r}"


def test_recording_rules_are_defined_before_the_alerts_that_use_them() -> None:
    """Prometheus evaluates groups in file order, so an alert in an earlier group than the
    recording rule it reads would evaluate against nothing on the first cycle."""
    groups = _rules()["groups"]
    defined: set[str] = set()
    for group in groups:
        for rule in group["rules"]:
            if "alert" in rule:
                for raw in _METRIC_RE.findall(rule["expr"]):
                    if ":" in raw:
                        assert raw in defined, f"{rule['alert']} uses {raw} before it exists"
        for rule in group["rules"]:
            if "record" in rule:
                defined.add(rule["record"])


# --- the chart's shared files -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        ("files/grafana-dashboard.json", DASHBOARD),
        ("files/prometheus-rules.yaml", RULES),
    ],
)
def test_chart_files_match_the_canonical_assets(name: str, canonical: Path) -> None:
    """Helm's ``.Files.Get`` cannot read outside the chart directory, so the chart needs
    its own copy of each shared asset. A symlink would also work on Linux, but ``helm
    lint``/``template``/``package`` warn on every invocation and a checkout without symlink
    support gets a dangling file, so these are real copies kept honest here instead.
    ``make sync-chart-files`` regenerates them."""
    path = CHART / name
    assert not path.is_symlink(), f"{name} must be a real file, not a symlink"
    assert path.read_bytes() == canonical.read_bytes(), (
        f"{name} has drifted from {canonical.name}; run `make sync-chart-files`"
    )


def test_chart_metadata_is_present() -> None:
    chart = yaml.safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))
    assert chart["apiVersion"] == "v2"
    assert chart["name"] == "turboserve"
    assert chart["version"] and chart["appVersion"]


def test_chart_values_parse_and_declare_every_engine_mode() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    assert values["engine"]["mode"] == "mock"
    # Both production engines are configured in the chart, each with its own pinned image,
    # and neither is the default: a mode that needs a GPU must be asked for.
    for engine, repository in (("vllm", "vllm/vllm-openai"), ("sglang", "lmsysorg/sglang")):
        image = values["engine"][engine]["image"]
        assert image["repository"] == repository
        assert image["tag"] and image["tag"] != "latest", f"{engine} image tag must be pinned"
    # engine.image is the override, so it must not name an engine of its own: a repository
    # here would silently win over engine.mode.
    assert values["engine"]["image"]["repository"] == ""
    assert values["gateway"]["service"]["targetPort"] == 8000
    # The default fleet data must parse as the schemas the gateway actually reads, and the
    # tenants file must carry hashed keys rather than plaintext ones.
    tenants = yaml.safe_load(values["tenants"]["config"])
    assert tenants["version"] == 1
    assert all(
        "keys_sha256" in tenant or tenant["id"] == "default" for tenant in tenants["tenants"]
    )
    models = yaml.safe_load(values["models"]["config"])
    assert models["version"] == 1
    assert models["models"][0]["backends"][0]["lane"] == "stable"
    # A pool named anything other than engine.model would leave engine.model unserved in
    # `config` mode.
    assert models["models"][0]["name"] == values["engine"]["model"]


# --- the kind end-to-end gate -------------------------------------------------------------


def _run_assert(payload: dict, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ASSERT_SCRIPT), "-", *extra],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


def test_error_rate_gate_passes_a_clean_run() -> None:
    result = _run_assert(
        {"scenario": "chaos", "summary": {"num_requests": 1200, "num_failed": 2}},
        "--min-requests",
        "600",
    )
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout


def test_error_rate_gate_fails_a_run_over_the_threshold() -> None:
    result = _run_assert(
        {"scenario": "chaos", "summary": {"num_requests": 1200, "num_failed": 40}},
        "--min-requests",
        "600",
    )
    assert result.returncode == 1
    assert "exceeds" in result.stderr


def test_error_rate_gate_fails_a_run_that_never_happened() -> None:
    """The check that matters most: a run that never reached the gateway has an error rate
    of 0.0 and would pass a threshold-only assertion, turning a broken deployment green."""
    result = _run_assert(
        {"scenario": "chaos", "summary": {"num_requests": 0, "num_failed": 0}},
        "--min-requests",
        "600",
    )
    assert result.returncode == 1
    assert "at least 600" in result.stderr


def test_error_rate_gate_falls_back_to_the_raw_records() -> None:
    """A run interrupted before ``finish()`` has records but no summary."""
    payload = {
        "scenario": "chaos",
        "requests": [{"request_id": str(index), "ok": index > 1} for index in range(200)],
    }
    result = _run_assert(payload, "--min-requests", "100")
    assert result.returncode == 1
    assert "1.0000% exceeds" in result.stderr


# --- the rest of the assets ----------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "docker-compose.yml",
        "deploy/prometheus/prometheus.yml",
        "deploy/kind/kind-config.yaml",
        "deploy/vllm/values-h100.yaml",
        "deploy/sglang/values-h100.yaml",
        "deploy/grafana/provisioning/datasources/prometheus.yaml",
        "deploy/grafana/provisioning/dashboards/turboserve.yaml",
        ".github/workflows/ci.yml",
        ".github/workflows/kind-e2e.yml",
        "src/turboserve/chaos/k8s/podchaos.yaml",
    ],
)
def test_yaml_assets_parse(relative: str) -> None:
    documents = list(yaml.safe_load_all((REPO_ROOT / relative).read_text(encoding="utf-8")))
    assert [document for document in documents if document]


def test_compose_offers_one_profile_per_production_engine() -> None:
    """Each GPU engine is its own profile: a single GPU fits one of them at a time, and the
    gateway in front of them is the same service with a different TURBOSERVE_ENGINE."""
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["services"]["vllm"]["profiles"] == ["gpu"]
    assert compose["services"]["sglang"]["profiles"] == ["sglang"]
    # Neither starts by default: `docker compose up` must stay a no-GPU stack.
    assert "profiles" not in compose["services"]["gateway"]
    for name in ("vllm", "sglang"):
        assert ":" in compose["services"][name]["image"], f"{name} image must be pinned"
        assert not compose["services"][name]["image"].endswith(":latest")


def test_compose_mounts_the_same_rules_file_the_chart_embeds() -> None:
    """Two deployment paths, one alert definition."""
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    mounts = compose["services"]["prometheus"]["volumes"]
    assert any("deploy/prometheus/rules.yaml" in mount for mount in mounts)


@pytest.mark.parametrize(
    "relative",
    [
        "deploy/kind/e2e.sh",
        "deploy/vllm/launch.sh",
        "deploy/sglang/launch.sh",
        "scripts/vastai/provision.sh",
        "scripts/vastai/onstart.sh",
        "scripts/vastai/sync.sh",
        "scripts/vastai/run_remote.sh",
        "scripts/vastai/pull_results.sh",
        "scripts/vastai/destroy.sh",
    ],
)
def test_shell_scripts_are_executable_and_strict(relative: str) -> None:
    """Every script sets `set -Eeuo pipefail`: these run against rented hardware and a
    silently-continuing script there costs money as well as correctness."""
    path = REPO_ROOT / relative
    assert path.stat().st_mode & 0o111, f"{relative} is not executable"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -Eeuo pipefail" in text
