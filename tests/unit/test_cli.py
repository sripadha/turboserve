"""CLI surface tests: ``turboserve version`` and ``turboserve hwinfo``.

These run the typer app in-process through ``CliRunner`` so they cost milliseconds and
still exercise the real argument parsing, the logging callback and the hwinfo collector.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from turboserve import __version__
from turboserve.cli import app

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # `no_args_is_help=True` makes typer exit with 2 after printing the help screen.
    assert result.exit_code == 2
    assert "hwinfo" in result.output
    assert "version" in result.output


def test_hwinfo_emits_json_with_the_documented_keys() -> None:
    result = runner.invoke(app, ["hwinfo"])
    assert result.exit_code == 0, result.output
    info = json.loads(result.output)
    assert set(info) >= {
        "schema_version",
        "collected_at",
        "host",
        "python",
        "gpu_name",
        "nvidia_smi",
        "torch",
        "triton_version",
        "git",
        "package_version",
    }
    assert info["package_version"] == __version__
    assert set(info["host"]) >= {"hostname", "platform", "cpu_count", "memory_total_bytes"}
    assert info["torch"]["available"] is True
    assert isinstance(info["torch"]["cuda_available"], bool)


def test_hwinfo_indent_zero_is_one_line() -> None:
    result = runner.invoke(app, ["hwinfo", "--indent", "0"])
    assert result.exit_code == 0, result.output
    payload = result.output.strip()
    assert "\n" not in payload
    assert json.loads(payload)["schema_version"] >= 1


# -- sub-application wiring --------------------------------------------------------------
#
# `cli.py` is the only place the module groups' typer apps are mounted, and nothing else
# imports them together. Without these tests a module could stop exporting its app, or be
# renamed, and the failure would surface as a missing command in a deployment manifest.

#: Every command the specification's CLI section names, and the ones the manifests call.
EXPECTED_COMMANDS: tuple[str, ...] = (
    "version",
    "hwinfo",
    "serve",
    "gateway",
    "engine",
    "bench",
    "results",
    "canary",
    "chaos",
    "lora",
)

#: `<group>: <subcommands>` that other parts of the repository invoke by name --
#: docker-compose.yml, the Helm chart, deploy/kind/e2e.sh, scripts/run_all_benchmarks.sh
#: and the runbook.
EXPECTED_SUBCOMMANDS: dict[str, tuple[str, ...]] = {
    "gateway": ("serve", "hash-key", "config-check"),
    "engine": ("generate", "kv-size"),
    "bench": (
        "loadgen",
        "naive-vs-cb",
        "prefix-cache",
        "spec-decode",
        "multi-lora",
        "chaos",
        "render",
        "profiles",
    ),
    "results": ("render", "show"),
    "canary": ("plan", "run", "abort"),
    "chaos": ("run", "plan"),
    "lora": ("make-adapters",),
}


def test_every_module_app_is_mounted() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in EXPECTED_COMMANDS:
        assert command in result.output, f"{command} is not on the root app"


def test_each_group_lists_the_commands_the_repository_calls() -> None:
    for group, subcommands in EXPECTED_SUBCOMMANDS.items():
        result = runner.invoke(app, [group, "--help"])
        assert result.exit_code == 0, f"{group} --help failed: {result.output}"
        for sub in subcommands:
            assert sub in result.output, f"{group} has no {sub} command"


def test_serve_is_the_gateways_serve_command() -> None:
    """``turboserve serve`` must be the same flags as ``turboserve gateway serve``.

    They are deliberately the same function object; a second implementation would drift.
    """
    top = runner.invoke(app, ["serve", "--help"])
    nested = runner.invoke(app, ["gateway", "serve", "--help"])
    assert top.exit_code == 0, top.output
    assert nested.exit_code == 0, nested.output
    for flag in ("--engine", "--model", "--tenants", "--models", "--host", "--port"):
        assert flag in top.output, f"turboserve serve has no {flag}"
        assert flag in nested.output
