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
