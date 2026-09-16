"""``turboserve`` command line entry point.

This module owns the application object, the global options every subcommand shares, the
two commands that need no engine (``version`` and ``hwinfo``), and the wiring that mounts
each module's own typer application:

===========================  ==============================================================
``turboserve serve``         the gateway's ``serve`` command, promoted to the top level
``turboserve gateway ...``   serve, hash a key, check a model/tenant configuration
``turboserve engine ...``    run the reference engine directly (generate, kv-size)
``turboserve bench ...``     load generator, the five scenarios, report rendering
``turboserve results ...``   render or show the tables under ``results/``
``turboserve canary ...``    SLO-gated progressive delivery
``turboserve chaos ...``     fault schedules and the chaos harness
``turboserve lora ...``      train the synthetic multi-tenant LoRA adapters
===========================  ==============================================================

Sub-applications are imported eagerly. The alternative -- resolving each one on first use --
would keep ``turboserve hwinfo`` cheap, but it would also make ``turboserve --help`` unable
to list what the installation can actually do, which is the one thing a CLI's help screen
has to get right. ``hwinfo`` already imports torch to report the driver and device, so
there is no cheap path to protect.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from turboserve._version import __version__
from turboserve.bench.cli import bench_app
from turboserve.bench.report import results_app
from turboserve.canary.k8s import canary_app
from turboserve.chaos.harness import chaos_app
from turboserve.engine.lora.make_adapters import lora_app
from turboserve.engine.runtime.engine import engine_app
from turboserve.gateway.app import gateway_app, serve_command
from turboserve.hwinfo import collect
from turboserve.logging_utils import configure_logging

app = typer.Typer(
    name="turboserve",
    help="Multi-tenant LLM inference: reference engine, gateway, canary and chaos tooling.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            envvar="TURBOSERVE_LOG_LEVEL",
            help="Root log level for this invocation.",
        ),
    ] = "INFO",
) -> None:
    """Configure logging before any subcommand runs."""
    configure_logging(log_level)


@app.command()
def version() -> None:
    """Print the installed turboserve version."""
    typer.echo(f"turboserve {__version__}")


@app.command()
def hwinfo(
    indent: Annotated[
        int,
        typer.Option("--indent", min=0, help="JSON indentation; 0 prints one line."),
    ] = 2,
) -> None:
    """Print the hardware and software record embedded in every benchmark result."""
    typer.echo(json.dumps(collect(), indent=indent or None, sort_keys=True))


# ``turboserve serve`` is the spec's headline command and is deliberately the *same*
# function object as ``turboserve gateway serve`` rather than a second implementation of
# it: one set of flags, one set of defaults, one place where the engine selection lives.
app.command("serve")(serve_command)

app.add_typer(gateway_app, name="gateway")
app.add_typer(engine_app, name="engine")
app.add_typer(bench_app, name="bench")
app.add_typer(results_app, name="results")
app.add_typer(canary_app, name="canary")
app.add_typer(chaos_app, name="chaos")
app.add_typer(lora_app, name="lora")


if __name__ == "__main__":  # pragma: no cover - module entry point
    app()
