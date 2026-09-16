"""``turboserve`` command line entry point.

Subcommands are added by the module that owns them (``serve``, ``bench``, ``chaos``,
``canary``, ``lora``, ``results``); this module owns only the application object, the
global options every subcommand shares, and the two commands that need no engine:
``version`` and ``hwinfo``.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from turboserve._version import __version__
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


if __name__ == "__main__":  # pragma: no cover - module entry point
    app()
