"""``turboserve bench``: the benchmark sub-application.

One typer app that gathers every scenario plus the ad-hoc load generator and the report
renderer, so that the whole measurement surface is discoverable from ``turboserve bench
--help`` instead of being spread over several entry points.

Two scenarios -- ``spec-decode`` and ``multi-lora`` -- live in modules owned by other parts
of the engine, and they are registered *optionally*: the module is imported when this app
is built, and a module that is not present is logged at debug level and skipped rather than
breaking every other benchmark command. That is not a placeholder for missing work; it is
the same tolerance :func:`turboserve.gateway.backends.load_builtin_backends` applies to its
own registry, and it means an installation that trims the package still has a working
``bench loadgen``.

A scenario module is discovered by looking for, in order, a ``typer.Typer`` it exports or a
command function; the names tried are listed in :data:`OPTIONAL_SCENARIOS`. A Typer object
is mounted as a sub-command group, a function is registered as a single command.
"""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from turboserve.bench.scenarios.chaos import chaos_command
from turboserve.bench.scenarios.loadgen_cli import loadgen_command
from turboserve.bench.scenarios.naive_vs_cb import naive_vs_cb_command
from turboserve.bench.scenarios.prefix_cache import prefix_cache_command

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = ["OPTIONAL_SCENARIOS", "bench_app", "register_optional_scenario"]

#: Scenarios owned by another module group, registered when importable.
#: ``(cli name, module, candidate attribute names)``. The first attribute that exists wins;
#: a :class:`typer.Typer` is mounted with ``add_typer`` and a callable with ``command``.
OPTIONAL_SCENARIOS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "spec-decode",
        "turboserve.bench.scenarios.spec_decode",
        ("spec_decode_app", "spec_decode_command", "app", "command", "main", "run"),
    ),
    (
        "multi-lora",
        "turboserve.bench.scenarios.multi_lora",
        ("multi_lora_app", "multi_lora_command", "app", "command", "main", "run"),
    ),
)

bench_app = typer.Typer(
    name="bench",
    help="Load generation, benchmark scenarios and report rendering.",
    no_args_is_help=True,
    add_completion=False,
)

bench_app.command("loadgen")(loadgen_command)
bench_app.command("naive-vs-cb")(naive_vs_cb_command)
bench_app.command("prefix-cache")(prefix_cache_command)
bench_app.command("chaos")(chaos_command)


def register_optional_scenario(
    app: typer.Typer, cli_name: str, module_name: str, candidates: Sequence[str]
) -> bool:
    """Attach a scenario that may not be installed; returns whether it was attached.

    An absent module is a skip. An import error from a module that *is* present is also a
    skip, but logged as a warning with the traceback: a scenario whose own imports are
    broken is a real problem, and swallowing it silently would make ``bench --help`` lie
    about what this build can do.
    """
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or (exc.name or "") in module_name:
            logger.debug("scenario %s is not installed (%s)", cli_name, exc)
        else:
            logger.warning("scenario %s could not be imported: %s", cli_name, exc)
        return False
    except ImportError:
        logger.warning("scenario %s could not be imported", cli_name, exc_info=True)
        return False
    entry: Any = next(
        (getattr(module, name) for name in candidates if hasattr(module, name)),
        None,
    )
    if entry is None:
        logger.warning(
            "scenario module %s exports none of %s; not registering %s",
            module_name,
            ", ".join(candidates),
            cli_name,
        )
        return False
    if isinstance(entry, typer.Typer):
        app.add_typer(entry, name=cli_name)
        return True
    if callable(entry):
        app.command(cli_name)(entry)
        return True
    logger.warning("scenario %s exports %r, which is not a command", cli_name, entry)
    return False


for _cli_name, _module_name, _candidates in OPTIONAL_SCENARIOS:
    register_optional_scenario(bench_app, _cli_name, _module_name, _candidates)


@bench_app.command("render")
def render_command(
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Directory holding the result JSON files.")
    ] = Path("results"),
    readme: Annotated[
        Path | None, typer.Option("--readme", help="Where to write the results README.")
    ] = None,
    docs: Annotated[
        Path | None, typer.Option("--docs", help="Where to write the documentation page.")
    ] = None,
    plots: Annotated[
        Path | None, typer.Option("--plots-dir", help="Where to write the PNG plots.")
    ] = None,
    make_plots: Annotated[
        bool, typer.Option("--plots/--no-plots", help="Draw the plots as well as the tables.")
    ] = True,
) -> None:
    """Regenerate the results pages and plots from every result file under --results-dir.

    The same rendering ``turboserve results render`` performs, exposed here so that a
    measurement session is one command group from end to end.
    """
    from turboserve.bench.report import render_index

    rendered = render_index(
        results_dir,
        readme_path=readme,
        docs_path=docs,
        plots_dir=plots,
        make_plots=make_plots,
    )
    typer.echo(
        f"rendered {rendered.num_runs} run(s) across {len(rendered.scenarios)} scenario(s); "
        f"{len(rendered.plot_paths)} plot(s)"
    )
    for path in (rendered.readme_path, rendered.docs_path):
        if path is not None:
            typer.echo(f"  {path}")


@bench_app.command("profiles")
def profiles_command(
    profiles_path: Annotated[
        Path | None, typer.Option("--profiles", help="Alternative profiles file.")
    ] = None,
) -> None:
    """List the workload profiles the scenarios can be sized from."""
    from turboserve.bench.profiles import ProfileError, load_profiles

    try:
        loaded = load_profiles(profiles_path)
    except ProfileError as exc:
        raise typer.BadParameter(str(exc), param_hint="--profiles") from exc
    for name, profile in sorted(loaded.items()):
        typer.echo(f"{name}\t{profile.device}\t{profile.description}")
