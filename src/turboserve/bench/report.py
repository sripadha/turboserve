"""Rendering result files into the markdown pages and plots the repository publishes.

The rule this module exists to enforce: **no number is ever typed into a markdown file by
hand**. `results/README.md`, `docs/results.md` and every PNG beside them are regenerated
from `results/**/*.json` by :func:`render_index`, so a figure in the documentation can
always be traced back to the run that produced it, complete with the hardware, the software
versions, the configuration and the git sha recorded in that file. ``turboserve results
render`` is this function; a pull request that edits a table without editing a result file
is wrong by construction, because the next render will overwrite it.

The second rule: **every table says where its numbers came from**. Each table is followed
by a provenance line naming whether the runs behind it were measured or projected, on what
GPU, and when. A projected table carries the note from its result files saying how to
replace it with a measured one. A reader never has to guess.

The layout is: one section per scenario, one sub-section per profile, an absolute table of
every arm, then -- when the arms declare a baseline -- a relative table of ratios and
deltas against it, then any scenario-specific derived figures, then the plots, then the
list of files the section was built from.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from turboserve.bench.metrics import (
    DERIVED_KEY,
    baseline_label,
    compare_runs,
    run_label,
    run_summary,
)
from turboserve.bench.plots import concurrency_of
from turboserve.bench.records import RunResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "MISSING",
    "RenderedIndex",
    "discover_runs",
    "provenance_line",
    "render_index",
    "render_run",
    "results_app",
]

#: What an absent number renders as. An em dash, never a zero: a zero in a latency column
#: reads as "instant" and would be the single most misleading character in these pages.
MISSING = "—"

#: Columns of the absolute per-arm table, as (header, accessor key, statistic, digits).
_SUMMARY_COLUMNS: tuple[tuple[str, str, str | None, int], ...] = (
    ("Requests", "num_requests", None, 0),
    ("Errors", "error_rate", None, 4),
    ("TTFT p50 (ms)", "ttft_ms", "p50", 1),
    ("TTFT p95 (ms)", "ttft_ms", "p95", 1),
    ("ITL p50 (ms)", "itl_ms", "p50", 2),
    ("TPOT p95 (ms)", "tpot_ms", "p95", 2),
    ("E2E p95 (ms)", "e2e_ms", "p95", 1),
    ("Output tok/s", "output_tok_s", None, 1),
    ("Req/s", "req_s", None, 2),
    ("USD / 1M out", "cost_per_1m_output_tokens_usd", None, 3),
)

#: Columns of the relative table, as (header, :class:`Comparison` field, suffix, digits).
_COMPARISON_COLUMNS: tuple[tuple[str, str, str, int], ...] = (
    ("Output tok/s", "output_tok_s_ratio", "x", 2),
    ("Req/s", "req_s_ratio", "x", 2),
    ("TTFT p50", "ttft_p50_delta_pct", "%", 1),
    ("TTFT p95", "ttft_p95_delta_pct", "%", 1),
    ("ITL p95", "itl_p95_delta_pct", "%", 1),
    ("E2E p95", "e2e_p95_delta_pct", "%", 1),
    ("USD / 1M out", "cost_ratio", "x", 2),
)


@dataclass(frozen=True, slots=True)
class RenderedIndex:
    """What one :func:`render_index` call produced."""

    readme_text: str = ""
    docs_text: str = ""
    readme_path: Path | None = None
    docs_path: Path | None = None
    plot_paths: list[Path] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    num_runs: int = 0


# -- small markdown helpers ---------------------------------------------------------------


def _number(value: Any, digits: int = 1, *, suffix: str = "") -> str:
    """Format a number for a table cell, or :data:`MISSING` when there is none."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return MISSING
    if digits == 0:
        return f"{int(round(value))}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def _signed(value: Any, digits: int = 1, *, suffix: str = "%") -> str:
    """A percentage delta with an explicit sign, so an improvement is unmistakable."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return MISSING
    return f"{value:+.{digits}f}{suffix}"


def _escape(text: str) -> str:
    """Escape the one character that can break a markdown table cell."""
    return text.replace("|", "\\|")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """A GitHub-flavoured markdown table; returns an empty string when there are no rows."""
    body = [list(row) for row in rows]
    if not body:
        return ""
    lines = [
        "| " + " | ".join(_escape(str(header)) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_escape(str(cell)) for cell in row) + " |" for row in body)
    return "\n".join(lines)


def _stat(summary: Mapping[str, Any], metric: str, statistic: str | None) -> Any:
    """One value from a summary, scalar or inside a percentile block."""
    value = summary.get(metric)
    if statistic is None:
        return value
    if not isinstance(value, Mapping):
        return None
    return value.get(statistic)


# -- provenance ---------------------------------------------------------------------------


def _gpu_name(result: RunResult) -> str | None:
    """The GPU a run was made on, as recorded by ``hwinfo.collect()``."""
    name = result.hardware.get("gpu_name")
    return name if isinstance(name, str) and name else None


def _short_sha(result: RunResult) -> str | None:
    """The first seven characters of the recorded commit, the usual citation length."""
    return result.git_sha[:7] if isinstance(result.git_sha, str) and result.git_sha else None


def _day(value: str | None) -> str | None:
    """The date part of an ISO timestamp; the time of day adds nothing to a table."""
    if not isinstance(value, str) or not value:
        return None
    return value.split("T", 1)[0]


def provenance_line(results: Sequence[RunResult]) -> str:
    """The one-line statement printed under every table.

    Says which of ``measured`` and ``projected`` the runs are, on what GPU and when, and
    carries the projected runs' own note about how to replace them. A mixed set is reported
    as mixed rather than as the majority, because a table whose rows have different
    standing is exactly the case a reader must be warned about.
    """
    if not results:
        return "_No runs._"
    kinds = sorted({result.provenance for result in results})
    gpus = sorted({name for name in (_gpu_name(result) for result in results) if name})
    days = sorted({day for day in (_day(result.started_at) for result in results) if day})
    shas = sorted({sha for sha in (_short_sha(result) for result in results) if sha})
    parts = [f"Provenance: {' + '.join(kinds)}"]
    if gpus:
        parts.append(f"GPU {', '.join(gpus)}")
    if days:
        parts.append(days[0] if len(days) == 1 else f"{days[0]}..{days[-1]}")
    if shas:
        parts.append(f"git {', '.join(shas)}")
    notes = sorted(
        {
            result.provenance_note
            for result in results
            if result.provenance != "measured" and result.provenance_note
        }
    )
    line = "; ".join(parts) + "."
    if notes:
        line = f"{line} {' '.join(notes)}"
    elif "projected" in kinds:
        line = f"{line} Projected runs carry no note; regenerate them from a real run."
    return f"_{line}_"


# -- one run ------------------------------------------------------------------------------


def render_run(result: RunResult, *, title: str | None = None, heading_level: int = 2) -> str:
    """Markdown for a single run: what it was, what it measured, and where it came from.

    Used by ``turboserve results show`` to inspect one file, and by the index renderer for
    a scenario that produced exactly one run.
    """
    heading = "#" * max(heading_level, 1)
    label = run_label(result)
    summary = run_summary(result)
    lines = [f"{heading} {title or f'{result.scenario} — {label}'}", ""]
    facts = [
        ("Scenario", result.scenario),
        ("Profile", result.profile),
        ("Arm", label),
        ("GPU", _gpu_name(result) or MISSING),
        ("Started", result.started_at or MISSING),
        ("Git", _short_sha(result) or MISSING),
        ("Requests", str(len(result.requests))),
    ]
    lines.append(md_table(["Field", "Value"], [[name, str(value)] for name, value in facts]))
    lines.extend(["", md_table(*_summary_table([result])), "", provenance_line([result])])
    derived = summary.get(DERIVED_KEY)
    if isinstance(derived, Mapping) and derived:
        lines.extend(["", md_table(*_derived_table([result]))])
    return "\n".join(lines).rstrip() + "\n"


# -- tables -------------------------------------------------------------------------------


def _summary_table(results: Sequence[RunResult]) -> tuple[list[str], list[list[str]]]:
    """Headers and rows of the absolute per-arm table.

    Concurrency gets its own column because an arm is usually measured at several load
    levels, and two rows with the same name and different numbers would otherwise look
    like a contradiction rather than a sweep.
    """
    headers = ["Arm", "Concurrency", *(header for header, _, _, _ in _SUMMARY_COLUMNS)]
    rows: list[list[str]] = []
    for result in results:
        summary = run_summary(result)
        concurrency = concurrency_of(result)
        row = [run_label(result), str(concurrency) if concurrency is not None else MISSING]
        for _, key, statistic, digits in _SUMMARY_COLUMNS:
            row.append(_number(_stat(summary, key, statistic), digits))
        rows.append(row)
    return headers, rows


def _by_concurrency(results: Sequence[RunResult]) -> list[tuple[int | None, list[RunResult]]]:
    """Group runs by the load level they were driven at, lowest first.

    Arms are only comparable against each other at the same offered load, so every
    relative table is built inside one of these groups.
    """
    groups: dict[int | None, list[RunResult]] = {}
    for result in results:
        groups.setdefault(concurrency_of(result), []).append(result)
    return sorted(groups.items(), key=lambda item: (item[0] is None, item[0] or 0))


def _comparison_table(results: Sequence[RunResult]) -> tuple[list[str], list[list[str]]]:
    """Headers and rows of the table of ratios and deltas against the baseline arm."""
    headers = ["Arm vs baseline", *(header for header, _, _, _ in _COMPARISON_COLUMNS)]
    rows: list[list[str]] = []
    for comparison in compare_runs(results):
        row = [comparison.candidate]
        for _, attribute, suffix, digits in _COMPARISON_COLUMNS:
            value = getattr(comparison, attribute)
            row.append(
                _signed(value, digits) if suffix == "%" else _number(value, digits, suffix=suffix)
            )
        rows.append(row)
    return headers, rows


def _derived_table(results: Sequence[RunResult]) -> tuple[list[str], list[list[str]]]:
    """Scenario-specific figures a run recorded under ``summary["derived"]``."""
    keys: list[str] = []
    for result in results:
        derived = run_summary(result).get(DERIVED_KEY)
        if isinstance(derived, Mapping):
            keys.extend(key for key in derived if key not in keys)
    headers = ["Arm", *keys]
    rows: list[list[str]] = []
    for result in results:
        derived = run_summary(result).get(DERIVED_KEY)
        if not isinstance(derived, Mapping) or not derived:
            continue
        row = [run_label(result)]
        for key in keys:
            value = derived.get(key)
            row.append(
                _number(value, 3) if isinstance(value, int | float) else str(value or MISSING)
            )
        rows.append(row)
    return headers, rows


def _files_table(entries: Sequence[tuple[Path, RunResult]], base: Path) -> str:
    """The list of result files a section was rendered from, linked relative to the page.

    Every rendered row can therefore be opened and checked against the raw records that
    produced it, which is the whole point of publishing the JSON alongside the tables.
    """
    rows: list[list[str]] = []
    for path, result in entries:
        location = _relative(path, base)
        concurrency = concurrency_of(result)
        rows.append(
            [
                run_label(result),
                str(concurrency) if concurrency is not None else MISSING,
                result.started_at or MISSING,
                f"[`{location}`]({location})",
            ]
        )
    return md_table(["Arm", "Concurrency", "Started", "File"], rows)


def _relative(path: Path, base: Path) -> str:
    """``path`` as written into a page living in ``base``, POSIX-style for markdown."""
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


# -- discovery ----------------------------------------------------------------------------


def discover_runs(results_dir: Path | str) -> list[tuple[Path, RunResult]]:
    """Load every result file under ``results_dir``, newest last.

    ``index.json`` is skipped (it is the index, not a run) and a file that cannot be parsed
    is logged and skipped rather than aborting the render: one corrupt file from an
    interrupted run must not cost the whole results page.
    """
    root = Path(results_dir)
    if not root.is_dir():
        return []
    loaded: list[tuple[Path, RunResult]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "index.json" or path.name.startswith("."):
            continue
        try:
            loaded.append((path, RunResult.load(path)))
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("skipping unreadable result %s: %s", path, exc)
    loaded.sort(key=lambda item: (item[1].started_at or "", item[0].name))
    return loaded


def _latest_per_arm(loaded: Sequence[tuple[Path, RunResult]]) -> list[tuple[Path, RunResult]]:
    """Keep only the newest run of each (scenario, profile, arm).

    Re-running one arm of a scenario must update the table, not add a second row for the
    same thing; the superseded file stays on disk and in ``index.json``, so the history is
    not lost, it is just not what the page shows. The load level is part of the identity,
    because the same arm swept across concurrencies is several measurements, not one
    measurement repeated.
    """
    newest: dict[tuple[str, str, str, int | None], tuple[Path, RunResult]] = {}
    for path, result in loaded:
        key = (result.scenario, result.profile, run_label(result), concurrency_of(result))
        newest[key] = (path, result)
    return list(newest.values())


# -- the index page -----------------------------------------------------------------------


def _scenario_section(
    scenario: str,
    entries: Sequence[tuple[Path, RunResult]],
    *,
    page_dir: Path,
    plot_paths: Mapping[str, list[Path]],
) -> str:
    """One scenario's section: a sub-section per profile, then the plots for the scenario."""
    lines = [f"## `{scenario}`", ""]
    profiles = sorted({result.profile for _, result in entries})
    for profile in profiles:
        in_profile = [item for item in entries if item[1].profile == profile]
        runs = [result for _, result in in_profile]
        lines.extend([f"### profile `{profile}`", ""])
        lines.extend([md_table(*_summary_table(runs)), "", provenance_line(runs), ""])
        groups = _by_concurrency(runs)
        for concurrency, group in groups:
            comparison_headers, comparison_rows = _comparison_table(group)
            if not comparison_rows:
                continue
            reference = baseline_label(group) or MISSING
            at_load = f" at concurrency {concurrency}" if concurrency is not None else ""
            lines.extend(
                [
                    f"Relative to `{reference}`{at_load}:",
                    "",
                    md_table(comparison_headers, comparison_rows),
                    "",
                    provenance_line(group),
                    "",
                ]
            )
        derived_headers, derived_rows = _derived_table(runs)
        if derived_rows:
            lines.extend(
                [
                    "Scenario-specific figures:",
                    "",
                    md_table(derived_headers, derived_rows),
                    "",
                    provenance_line(runs),
                    "",
                ]
            )
        lines.extend(["Result files:", "", _files_table(in_profile, page_dir), ""])
    for plot in plot_paths.get(scenario, []):
        location = _relative(plot, page_dir)
        lines.extend([f"![{scenario}]({location})", ""])
    return "\n".join(lines)


def _page(
    title: str,
    loaded: Sequence[tuple[Path, RunResult]],
    *,
    page_dir: Path,
    plot_paths: Mapping[str, list[Path]],
) -> str:
    """The whole page: a header saying it is generated, then one section per scenario."""
    scenarios = sorted({result.scenario for _, result in loaded})
    lines = [
        f"# {title}",
        "",
        "<!-- Generated by `turboserve results render`. Do not edit by hand:",
        "     every number below is read from a result file under `results/`. -->",
        "",
    ]
    if not loaded:
        lines.extend(
            [
                "No result files were found. Run a scenario (`turboserve bench <scenario>`)",
                "and render again with `make results`.",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            f"{len(loaded)} run(s) across {len(scenarios)} scenario(s). "
            "Each table is followed by the provenance of the runs behind it.",
            "",
            md_table(
                ["Scenario", "Runs", "Profiles"],
                [
                    [
                        f"[`{scenario}`](#{scenario.lower()})",
                        str(sum(1 for _, r in loaded if r.scenario == scenario)),
                        ", ".join(sorted({r.profile for _, r in loaded if r.scenario == scenario})),
                    ]
                    for scenario in scenarios
                ],
            ),
            "",
        ]
    )
    for scenario in scenarios:
        entries = [item for item in loaded if item[1].scenario == scenario]
        lines.append(_scenario_section(scenario, entries, page_dir=page_dir, plot_paths=plot_paths))
    return "\n".join(lines).rstrip() + "\n"


def render_index(
    results_dir: Path | str = "results",
    *,
    readme_path: Path | str | None = None,
    docs_path: Path | str | None = None,
    plots_dir: Path | str | None = None,
    make_plots: bool = True,
    write: bool = True,
) -> RenderedIndex:
    """Regenerate ``results/README.md``, ``docs/results.md`` and the plots beside them.

    The two pages carry identical content and differ only in the relative paths they use to
    reach the plots and the result files, which is why they are rendered from one pass over
    the data instead of from two.
    """
    root = Path(results_dir)
    readme = Path(readme_path) if readme_path is not None else root / "README.md"
    docs = Path(docs_path) if docs_path is not None else root.parent / "docs" / "results.md"
    plots = Path(plots_dir) if plots_dir is not None else root / "plots"

    loaded = _latest_per_arm(discover_runs(root))
    loaded.sort(key=lambda item: (item[1].scenario, item[1].profile, run_label(item[1])))
    scenarios = sorted({result.scenario for _, result in loaded})

    plot_paths: dict[str, list[Path]] = {}
    if make_plots and loaded:
        from turboserve.bench.plots import render_scenario_plots

        for scenario in scenarios:
            runs = [result for _, result in loaded if result.scenario == scenario]
            plot_paths[scenario] = render_scenario_plots(scenario, runs, plots)

    readme_text = _page("turboserve results", loaded, page_dir=readme.parent, plot_paths=plot_paths)
    docs_text = _page("Results", loaded, page_dir=docs.parent, plot_paths=plot_paths)
    if write:
        for path, text in ((readme, readme_text), (docs, docs_text)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            logger.info("wrote %s", path)
    return RenderedIndex(
        readme_text=readme_text,
        docs_text=docs_text,
        readme_path=readme if write else None,
        docs_path=docs if write else None,
        plot_paths=[path for paths in plot_paths.values() for path in paths],
        scenarios=scenarios,
        num_runs=len(loaded),
    )


# -- CLI ----------------------------------------------------------------------------------

results_app = typer.Typer(
    name="results",
    help="Render the results pages and plots from the JSON files under results/.",
    no_args_is_help=True,
    add_completion=False,
)


@results_app.command("render")
def render_command(
    results_dir: Annotated[
        Path,
        typer.Option("--results-dir", help="Directory holding the result JSON files."),
    ] = Path("results"),
    readme: Annotated[
        Path | None,
        typer.Option("--readme", help="Where to write the results README."),
    ] = None,
    docs: Annotated[
        Path | None,
        typer.Option("--docs", help="Where to write the documentation results page."),
    ] = None,
    plots: Annotated[
        Path | None,
        typer.Option("--plots-dir", help="Where to write the PNG plots."),
    ] = None,
    make_plots: Annotated[
        bool,
        typer.Option("--plots/--no-plots", help="Draw the plots as well as the tables."),
    ] = True,
) -> None:
    """Regenerate the results pages and plots from every result file in --results-dir."""
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


@results_app.command("show")
def show_command(
    path: Annotated[Path, typer.Argument(help="A result JSON file to summarise.")],
) -> None:
    """Print one result file as a markdown section, without writing anything."""
    try:
        result = RunResult.load(path)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"cannot read {path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(render_run(result))
