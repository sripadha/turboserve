"""Charts for the results pages, drawn from the same JSON the tables are drawn from.

Every figure here is a function of a list of :class:`~turboserve.bench.records.RunResult`
objects and nothing else: no data is passed in alongside them, so a plot cannot disagree
with the table above it. If a figure would have no data -- one arm only, or no concurrency
recorded -- the function returns ``None`` and no file is written, because an axis drawn
through a single point suggests a trend that was not measured.

matplotlib is imported lazily and forced onto the ``Agg`` backend. Lazily because the
import costs a noticeable fraction of a second and the report renderer is also used to
print one run to a terminal; ``Agg`` because plots are generated on headless measurement
hosts and in CI, where selecting a GUI backend would fail outright.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from turboserve.bench.metrics import run_label, run_summary

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from turboserve.bench.records import RunResult

logger = logging.getLogger(__name__)

__all__ = [
    "FIGURE_DPI",
    "FIGURE_SIZE",
    "latency_bars",
    "latency_vs_concurrency",
    "render_scenario_plots",
    "throughput_bars",
]

#: Wide enough for a dozen labelled bars at a readable font, and the aspect ratio GitHub
#: renders without letterboxing in a markdown page.
FIGURE_SIZE = (9.0, 4.5)
FIGURE_DPI = 140

#: Human names for the metric keys a plot can be asked for.
METRIC_LABELS: dict[str, str] = {
    "ttft_ms": "TTFT (ms)",
    "itl_ms": "ITL (ms)",
    "tpot_ms": "TPOT (ms)",
    "e2e_ms": "End-to-end latency (ms)",
    "output_tok_s": "Output tokens/s",
    "total_tok_s": "Total tokens/s",
    "req_s": "Requests/s",
    "cost_per_1m_output_tokens_usd": "USD per 1M output tokens",
}


def _pyplot() -> Any | None:
    """The pyplot module on the Agg backend, or ``None`` if matplotlib is unavailable."""
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - matplotlib is a declared dependency
        logger.warning("matplotlib is not installed; skipping plots")
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def metric_label(metric: str, statistic: str | None = None) -> str:
    """Axis label for a metric, with the percentile appended when there is one."""
    base = METRIC_LABELS.get(metric, metric)
    return f"{base} {statistic}" if statistic else base


def _stat(summary: Mapping[str, Any], metric: str, statistic: str | None) -> float | None:
    """One number out of a summary, whether it is scalar or inside a percentile block."""
    value = summary.get(metric)
    if statistic is not None:
        if not isinstance(value, Mapping):
            return None
        value = value.get(statistic)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def concurrency_of(result: RunResult) -> int | None:
    """The concurrency a run was driven at, from its ``config`` block.

    Scenarios embed the load-generator settings under ``config["load"]``; the flat key is
    accepted too so a config written by hand for a one-off run still plots.
    """
    load = result.config.get("load")
    for source in (load if isinstance(load, Mapping) else None, result.config):
        if not isinstance(source, Mapping):
            continue
        value = source.get("concurrency")
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        return value
    return None


def _finish(plt: Any, figure: Any, path: Path) -> Path:
    """Write a figure and release it; matplotlib keeps figures alive until closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    logger.debug("wrote plot %s", path)
    return path


def throughput_bars(
    results: Sequence[RunResult],
    path: Path | str,
    *,
    metric: str = "output_tok_s",
    title: str | None = None,
) -> Path | None:
    """One bar per arm of a scenario, for a scalar throughput or cost metric."""
    pairs = [(run_label(result), _stat(run_summary(result), metric, None)) for result in results]
    usable = [(label, value) for label, value in pairs if value is not None]
    if len(usable) < 2:
        logger.debug("not enough arms to plot %s (%d)", metric, len(usable))
        return None
    plt = _pyplot()
    if plt is None:
        return None
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    labels = [label for label, _ in usable]
    values = [value for _, value in usable]
    bars = axes.bar(labels, values, color="#3b6ea5")
    axes.set_ylabel(metric_label(metric))
    axes.set_title(title or metric_label(metric))
    axes.grid(axis="y", linestyle=":", alpha=0.6)
    axes.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axes.tick_params(axis="x", labelrotation=20)
    return _finish(plt, figure, Path(path))


def latency_bars(
    results: Sequence[RunResult],
    path: Path | str,
    *,
    metric: str = "ttft_ms",
    statistics: Sequence[str] = ("p50", "p95"),
    title: str | None = None,
) -> Path | None:
    """Grouped bars of one latency metric at several percentiles, one group per arm.

    Median and tail side by side because a change that improves one and wrecks the other is
    the most common way a serving optimisation goes wrong, and a single-percentile chart
    hides it.
    """
    rows: list[tuple[str, list[float | None]]] = []
    for result in results:
        summary = run_summary(result)
        rows.append((run_label(result), [_stat(summary, metric, stat) for stat in statistics]))
    usable = [(label, values) for label, values in rows if any(v is not None for v in values)]
    if len(usable) < 2:
        logger.debug("not enough arms to plot %s (%d)", metric, len(usable))
        return None
    plt = _pyplot()
    if plt is None:
        return None
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    positions = range(len(usable))
    width = 0.8 / len(statistics)
    for index, statistic in enumerate(statistics):
        offsets = [position + index * width - 0.4 + width / 2 for position in positions]
        heights = [row[index] or 0.0 for _, row in usable]
        axes.bar(offsets, heights, width=width, label=statistic)
    axes.set_xticks(list(positions))
    axes.set_xticklabels([label for label, _ in usable], rotation=20, ha="right")
    axes.set_ylabel(metric_label(metric))
    axes.set_title(title or metric_label(metric))
    axes.grid(axis="y", linestyle=":", alpha=0.6)
    axes.legend(title="percentile")
    return _finish(plt, figure, Path(path))


def latency_vs_concurrency(
    results: Sequence[RunResult],
    path: Path | str,
    *,
    metric: str = "e2e_ms",
    statistic: str = "p95",
    title: str | None = None,
) -> Path | None:
    """One line per arm: a latency percentile against the concurrency it was driven at.

    This is the plot that shows a scheduler's real character. Two engines can report the
    same throughput at one load level and diverge completely as concurrency rises, and the
    slope of this line is what a capacity plan is built from.
    """
    series: dict[str, list[tuple[int, float]]] = {}
    for result in results:
        concurrency = concurrency_of(result)
        value = _stat(run_summary(result), metric, statistic)
        if concurrency is None or value is None:
            continue
        series.setdefault(run_label(result), []).append((concurrency, value))
    plottable = {
        label: sorted(points) for label, points in series.items() if len({p[0] for p in points}) > 1
    }
    if not plottable:
        logger.debug("no arm has more than one concurrency; skipping %s", path)
        return None
    plt = _pyplot()
    if plt is None:
        return None
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    for label, points in plottable.items():
        axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=label,
        )
    axes.set_xlabel("Concurrent requests")
    axes.set_ylabel(metric_label(metric, statistic))
    axes.set_title(title or f"{metric_label(metric, statistic)} vs concurrency")
    axes.grid(linestyle=":", alpha=0.6)
    axes.legend()
    return _finish(plt, figure, Path(path))


def render_scenario_plots(
    scenario: str, results: Sequence[RunResult], out_dir: Path | str
) -> list[Path]:
    """Draw the standard set for one scenario; returns the files actually written.

    Three figures at most -- throughput, TTFT percentiles, and latency against concurrency
    -- and each is skipped when the runs do not support it, so a scenario measured at a
    single concurrency simply gets two.
    """
    directory = Path(out_dir)
    candidates = [
        throughput_bars(
            results,
            directory / f"{scenario}-output-tok-s.png",
            title=f"{scenario}: output tokens/s",
        ),
        latency_bars(
            results,
            directory / f"{scenario}-ttft.png",
            metric="ttft_ms",
            title=f"{scenario}: time to first token",
        ),
        latency_vs_concurrency(
            results,
            directory / f"{scenario}-e2e-vs-concurrency.png",
            title=f"{scenario}: p95 end-to-end latency vs concurrency",
        ),
    ]
    return [path for path in candidates if path is not None]
