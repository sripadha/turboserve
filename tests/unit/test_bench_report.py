"""Report tests: markdown tables, provenance lines, plots and the two generated pages.

The runs here are synthetic (built in-process, hardware block supplied so nothing shells
out to nvidia-smi or git) and are written into ``tmp_path``. Nothing in the repository's
own ``results/`` directory is read or written.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from turboserve.bench.plots import (
    concurrency_of,
    latency_bars,
    latency_vs_concurrency,
    render_scenario_plots,
    throughput_bars,
)
from turboserve.bench.records import RequestRecord, RunResult
from turboserve.bench.report import (
    MISSING,
    discover_runs,
    md_table,
    provenance_line,
    render_index,
    render_run,
    results_app,
)

MS = 1_000_000

HARDWARE = {
    "gpu_name": "NVIDIA H100 80GB HBM3",
    "git": {"sha": "0123456789abcdef", "branch": "main", "dirty": False},
}

PROJECTED_NOTE = "reference results for the h100 profile; regenerate with make bench-h100"


def make_records(count: int, *, ttft_ms: float, tpot_ms: float = 10.0) -> list[RequestRecord]:
    """Records with exactly controlled latencies so the rendered numbers are predictable."""
    output_tokens = 5
    return [
        RequestRecord(
            request_id=f"r{index}",
            tenant="tenant-a",
            prompt_tokens=100,
            output_tokens=output_tokens,
            t_send_ns=index * MS,
            t_first_ns=int((index + ttft_ms) * MS),
            t_last_ns=int((index + ttft_ms + tpot_ms * (output_tokens - 1)) * MS),
            itl_ns=[int(tpot_ms * MS)] * (output_tokens - 1),
            backend="reference",
        )
        for index in range(count)
    ]


def make_run(
    scenario: str,
    label: str,
    *,
    profile: str = "h100",
    baseline: str = "naive",
    ttft_ms: float = 100.0,
    concurrency: int = 32,
    started_at: str = "2026-09-16T10:00:00+00:00",
    provenance: str = "projected",
    derived: dict[str, float] | None = None,
    count: int = 8,
) -> RunResult:
    """One finished synthetic run carrying the config conventions the renderer reads."""
    run = RunResult(
        scenario=scenario,
        profile=profile,
        hardware=dict(HARDWARE),
        software={"turboserve": "0.1.0"},
        git_sha="0123456789abcdef",
        started_at=started_at,
        gpu_price_per_hour=2.49,
        price_source="synthetic fixture",
        provenance=provenance,  # type: ignore[arg-type]
        provenance_note=PROJECTED_NOTE if provenance == "projected" else None,
        config={
            "label": label,
            "backend": "reference",
            "baseline_label": baseline,
            "load": {"concurrency": concurrency},
        },
        requests=make_records(count, ttft_ms=ttft_ms),
    )
    run.finish()
    if derived:
        run.summary["derived"] = dict(derived)
    return run


def write_runs(root: Path, runs: list[RunResult]) -> list[Path]:
    """Save runs the way a scenario does: ``results/<scenario>/<n>.json`` plus the index."""
    paths = []
    for index, run in enumerate(runs):
        paths.append(run.save(root / run.scenario / f"{index:03d}.json"))
    return paths


# -- small helpers ------------------------------------------------------------------------


def test_md_table_escapes_pipes_and_skips_empty_bodies() -> None:
    table = md_table(["a|b", "c"], [["x|y", "z"]])
    assert table.splitlines()[0] == "| a\\|b | c |"
    assert table.splitlines()[2] == "| x\\|y | z |"
    assert md_table(["a"], []) == ""


def test_provenance_line_names_the_kind_gpu_day_and_note() -> None:
    line = provenance_line([make_run("naive_vs_cb", "naive")])
    assert line.startswith("_Provenance: projected")
    assert "NVIDIA H100 80GB HBM3" in line
    assert "2026-09-16" in line
    assert "git 0123456" in line
    assert PROJECTED_NOTE in line


def test_provenance_line_reports_a_mixed_set_as_mixed() -> None:
    runs = [
        make_run("naive_vs_cb", "a", provenance="measured"),
        make_run("naive_vs_cb", "b", provenance="projected"),
    ]
    assert "measured + projected" in provenance_line(runs)
    assert provenance_line([]) == "_No runs._"


def test_provenance_line_flags_a_projected_run_with_no_note() -> None:
    run = make_run("naive_vs_cb", "a")
    run.provenance_note = None
    assert "carry no note" in provenance_line([run])


# -- single run ---------------------------------------------------------------------------


def test_render_run_shows_the_run_facts_its_table_and_its_provenance() -> None:
    run = make_run("prefix_cache", "cache on", derived={"hit_rate": 0.87})
    text = render_run(run)
    assert text.startswith("## prefix_cache — cache on")
    assert "| Scenario | prefix_cache |" in text
    assert "| GPU | NVIDIA H100 80GB HBM3 |" in text
    assert "TTFT p50 (ms)" in text
    assert "_Provenance: projected" in text
    assert "hit_rate" in text


def test_render_run_prints_an_em_dash_for_a_metric_with_no_sample() -> None:
    run = RunResult(scenario="chaos", profile="h100", hardware={}, config={"label": "empty"})
    run.finish()
    text = render_run(run, title="empty run")
    assert "## empty run" in text
    assert MISSING in text


# -- discovery ----------------------------------------------------------------------------


def test_discover_runs_skips_the_index_and_unreadable_files(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_runs(root, [make_run("naive_vs_cb", "naive")])
    assert (root / "index.json").is_file()
    (root / "naive_vs_cb" / "broken.json").write_text("{not json", encoding="utf-8")
    loaded = discover_runs(root)
    assert [path.name for path, _ in loaded] == ["000.json"]
    assert discover_runs(tmp_path / "absent") == []


def test_render_index_keeps_only_the_newest_run_of_each_arm(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_runs(
        root,
        [
            make_run("naive_vs_cb", "naive", started_at="2026-09-15T10:00:00+00:00", ttft_ms=200.0),
            make_run("naive_vs_cb", "naive", started_at="2026-09-16T10:00:00+00:00", ttft_ms=100.0),
        ],
    )
    rendered = render_index(root, docs_path=tmp_path / "docs" / "results.md", make_plots=False)
    assert rendered.num_runs == 1
    assert "2026-09-15" not in rendered.readme_text


# -- the index pages ----------------------------------------------------------------------


@pytest.fixture
def rendered_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A results tree with two scenarios, several arms, and both pages rendered."""
    root = tmp_path / "results"
    docs = tmp_path / "docs" / "results.md"
    write_runs(
        root,
        [
            make_run("naive_vs_cb", "naive", ttft_ms=300.0, concurrency=32),
            make_run("naive_vs_cb", "continuous batching", ttft_ms=100.0, concurrency=32),
            make_run("naive_vs_cb", "continuous batching", ttft_ms=120.0, concurrency=64),
            make_run("naive_vs_cb", "naive", ttft_ms=340.0, concurrency=64),
            make_run(
                "prefix_cache",
                "cache on",
                baseline="cache off",
                ttft_ms=70.0,
                derived={"hit_rate": 0.9},
            ),
            make_run("prefix_cache", "cache off", baseline="cache off", ttft_ms=100.0),
        ],
    )
    render_index(root, docs_path=docs)
    return root, root / "README.md", docs


def test_both_pages_are_written_and_carry_the_same_numbers(
    rendered_tree: tuple[Path, Path, Path],
) -> None:
    _, readme, docs = rendered_tree
    assert readme.is_file() and docs.is_file()
    readme_text = readme.read_text(encoding="utf-8")
    docs_text = docs.read_text(encoding="utf-8")
    assert "Do not edit by hand" in readme_text
    for text in (readme_text, docs_text):
        assert "## `naive_vs_cb`" in text
        assert "### profile `h100`" in text
        assert "| continuous batching |" in text
        assert "Relative to `naive`" in text
        assert "_Provenance: projected" in text
        assert PROJECTED_NOTE in text
    # Same rows, different link depth: the docs page reaches back out of docs/.
    assert "](../results/plots/" in docs_text
    assert "](plots/" in readme_text


def test_the_relative_table_shows_ratios_and_signed_deltas(
    rendered_tree: tuple[Path, Path, Path],
) -> None:
    text = rendered_tree[1].read_text(encoding="utf-8")
    prefix = "| continuous batching |"
    relative_rows = [line for line in text.splitlines() if line.startswith(prefix)]
    # One absolute row and one relative row for the arm.
    assert len(relative_rows) >= 2
    assert any("-66.7%" in row for row in relative_rows)
    assert "Arm vs baseline" in text
    assert "x |" in text


def test_derived_figures_and_run_files_are_listed(
    rendered_tree: tuple[Path, Path, Path],
) -> None:
    text = rendered_tree[1].read_text(encoding="utf-8")
    assert "Scenario-specific figures:" in text
    assert "hit_rate" in text
    assert "0.900" in text
    assert "Result files:" in text
    assert "[`prefix_cache/004.json`](prefix_cache/004.json)" in text
    assert "[`prefix_cache/005.json`](prefix_cache/005.json)" in text


def test_plots_are_written_next_to_the_readme(rendered_tree: tuple[Path, Path, Path]) -> None:
    root = rendered_tree[0]
    plots = sorted(path.name for path in (root / "plots").glob("*.png"))
    assert "naive_vs_cb-output-tok-s.png" in plots
    assert "naive_vs_cb-ttft.png" in plots
    assert "naive_vs_cb-e2e-vs-concurrency.png" in plots
    assert all((root / "plots" / name).stat().st_size > 0 for name in plots)


def test_an_empty_results_directory_renders_a_page_that_says_so(tmp_path: Path) -> None:
    rendered = render_index(tmp_path / "results", docs_path=tmp_path / "docs" / "results.md")
    assert rendered.num_runs == 0
    assert "No result files were found" in rendered.readme_text
    assert rendered.plot_paths == []


def test_render_index_can_render_without_writing(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_runs(root, [make_run("chaos", "steady")])
    rendered = render_index(root, make_plots=False, write=False)
    assert rendered.readme_path is None
    assert not (root / "README.md").exists()
    assert "## `chaos`" in rendered.readme_text


# -- plots --------------------------------------------------------------------------------


def test_concurrency_is_read_from_the_load_block_or_the_flat_key() -> None:
    run = make_run("naive_vs_cb", "a", concurrency=64)
    assert concurrency_of(run) == 64
    run.config = {"concurrency": 16}
    assert concurrency_of(run) == 16
    run.config = {}
    assert concurrency_of(run) is None


def test_plots_are_skipped_when_the_data_cannot_support_them(tmp_path: Path) -> None:
    single = [make_run("naive_vs_cb", "only")]
    assert throughput_bars(single, tmp_path / "a.png") is None
    assert latency_bars(single, tmp_path / "b.png") is None
    # Two arms but one concurrency: no curve to draw.
    pair = [make_run("naive_vs_cb", "a"), make_run("naive_vs_cb", "b")]
    assert latency_vs_concurrency(pair, tmp_path / "c.png") is None
    assert not list(tmp_path.glob("*.png"))


def test_render_scenario_plots_writes_what_it_can(tmp_path: Path) -> None:
    runs = [
        make_run("spec_decode", "k=2", concurrency=1, ttft_ms=90.0),
        make_run("spec_decode", "k=2 @4", concurrency=4, ttft_ms=95.0),
    ]
    written = render_scenario_plots("spec_decode", runs, tmp_path)
    assert {path.name for path in written} == {
        "spec_decode-output-tok-s.png",
        "spec_decode-ttft.png",
    }


# -- CLI ----------------------------------------------------------------------------------


def test_results_render_command_writes_both_pages(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_runs(root, [make_run("multi_lora", "16 adapters"), make_run("multi_lora", "base only")])
    docs = tmp_path / "docs" / "results.md"
    result = CliRunner().invoke(
        results_app,
        ["render", "--results-dir", str(root), "--docs", str(docs), "--no-plots"],
    )
    assert result.exit_code == 0, result.output
    assert "rendered 2 run(s)" in result.output
    assert docs.is_file()
    assert (root / "README.md").is_file()


def test_results_show_command_prints_one_run(tmp_path: Path) -> None:
    root = tmp_path / "results"
    paths = write_runs(root, [make_run("chaos", "steady")])
    runner = CliRunner()
    result = runner.invoke(results_app, ["show", str(paths[0])])
    assert result.exit_code == 0, result.output
    assert "## chaos — steady" in result.output

    missing = runner.invoke(results_app, ["show", str(root / "nope.json")])
    assert missing.exit_code == 1
