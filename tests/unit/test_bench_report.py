"""Report tests: markdown tables, provenance lines, plots and the two generated pages.

The runs here are synthetic (built in-process, hardware block supplied so nothing shells
out to nvidia-smi or git) and are written into ``tmp_path``. Nothing in the repository's
own ``results/`` directory is read or written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    README_END,
    README_START,
    discover_runs,
    hardware_line,
    label_sort_key,
    md_table,
    provenance_line,
    readme_section,
    render_index,
    render_run,
    results_app,
    suite_line,
    update_between_markers,
)

MS = 1_000_000

HARDWARE = {
    "gpu_name": "NVIDIA H100 80GB HBM3",
    "nvidia_smi": {
        "driver_version": "570.86.16",
        "driver_cuda_version": "12.8",
        "gpus": [{"name": "NVIDIA H100 80GB HBM3"}],
    },
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
    derived: dict[str, Any] | None = None,
    count: int = 8,
    config_extra: dict[str, Any] | None = None,
) -> RunResult:
    """One finished synthetic run carrying the config conventions the renderer reads."""
    run = RunResult(
        scenario=scenario,
        profile=profile,
        hardware=dict(HARDWARE),
        software={"turboserve": "0.1.0", "torch": "2.6.0+cu124", "vllm": "0.11.0"},
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
            **(config_extra or {}),
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


#: The two charts the project README embeds, as (scenario, plot suffix). Their paths are a
#: function of the scenario name, which is what makes a relative link to one of them safe to
#: put on a page nobody re-renders.
README_PLOTS: tuple[tuple[str, str], ...] = (
    ("naive_vs_cb", "output-tok-s"),
    ("prefix_cache", "ttft"),
)


@pytest.mark.parametrize(("scenario", "suffix"), README_PLOTS)
def test_the_readme_embeds_plots_the_renderer_keeps_writing(
    scenario: str, suffix: str, tmp_path: Path
) -> None:
    """The README's two charts must survive every re-render.

    Three things have to agree and none of them is checked anywhere else: the README links a
    path, ``render_scenario_plots`` writes that exact name for that scenario, and the file is
    in the repository right now. A renamed plot would otherwise show up as two broken images
    on the front page and nowhere in the test suite.
    """
    repo_root = Path(__file__).resolve().parents[2]
    link = f"results/plots/{scenario}-{suffix}.png"
    assert f"]({link})" in (repo_root / "README.md").read_text(encoding="utf-8")
    assert (repo_root / link).is_file(), f"{link} is linked but not committed"

    runs = [
        make_run(scenario, "a", concurrency=32, ttft_ms=10.0),
        make_run(scenario, "b", concurrency=64, ttft_ms=20.0),
    ]
    written = {path.name for path in render_scenario_plots(scenario, runs, tmp_path)}
    assert f"{scenario}-{suffix}.png" in written


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


# -- hardware line, several baselines, the README section -----------------------------------


def test_hardware_line_names_the_gpu_driver_stack_and_price() -> None:
    line = hardware_line([make_run("naive_vs_cb", "naive")])
    assert line.startswith("**Hardware:** 1x NVIDIA H100 80GB HBM3")
    assert "driver 570.86.16" in line
    assert "CUDA 12.8" in line
    assert "torch 2.6.0+cu124" in line
    assert "vllm 0.11.0" in line
    assert "$2.49/GPU-hour (synthetic fixture)" in line
    assert hardware_line([]) == ""


def test_hardware_line_omits_what_no_run_recorded() -> None:
    run = make_run("naive_vs_cb", "naive")
    run.hardware = {}
    run.software = {}
    run.gpu_price_per_hour = None
    assert hardware_line([run]) == ""


def test_suite_line_prices_the_whole_sweep_from_the_run_timestamps() -> None:
    """What a reader renting the hardware needs: how long, and how much.

    The span deliberately runs start-to-finish across every run rather than summing the
    runs, because the gaps -- checkpoint loads, warm-ups -- are billed too.
    """
    first = make_run("naive_vs_cb", "naive", started_at="2026-09-16T10:00:00+00:00")
    first.finished_at = "2026-09-16T10:30:00+00:00"
    last = make_run("prefix_cache", "cache on", started_at="2026-09-16T11:30:00+00:00")
    last.finished_at = "2026-09-16T12:00:00+00:00"
    line = suite_line([first, last])
    assert line.startswith("**Suite:** 2 run(s) across 2 scenario(s), 2h 00m")
    assert "$4.98 of GPU time at $2.49/hour" in line


def test_suite_line_drops_the_money_when_the_price_is_unknown_or_disputed() -> None:
    cheap = make_run("naive_vs_cb", "naive")
    cheap.finished_at = "2026-09-16T10:30:00+00:00"
    dear = make_run("naive_vs_cb", "static batch", started_at="2026-09-16T10:30:00+00:00")
    dear.finished_at = "2026-09-16T11:00:00+00:00"
    dear.gpu_price_per_hour = 3.10
    line = suite_line([cheap, dear])
    assert "1h 00m" in line
    assert "$" not in line


def test_suite_line_is_empty_without_usable_timestamps() -> None:
    assert suite_line([]) == ""
    run = make_run("naive_vs_cb", "naive", started_at="not a timestamp")
    assert suite_line([run]) == ""


def test_each_declared_baseline_gets_its_own_relative_table(tmp_path: Path) -> None:
    # Two families in one concurrency group, as the speculative sweep writes them: each pair
    # is only comparable against its own target-only arm.
    runs = [
        make_run("spec_decode", "pair-a / target only", baseline="pair-a / target only"),
        make_run("spec_decode", "pair-a / k=4", baseline="pair-a / target only", ttft_ms=50.0),
        make_run("spec_decode", "pair-b / target only", baseline="pair-b / target only"),
        make_run("spec_decode", "pair-b / k=4", baseline="pair-b / target only", ttft_ms=80.0),
    ]
    write_runs(tmp_path / "results", runs)
    text = render_index(
        tmp_path / "results", docs_path=tmp_path / "docs.md", make_plots=False
    ).readme_text
    assert "Relative to `pair-a / target only` at concurrency 32:" in text
    assert "Relative to `pair-b / target only` at concurrency 32:" in text
    # ... and the families do not leak into each other's table.
    table_a = text.split("Relative to `pair-a / target only`")[1].split("Relative to")[0]
    assert "pair-a / k=4" in table_a
    assert "pair-b" not in table_a


def test_compare_to_adds_a_relative_table_against_a_second_arm(tmp_path: Path) -> None:
    extra = {"compare_to": ["static batch"]}
    runs = [
        make_run("naive_vs_cb", "naive", config_extra=extra),
        make_run("naive_vs_cb", "static batch", ttft_ms=200.0),
        make_run("naive_vs_cb", "continuous batching", ttft_ms=50.0, config_extra=extra),
    ]
    write_runs(tmp_path / "results", runs)
    text = render_index(
        tmp_path / "results", docs_path=tmp_path / "docs.md", make_plots=False
    ).readme_text
    assert "Relative to `naive` at concurrency 32:" in text
    assert "Relative to `static batch` at concurrency 32:" in text


def test_a_declared_baseline_nobody_measured_is_skipped(tmp_path: Path) -> None:
    runs = [make_run("chaos", "steady", baseline="a lane that was not run")]
    write_runs(tmp_path / "results", runs)
    text = render_index(
        tmp_path / "results", docs_path=tmp_path / "docs.md", make_plots=False
    ).readme_text
    assert "Relative to" not in text


def test_derived_blocks_render_as_their_own_tables_and_counts_stay_counts(
    tmp_path: Path,
) -> None:
    derived: dict[str, Any] = {
        "num_adapters": 16,
        "p95_ttft_loss_pct": 5.25,
        "vram": {"saved_pct": 96.5, "num_slots": 16},
    }
    runs = [make_run("multi_lora", "16 adapters", derived=derived)]
    write_runs(tmp_path / "results", runs)
    text = render_index(
        tmp_path / "results", docs_path=tmp_path / "docs.md", make_plots=False
    ).readme_text
    assert "Scenario-specific figures:" in text
    assert "Scenario-specific figures — `vram`:" in text
    assert "| 16 adapters | 16 | 5.250 |" in text  # an int is not 16.000
    assert "96.500" in text
    assert "{'saved_pct'" not in text  # never a python dict in a cell


def test_labels_sort_the_way_a_reader_reads_them() -> None:
    labels = ["100 adapters", "10 adapters", "base only", "32 adapters"]
    assert sorted(labels, key=label_sort_key) == [
        "10 adapters",
        "32 adapters",
        "100 adapters",
        "base only",
    ]


def test_readme_section_carries_the_headline_scenario_and_the_machine() -> None:
    loaded = [
        (Path("results/naive_vs_cb/000.json"), make_run("naive_vs_cb", "naive")),
        (
            Path("results/naive_vs_cb/001.json"),
            make_run("naive_vs_cb", "continuous batching", ttft_ms=20.0),
        ),
        (Path("results/chaos/002.json"), make_run("chaos", "steady")),
    ]
    section = readme_section(loaded)
    assert "**Hardware:**" in section
    assert "### `naive_vs_cb`" in section
    assert "Relative to `naive` at concurrency 32:" in section
    assert "Provenance: projected" in section
    assert "`chaos`" in section  # the other scenarios are named, not tabled
    assert "steady" not in section
    assert readme_section([]).startswith("No result files were found")


def test_render_index_fills_the_project_readme_between_its_markers(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_runs(root, [make_run("naive_vs_cb", "naive"), make_run("naive_vs_cb", "cb", ttft_ms=9.0)])
    readme = tmp_path / "README.md"
    readme.write_text(f"# demo\n\n{README_START}\nstale\n{README_END}\n\n## License\n", "utf-8")
    render_index(root, docs_path=tmp_path / "docs.md", make_plots=False)
    text = readme.read_text(encoding="utf-8")
    assert "stale" not in text
    assert "### `naive_vs_cb`" in text
    assert text.startswith("# demo")
    assert text.endswith("## License\n")


def test_a_readme_without_markers_is_left_alone(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# demo\n", encoding="utf-8")
    assert update_between_markers(readme, "tables") is False
    assert readme.read_text(encoding="utf-8") == "# demo\n"
    assert update_between_markers(tmp_path / "absent.md", "tables") is False
