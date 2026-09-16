"""The ``multi_lora`` benchmark scenario, driven end to end on CPU.

This is a pipeline test, not a measurement: a two-layer random model, six requests and
three-token completions say nothing about serving performance and nothing here asserts a
latency or a throughput. What it does assert is everything a result file's *reader* depends
on -- that the arms run, that every request succeeds through the real engine with real
adapters, that the control arm is recorded as the baseline of the others, that the VRAM
block is arithmetic over the tensors that exist, and that the files land where the report
renderer looks for them.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from turboserve.bench.profiles import BenchProfile, ProfileError
from turboserve.bench.records import RunResult, default_index_path
from turboserve.bench.scenarios.multi_lora import (
    BASE_LABEL,
    SCENARIO,
    adapter_assignment,
    arm_label,
    derived_block,
    result_path,
    run_scenario,
)

#: A hardware block injected into every RunResult so the test never shells out to
#: nvidia-smi or git the way ``hwinfo.collect()`` does.
FAKE_HARDWARE: dict[str, Any] = {"gpu": None, "git": {"sha": "0" * 40}, "host": {"os": "test"}}


def load_adapter_script() -> Any:
    """Return the synthetic-adapter trainer module.

    The trainer used to be a bare script under ``scripts/``; it now lives in the package as
    :mod:`turboserve.engine.lora.make_adapters` and is exposed as
    ``turboserve lora make-adapters``, so loading it is a plain import. The helper is kept
    so the fixtures below read the same as they did while it was a script.
    """
    return importlib.import_module("turboserve.engine.lora.make_adapters")


@pytest.fixture(scope="session")
def adapters_dir(tmp_path_factory: pytest.TempPathFactory, tiny_qwen2_path: Path) -> Path:
    """Three real PEFT adapters over the tiny Qwen2 checkpoint, trained for two steps."""
    module = load_adapter_script()
    out = tmp_path_factory.mktemp("scenario-adapters")
    module.make_adapters(
        str(tiny_qwen2_path),
        out,
        count=3,
        rank=4,
        alpha=32,
        steps=2,
        batch_size=2,
        learning_rate=0.05,
        max_length=32,
        device="cpu",
        local_files_only=True,
    )
    return out


def tiny_profile(model_path: Path) -> BenchProfile:
    """A profile shaped like ``configs/bench/profiles.yaml`` but sized for a unit test."""
    workload = {
        "model": str(model_path),
        "dtype": "float32",
        "num_requests": 6,
        "input_tokens": {"min": 8, "max": 12},
        "output_tokens": {"min": 3, "max": 4},
        "backends": ["reference"],
    }
    return BenchProfile.model_validate(
        {
            "name": "unit",
            "description": "cpu pipeline check",
            "device": "cpu",
            "seed": 7,
            "tenants": ["t0", "t1"],
            "scenarios": {
                "naive_vs_cb": {**workload, "concurrencies": [2]},
                "prefix_cache": {
                    **workload,
                    "shared_prefix_tokens": 4,
                    "concurrency": 2,
                    "prefix_caching": [True],
                },
                "spec_decode": {
                    "dtype": "float32",
                    "num_requests": 2,
                    "input_tokens": {"min": 8, "max": 8},
                    "output_tokens": {"min": 2, "max": 2},
                    "backends": ["reference"],
                    "concurrencies": [1],
                    "speculative_tokens": [2],
                    "pairs": [
                        {
                            "name": "ngram",
                            "target": str(model_path),
                            "draft": None,
                            "drafter": "ngram",
                        }
                    ],
                },
                "multi_lora": {
                    **workload,
                    "adapter_counts": [2, 3],
                    "lora_rank": 4,
                    "concurrency": 3,
                    "include_base_only": True,
                },
                "chaos": {
                    "model": str(model_path),
                    "dtype": "float32",
                    "input_tokens": {"min": 8, "max": 12},
                    "output_tokens": {"min": 3, "max": 4},
                    "backends": ["reference"],
                    "num_workers": 1,
                    "rate_rps": 1.0,
                    "duration_s": 1.0,
                    "fault_interval_s": 1.0,
                },
            },
        }
    )


# -- pure helpers --------------------------------------------------------------------------


def test_arm_labels_name_the_control_and_the_counts() -> None:
    assert arm_label(0) == BASE_LABEL
    assert arm_label(64) == "64 adapters"


def test_result_paths_carry_the_scenario_and_the_arm() -> None:
    path = result_path("results", "64 adapters")
    assert path.parent.name == SCENARIO
    assert path.name.endswith("-64-adapters.json")


def test_adapter_assignment_is_round_robin_and_even() -> None:
    class Prompt:
        def __init__(self, prompt_id: str) -> None:
            self.prompt_id = prompt_id

    prompts = [Prompt(f"p{i}") for i in range(6)]
    assignment = adapter_assignment(prompts, ["a", "b", "c"])  # type: ignore[arg-type]
    assert assignment == {"p0": "a", "p1": "b", "p2": "c", "p3": "a", "p4": "b", "p5": "c"}
    assert adapter_assignment(prompts, []) == {}  # type: ignore[arg-type]


def test_derived_block_reports_no_loss_without_a_control() -> None:
    summary = {"ttft_ms": {"p95": 20.0}, "tpot_ms": {"p95": 4.0}, "e2e_ms": {"p95": 40.0}}
    control = {"ttft_ms": {"p95": 10.0}, "tpot_ms": {"p95": 2.0}, "e2e_ms": {"p95": 20.0}}
    assert (
        derived_block(num_adapters=0, baseline=None, summary=control, vram=None, lora_stats=None)[
            "p95_ttft_loss_pct"
        ]
        is None
    )
    block = derived_block(
        num_adapters=8, baseline=control, summary=summary, vram={"saved_pct": 90.0}, lora_stats=None
    )
    assert block["p95_ttft_loss_pct"] == pytest.approx(100.0)
    assert block["p95_e2e_loss_pct"] == pytest.approx(100.0)
    assert block["vram"]["saved_pct"] == 90.0
    assert "lora" not in block


def test_derived_block_survives_a_summary_without_percentiles() -> None:
    block = derived_block(
        num_adapters=4,
        baseline={"ttft_ms": {}},
        summary={},
        vram=None,
        lora_stats={"lora_loads": 1},
    )
    assert block["p95_ttft_loss_pct"] is None
    assert block["lora"] == {"lora_loads": 1}


def test_a_non_positive_adapter_count_is_refused_before_anything_loads(
    tiny_qwen2_path: Path,
) -> None:
    with pytest.raises(ProfileError, match="must be positive"):
        asyncio.run(
            run_scenario(
                tiny_profile(tiny_qwen2_path),
                adapters_dir="nowhere",
                adapter_counts=[0],
                write=False,
            )
        )


# -- the whole scenario ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenario_arms(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> list[Any]:
    """Run the scenario once; every assertion below reads the same three arms."""
    adapters = request.getfixturevalue("adapters_dir")
    model_path = request.getfixturevalue("tiny_qwen2_path")
    results = tmp_path_factory.mktemp("results")
    return asyncio.run(
        run_scenario(
            tiny_profile(model_path),
            adapters_dir=adapters,
            results_dir=results,
            local_files_only=True,
            warmup_requests=1,
            hardware=FAKE_HARDWARE,
        )
    )


def test_every_arm_runs_every_request_without_a_failure(scenario_arms: list[Any]) -> None:
    assert [arm.label for arm in scenario_arms] == [BASE_LABEL, "2 adapters", "3 adapters"]
    for arm in scenario_arms:
        assert arm.summary["num_requests"] == 6, arm.label
        assert arm.summary["num_failed"] == 0, arm.label
        assert arm.summary["output_tokens"] > 0


def test_every_arm_names_the_control_as_its_baseline(scenario_arms: list[Any]) -> None:
    for arm in scenario_arms:
        assert arm.run.config["baseline_label"] == BASE_LABEL
        assert arm.run.config["label"] == arm.label
        assert arm.run.config["backend"] == "reference"
        assert arm.run.config["load"]["concurrency"] == 3
        assert arm.run.config["workload"]["adapter_counts"] == [2, 3]


def test_the_adapter_arms_really_used_their_adapters(scenario_arms: list[Any]) -> None:
    base, two, three = scenario_arms
    assert base.run.config["adapter_names"] == []
    assert two.run.config["adapter_names"] == ["tenant-0", "tenant-1"]
    assert three.run.config["adapter_names"] == ["tenant-0", "tenant-1", "tenant-2"]
    stats = three.summary["derived"]["lora"]
    assert stats["lora_num_adapters"] == 3
    assert stats["lora_adapter_tokens"] > 0
    assert stats["lora_loads"] >= 3


def test_the_vram_block_is_arithmetic_over_the_real_tensors(scenario_arms: list[Any]) -> None:
    base, _, three = scenario_arms
    assert base.summary["derived"].get("vram") is None
    report = three.summary["derived"]["vram"]
    assert report["num_adapters"] == 3
    assert report["merged_bytes"] == 3 * report["base_bytes"]
    assert report["lora_bytes"] == report["base_bytes"] + report["adapter_pool_bytes"]
    assert report["saved_bytes"] == report["merged_bytes"] - report["lora_bytes"]
    assert 0.0 < report["saved_pct"] < 100.0


def test_the_loss_figures_are_relative_to_the_control(scenario_arms: list[Any]) -> None:
    base, two, _ = scenario_arms
    assert base.summary["derived"]["p95_ttft_loss_pct"] is None
    assert isinstance(two.summary["derived"]["p95_ttft_loss_pct"], float)
    assert isinstance(two.summary["derived"]["p95_e2e_loss_pct"], float)


def test_the_files_round_trip_through_the_result_schema(scenario_arms: list[Any]) -> None:
    for arm in scenario_arms:
        assert arm.path is not None and arm.path.is_file()
        assert arm.path.parent.name == SCENARIO
        reloaded = RunResult.load(arm.path)
        assert reloaded.scenario == SCENARIO
        assert reloaded.provenance == "measured"
        assert reloaded.summary["derived"]["num_adapters"] == arm.num_adapters
        json.loads(arm.path.read_text())

    index = default_index_path(arm.path)
    assert index.is_file(), "every run must append to the results index"
    entries = json.loads(index.read_text())
    assert len(entries) == len(scenario_arms)


def test_an_unknown_backend_name_is_refused(tiny_qwen2_path: Path, adapters_dir: Path) -> None:
    with pytest.raises(ProfileError, match="--url"):
        asyncio.run(
            run_scenario(
                tiny_profile(tiny_qwen2_path),
                adapters_dir=adapters_dir,
                backend="vllm",
                url=None,
                write=False,
            )
        )
