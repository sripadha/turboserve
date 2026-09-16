"""Tests for the benchmark scenarios, their shared harness and the ``bench`` CLI.

Everything here runs on CPU in seconds. The end-to-end scenario tests drive the real
engines against a cached tiny-random Qwen2 checkpoint with a handful of very short
requests: the point is that a scenario produces a *valid, complete* result file with the
conventions the report renderer reads, not that the numbers in it mean anything. The
figures such a run produces are never published -- see PLAN.md 1 and 2a.

The load-generator test binds a real loopback socket and serves the mock gateway from a
uvicorn server in a thread, because the contract the Kubernetes end-to-end job depends on
is ``turboserve bench loadgen --url ... --rps ... --duration ... --out ...`` over HTTP, and
an in-process ASGI transport would not exercise it.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import typer
import yaml
from typer.testing import CliRunner

from turboserve.bench.cli import bench_app, register_optional_scenario
from turboserve.bench.profiles import load_profiles
from turboserve.bench.records import RunResult
from turboserve.bench.scenarios import chaos as chaos_scenario
from turboserve.bench.scenarios import naive_vs_cb, prefix_cache
from turboserve.bench.scenarios.common import (
    BaselineBackend,
    EngineOptions,
    ScenarioError,
    build_prompt_pool,
    gpu_price,
    normalise_base_url,
    resolve_slo,
    result_path,
)
from turboserve.bench.scenarios.loadgen_cli import parse_token_range
from turboserve.engine.core.types import FinishReason, RequestOutput, SamplingParams
from turboserve.gateway.backends.protocol import BackendError, GenerateRequest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

    from turboserve.bench.profiles import BenchProfile

# --------------------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://h:8000", "http://h:8000/v1"),
        ("http://h:8000/", "http://h:8000/v1"),
        ("http://h:8000/v1", "http://h:8000/v1"),
        ("http://h:8000/v1/", "http://h:8000/v1"),
        ("https://h/v2", "https://h/v2"),
    ],
)
def test_normalise_base_url_appends_the_version_prefix_once(given: str, expected: str) -> None:
    assert normalise_base_url(given) == expected


@pytest.mark.parametrize("bad", ["", "   ", "h:8000", "ftp://h/v1"])
def test_normalise_base_url_rejects_anything_that_is_not_an_http_url(bad: str) -> None:
    with pytest.raises(ScenarioError):
        normalise_base_url(bad)


@pytest.mark.parametrize(
    ("text", "expected"), [("128", (128, 128)), ("8:12", (8, 12)), (" 4 : 4 ", (4, 4))]
)
def test_parse_token_range_accepts_a_scalar_or_a_range(
    text: str, expected: tuple[int, int]
) -> None:
    assert parse_token_range(text, option="--input-tokens") == expected


@pytest.mark.parametrize("bad", ["0", "12:4", "a:b", "1:2:3", ""])
def test_parse_token_range_rejects_malformed_and_inverted_ranges(bad: str) -> None:
    with pytest.raises(ScenarioError):
        parse_token_range(bad, option="--input-tokens")


def test_tokenizer_free_prompts_have_exact_lengths_and_a_literally_shared_prefix() -> None:
    prompts = build_prompt_pool(
        count=6,
        input_tokens=(20, 24),
        output_tokens=(3, 5),
        seed=11,
        tenants=("a", "b"),
        shared_prefix_tokens=8,
        vocab_size=500,
    )
    assert len(prompts) == 6
    prefix = prompts[0].token_ids[:8]
    for prompt in prompts:
        assert 20 <= prompt.num_prompt_tokens <= 24
        assert 3 <= prompt.max_tokens <= 5
        assert prompt.token_ids[:8] == prefix
        assert prompt.shared_prefix_tokens == 8
        assert all(0 <= token < 500 for token in prompt.token_ids)
    assert [p.tenant for p in prompts] == ["a", "b", "a", "b", "a", "b"]


def test_tokenizer_free_prompts_are_reproducible_from_the_seed() -> None:
    kwargs: dict[str, Any] = {
        "count": 4,
        "input_tokens": (10, 16),
        "output_tokens": (2, 4),
        "seed": 5,
        "vocab_size": 300,
    }
    first = build_prompt_pool(**kwargs)
    second = build_prompt_pool(**kwargs)
    assert [p.token_ids for p in first] == [p.token_ids for p in second]


def test_a_prefix_as_long_as_the_shortest_prompt_is_rejected() -> None:
    with pytest.raises(ScenarioError, match="shortest prompt"):
        build_prompt_pool(
            count=2, input_tokens=(8, 12), output_tokens=(2, 2), seed=1, shared_prefix_tokens=8
        )


def test_result_path_names_the_scenario_and_slugs_the_arm(tmp_path: Path) -> None:
    path = result_path("prefix_cache", tmp_path, label="reference no cache")
    assert path.parent == tmp_path / "prefix_cache"
    assert path.name.endswith("-reference-no-cache.json")


def test_resolve_slo_prefers_overrides_and_returns_none_when_nothing_is_asserted() -> None:
    from turboserve.bench.profiles import SLOSpec

    assert resolve_slo(None) is None
    merged = resolve_slo(SLOSpec(ttft_ms=100.0, e2e_ms=900.0), ttft_ms=50.0)
    assert merged is not None
    assert (merged.ttft_ms, merged.e2e_ms, merged.tpot_ms) == (50.0, 900.0, None)


def test_gpu_price_reads_the_provisioning_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TURBOSERVE_GPU_PRICE_PER_HOUR", "1.25")
    monkeypatch.setenv("TURBOSERVE_GPU_PRICE_SOURCE", "synthetic test value")
    assert gpu_price() == (1.25, "synthetic test value")
    monkeypatch.setenv("TURBOSERVE_GPU_PRICE_PER_HOUR", "not-a-number")
    assert gpu_price() == (None, None)
    monkeypatch.delenv("TURBOSERVE_GPU_PRICE_PER_HOUR")
    assert gpu_price() == (None, None)


# --------------------------------------------------------------------------------------
# the baseline adapter, against a fake engine so the behaviour is visible
# --------------------------------------------------------------------------------------


class _FakeBaselineEngine:
    """A stand-in with the surface :class:`BaselineBackend` drives.

    It records the size of every batch it was asked to generate, which is how the tests
    below can assert that co-arriving requests really do land in one batch and that a
    request arriving mid-batch waits for the next one.
    """

    backend_name = "fake"

    def __init__(self, *, batch_size: int = 8, reject: set[str] | None = None) -> None:
        self.config = types.SimpleNamespace(model="fake-model")
        self.batch_size = batch_size
        self.reject = reject or set()
        self.queue: list[tuple[str, list[int], SamplingParams]] = []
        self.batches: list[int] = []
        self.closed = False

    def add_request(
        self,
        request_id: str,
        prompt: Any,
        sampling: SamplingParams | None = None,
        **_: Any,
    ) -> None:
        if request_id in self.reject:
            raise ValueError(f"request {request_id!r} refused by the fake engine")
        self.queue.append((request_id, list(prompt), sampling or SamplingParams()))

    def abort(self, request_id: str, **_: Any) -> bool:
        before = len(self.queue)
        self.queue = [item for item in self.queue if item[0] != request_id]
        return len(self.queue) != before

    def has_unfinished(self) -> bool:
        return bool(self.queue)

    def step(self) -> list[RequestOutput]:
        if not self.queue:
            return []
        take = min(self.batch_size, len(self.queue))
        batch, self.queue = self.queue[:take], self.queue[take:]
        self.batches.append(len(batch))
        time.sleep(0.02)  # a generation call is slow; this is what lets arrivals overlap
        outputs = []
        for request_id, prompt, sampling in batch:
            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    new_token_ids=[7] * sampling.max_tokens,
                    text_delta="x" * sampling.max_tokens,
                    finished=True,
                    finish_reason=FinishReason.LENGTH,
                    prompt_tokens=len(prompt),
                    output_tokens=sampling.max_tokens,
                )
            )
        return outputs

    def stats(self) -> dict[str, int | float | str]:
        return {"backend": self.backend_name, "num_batches": len(self.batches)}

    def close(self) -> None:
        self.closed = True


def _request(request_id: str, *, max_tokens: int = 3) -> GenerateRequest:
    return GenerateRequest(
        request_id=request_id,
        tenant_id="t",
        model="fake-model",
        prompt=[1, 2, 3, 4],
        sampling=SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True),
    )


async def _drain(backend: BaselineBackend, request: GenerateRequest) -> list[Any]:
    return [event async for event in backend.generate(request)]


async def test_baseline_backend_emits_exactly_one_terminating_event() -> None:
    engine = _FakeBaselineEngine()
    backend = BaselineBackend(engine, batch_window_s=0.0)
    try:
        events = await _drain(backend, _request("r1", max_tokens=4))
    finally:
        await backend.close()
    assert len(events) == 1
    event = events[0]
    assert event.finished is True
    assert event.finish_reason is FinishReason.LENGTH
    assert event.num_tokens == 4
    assert event.usage is not None
    assert event.usage["completion_tokens"] == 4
    assert engine.closed is True


async def test_baseline_backend_puts_co_arriving_requests_in_one_batch() -> None:
    engine = _FakeBaselineEngine(batch_size=8)
    backend = BaselineBackend(engine, batch_window_s=0.02)
    try:
        await asyncio.gather(*(_drain(backend, _request(f"r{i}")) for i in range(4)))
    finally:
        await backend.close()
    assert engine.batches == [4]


async def test_baseline_backend_makes_a_late_arrival_wait_for_the_next_batch() -> None:
    engine = _FakeBaselineEngine(batch_size=8)
    backend = BaselineBackend(engine, batch_window_s=0.0)

    async def late() -> list[Any]:
        await asyncio.sleep(0.01)  # lands while the first batch is already generating
        return await _drain(backend, _request("late"))

    try:
        await asyncio.gather(_drain(backend, _request("early")), late())
    finally:
        await backend.close()
    assert engine.batches == [1, 1]


async def test_baseline_backend_fails_a_request_the_engine_refuses() -> None:
    engine = _FakeBaselineEngine(reject={"bad"})
    backend = BaselineBackend(engine, batch_window_s=0.0)
    try:
        with pytest.raises(BackendError, match="refused"):
            await _drain(backend, _request("bad"))
        # the adapter survives one rejection and still serves the next request
        events = await _drain(backend, _request("good"))
        assert events[0].finished is True
    finally:
        await backend.close()


async def test_baseline_backend_reports_health_and_models_and_closes_idempotently() -> None:
    engine = _FakeBaselineEngine()
    backend = BaselineBackend(engine, batch_window_s=0.0)
    assert await backend.health() is True
    assert await backend.models() == ["fake-model"]
    await backend.close()
    await backend.close()
    assert backend.closed is True
    assert await backend.health() is False


# --------------------------------------------------------------------------------------
# end-to-end scenarios against the tiny checkpoint
# --------------------------------------------------------------------------------------


@pytest.fixture
def tiny_profile(tmp_path: Path, tiny_qwen2_path: Path) -> BenchProfile:
    """A profile sized so every scenario finishes in seconds on CPU.

    Built by editing the shipped ``dev-2060`` profile rather than by writing a new document,
    so the fixture keeps validating against the real schema: a field renamed in
    ``configs/bench/profiles.yaml`` breaks this test rather than silently diverging from it.
    """
    shipped = yaml.safe_load(Path("configs/bench/profiles.yaml").read_text(encoding="utf-8"))
    profile = shipped["profiles"]["dev-2060"]
    profile.update(device="cpu", seed=7, tenants=["t1", "t2"], description="synthetic test sizing")
    scenarios = profile["scenarios"]
    scenarios["naive_vs_cb"].update(
        model=str(tiny_qwen2_path),
        dtype="float32",
        num_requests=2,
        input_tokens={"min": 8, "max": 12},
        output_tokens={"min": 4, "max": 4},
        concurrencies=[2],
        backends=["naive_hf", "static_batch", "reference"],
    )
    scenarios["prefix_cache"].update(
        model=str(tiny_qwen2_path),
        dtype="float32",
        num_requests=4,
        shared_prefix_tokens=32,
        input_tokens={"min": 48, "max": 56},
        output_tokens={"min": 3, "max": 4},
        concurrency=2,
        prefix_caching=[False, True],
        backends=["reference"],
    )
    scenarios["spec_decode"].update(dtype="float32")
    scenarios["multi_lora"].update(model=str(tiny_qwen2_path), dtype="float32")
    scenarios["chaos"].update(
        model=str(tiny_qwen2_path),
        dtype="float32",
        num_workers=2,
        rate_rps=8.0,
        duration_s=1.0,
        fault_interval_s=0.5,
        input_tokens={"min": 4, "max": 6},
        output_tokens={"min": 2, "max": 3},
    )
    path = tmp_path / "profiles.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "profiles": {"tiny": profile}}), encoding="utf-8"
    )
    return load_profiles(path)["tiny"]


def _tiny_engine_options() -> EngineOptions:
    """A KV pool fixed by hand, so a CPU run does not size itself against host RAM."""
    return EngineOptions(
        dtype="float32",
        device="cpu",
        seed=7,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        block_size=16,
        num_blocks=64,
    )


def _loaded(outcome_path: Path) -> dict[str, Any]:
    payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    RunResult.from_dict(payload)  # the file round-trips through the schema
    return payload


def test_naive_vs_cb_runs_every_arm_and_writes_valid_result_files(
    tiny_profile: BenchProfile, tmp_path: Path
) -> None:
    outcomes = asyncio.run(
        naive_vs_cb.run_scenario(
            tiny_profile,
            options=_tiny_engine_options(),
            results_dir=tmp_path / "results",
            warmup=1,
            local_files_only=True,
        )
    )
    assert [o.backend for o in outcomes] == ["naive_hf", "static_batch", "reference"]
    labels = {o.label for o in outcomes}
    assert labels == {"naive", "static batch", "continuous batching"}
    for outcome in outcomes:
        assert outcome.error_rate == 0.0
        assert outcome.num_requests == 2
        payload = _loaded(outcome.path)
        assert payload["scenario"] == "naive_vs_cb"
        assert payload["provenance"] == "measured"
        config = payload["config"]
        assert config["label"] == outcome.label
        assert config["baseline_label"] == "naive"
        assert config["load"]["concurrency"] == 2
        assert config["output_tokens"] == 4
        summary = payload["summary"]
        assert summary["num_failed"] == 0
        assert summary["output_tok_s"] > 0
        assert summary["e2e_ms"]["p95"] is not None
        assert set(payload["requests"][0]) >= {"request_id", "t_send_ns", "t_last_ns", "ok"}


def test_naive_vs_cb_gives_every_arm_the_same_prompts_and_output_length(
    tiny_profile: BenchProfile, tmp_path: Path
) -> None:
    outcomes = asyncio.run(
        naive_vs_cb.run_scenario(
            tiny_profile,
            arms=["naive_hf", "reference"],
            options=_tiny_engine_options(),
            results_dir=tmp_path / "results",
            warmup=0,
            local_files_only=True,
        )
    )
    per_arm = {}
    for outcome in outcomes:
        payload = _loaded(outcome.path)
        records = sorted(payload["requests"], key=lambda r: r["request_id"].split("-")[-1])
        per_arm[outcome.backend] = [
            (r["request_id"].split("-")[-1], r["prompt_tokens"], r["output_tokens"])
            for r in records
        ]
    assert per_arm["naive_hf"] == per_arm["reference"]


def test_naive_vs_cb_rejects_an_arm_the_profile_does_not_declare(
    tiny_profile: BenchProfile, tmp_path: Path
) -> None:
    with pytest.raises(ScenarioError, match="vllm"):
        asyncio.run(
            naive_vs_cb.run_scenario(
                tiny_profile,
                arms=["vllm"],
                options=_tiny_engine_options(),
                results_dir=tmp_path / "results",
            )
        )


def test_prefix_cache_records_hits_only_when_the_cache_is_on(
    tiny_profile: BenchProfile, tmp_path: Path
) -> None:
    outcomes = asyncio.run(
        prefix_cache.run_scenario(
            tiny_profile,
            options=_tiny_engine_options(),
            results_dir=tmp_path / "results",
            warmup=1,
            local_files_only=True,
        )
    )
    by_label = {outcome.label: outcome for outcome in outcomes}
    assert set(by_label) == {"cache off", "cache on"}
    off, on = by_label["cache off"], by_label["cache on"]
    assert off.error_rate == 0.0 and on.error_rate == 0.0
    assert off.derived["num_cached_prompt_tokens"] == 0
    assert off.derived["prefix_hit_rate"] == 0.0
    assert on.derived["num_cached_prompt_tokens"] > 0
    assert on.derived["prefix_hit_rate"] > 0.0
    payload = _loaded(on.path)
    assert payload["config"]["shared_prefix_tokens"] == 32
    assert payload["config"]["baseline_label"] == "cache off"
    assert payload["summary"]["derived"]["prefix_caching"] is True


def test_prefix_cache_needs_a_url_for_a_vllm_arm(
    tiny_profile: BenchProfile, tmp_path: Path
) -> None:
    work = tiny_profile.scenarios.prefix_cache.model_copy(update={"backends": ["vllm"]})
    profile = tiny_profile.model_copy(
        update={"scenarios": tiny_profile.scenarios.model_copy(update={"prefix_cache": work})}
    )
    with pytest.raises(ScenarioError, match="--url"):
        asyncio.run(
            prefix_cache.run_scenario(
                profile, options=_tiny_engine_options(), results_dir=tmp_path / "results"
            )
        )


@pytest.mark.timeout(120)
def test_chaos_scenario_writes_a_result_with_the_renderer_conventions(
    tiny_profile: BenchProfile, tmp_path: Path
) -> None:
    outcome = asyncio.run(
        chaos_scenario.run_scenario(
            tiny_profile,
            results_dir=tmp_path / "results",
            mode="inprocess",
            faults=["kill:every=0.4s,grace=0.1s,restart=0.1s"],
        )
    )
    payload = _loaded(outcome.path)
    assert payload["scenario"] == "chaos"
    assert payload["config"]["backend"] == "mock"
    assert payload["config"]["label"] == "kill:every=0.4s,grace=0.1s,restart=0.1s"
    assert payload["config"]["load"]["mode"] == "open"
    derived = payload["summary"]["derived"]
    assert derived["replicas"] == 2
    assert derived["disruptions"] >= 1
    assert payload["summary"]["num_requests"] > 0
    assert "chaos" in payload["summary"]


def test_chaos_scenario_defaults_its_schedule_to_the_profiles_kill_cadence(
    tiny_profile: BenchProfile,
) -> None:
    spec = chaos_scenario.build_spec(tiny_profile, mode="inprocess")
    assert spec.replicas == 2
    assert spec.duration_s == 1.0
    assert [fault.kind for fault in spec.faults] == ["kill"]
    assert spec.faults.faults[0].every_s == 0.5
    assert spec.model == "mock-model"


# --------------------------------------------------------------------------------------
# the loadgen CLI, over a real socket
# --------------------------------------------------------------------------------------


@pytest.fixture
def mock_server() -> Iterator[str]:
    """A mock gateway on a loopback port, served by uvicorn in a background thread."""
    import uvicorn

    from turboserve.gateway.backends.mock import build_mock_app

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    app = build_mock_app(ttft_ms=1.0, itl_ms=0.5, max_tokens=6, accept_any_model=True)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 30.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:  # pragma: no cover - only on a wedged machine
        server.should_exit = True
        pytest.skip("the mock server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15.0)


@pytest.mark.timeout(120)
def test_loadgen_cli_drives_a_running_server_and_records_an_error_rate(
    mock_server: str, tmp_path: Path
) -> None:
    out = tmp_path / "run.json"
    result = CliRunner().invoke(
        bench_app,
        [
            "loadgen",
            "--url",
            mock_server,
            "--rps",
            "20",
            "--duration",
            "1.0",
            "--out",
            str(out),
            "--input-tokens",
            "8:12",
            "--output-tokens",
            "4",
            "--no-index",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _loaded(out)
    summary = payload["summary"]
    assert payload["scenario"] == "loadgen"
    assert summary["num_requests"] > 0
    assert summary["error_rate"] == 0.0
    assert summary["ttft_ms"]["p95"] is not None
    assert payload["config"]["url"] == f"{mock_server}/v1"
    assert payload["config"]["model"] == "mock-model"
    assert payload["config"]["load"]["mode"] == "open"
    assert not (tmp_path / "index.json").exists()  # --no-index was honoured


@pytest.mark.timeout(120)
def test_loadgen_cli_runs_a_closed_loop_when_no_rate_is_given(
    mock_server: str, tmp_path: Path
) -> None:
    out = tmp_path / "closed.json"
    result = CliRunner().invoke(
        bench_app,
        [
            "loadgen",
            "--url",
            mock_server,
            "--concurrency",
            "2",
            "--num-requests",
            "4",
            "--out",
            str(out),
            "--output-tokens",
            "3",
            "--no-index",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _loaded(out)
    assert payload["config"]["load"]["mode"] == "closed"
    assert payload["summary"]["num_requests"] == 4


def test_loadgen_cli_refuses_a_rate_with_no_duration() -> None:
    result = CliRunner().invoke(bench_app, ["loadgen", "--url", "http://127.0.0.1:1", "--rps", "5"])
    assert result.exit_code != 0
    assert "--duration" in result.output


# --------------------------------------------------------------------------------------
# the bench app itself
# --------------------------------------------------------------------------------------


def test_bench_app_exposes_the_shipped_commands() -> None:
    result = CliRunner().invoke(bench_app, ["--help"])
    assert result.exit_code == 0
    for command in ("loadgen", "naive-vs-cb", "prefix-cache", "chaos", "render", "profiles"):
        assert command in result.output


def test_bench_profiles_command_lists_the_shipped_profiles() -> None:
    result = CliRunner().invoke(bench_app, ["profiles"])
    assert result.exit_code == 0, result.output
    assert "h100" in result.output
    assert "dev-2060" in result.output


def test_register_optional_scenario_skips_a_module_that_is_not_installed() -> None:
    app = typer.Typer()
    attached = register_optional_scenario(app, "nope", "turboserve.bench.scenarios.nope", ("run",))
    assert attached is False
    assert app.registered_commands == []


def test_register_optional_scenario_attaches_a_command_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("turboserve.bench.scenarios._fake_optional")

    def spec_decode_command() -> None:
        """Synthetic stand-in for a scenario owned by another module group."""

    module.spec_decode_command = spec_decode_command  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    app = typer.Typer()
    attached = register_optional_scenario(
        app, "spec-decode", module.__name__, ("spec_decode_command",)
    )
    assert attached is True
    assert [command.name for command in app.registered_commands] == ["spec-decode"]


def test_register_optional_scenario_attaches_a_sub_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("turboserve.bench.scenarios._fake_group")
    module.multi_lora_app = typer.Typer()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    app = typer.Typer()
    assert register_optional_scenario(app, "multi-lora", module.__name__, ("multi_lora_app",))
    assert [group.name for group in app.registered_groups] == ["multi-lora"]


def _minimal_run(scenario: str, label: str, *, baseline: str) -> RunResult:
    run = RunResult.start(
        scenario,
        "tiny",
        config={
            "label": label,
            "backend": label,
            "baseline_label": baseline,
            "load": {"concurrency": 2},
        },
        hardware={"gpu": {"name": "synthetic test fixture"}, "git": {"sha": "0" * 40}},
    )
    from turboserve.bench.records import RequestRecord

    run.add(
        RequestRecord(
            request_id=f"{label}-0",
            prompt_tokens=8,
            output_tokens=4,
            t_send_ns=0,
            t_first_ns=1_000_000,
            t_last_ns=4_000_000,
            itl_ns=[1_000_000, 1_000_000, 1_000_000],
        )
    )
    run.finish()
    return run


def test_bench_render_writes_the_pages_from_result_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for label, baseline in (("naive", "naive"), ("continuous batching", "naive")):
        run = _minimal_run("naive_vs_cb", label, baseline="naive")
        run.save(results / "naive_vs_cb" / f"{label.replace(' ', '-')}.json")
        assert baseline == "naive"
    readme = tmp_path / "README.md"
    docs = tmp_path / "results.md"
    result = CliRunner().invoke(
        bench_app,
        [
            "render",
            "--results-dir",
            str(results),
            "--readme",
            str(readme),
            "--docs",
            str(docs),
            "--plots-dir",
            str(tmp_path / "plots"),
            "--no-plots",
        ],
    )
    assert result.exit_code == 0, result.output
    text = readme.read_text(encoding="utf-8")
    assert "naive_vs_cb" in text
    assert "continuous batching" in text
    assert docs.exists()


def test_scenario_package_exports_are_lazy_but_resolvable() -> None:
    import turboserve.bench.scenarios as package

    assert package.SCENARIO_MODULES == ("naive_vs_cb", "prefix_cache", "chaos")
    assert callable(package.naive_vs_cb_command)
    with pytest.raises(AttributeError):
        _ = package.does_not_exist
