"""The chaos harness end to end, and the breakable replicas it drives.

These are correctness tests of the gateway's failure handling -- retry before the first
token, abort after it, eject an unhealthy replica, put it back when it recovers -- not
benchmarks. Nothing here measures how fast anything is; the latencies are set to a couple of
milliseconds precisely so that the experiments finish in seconds and the assertions are about
*what happened*, never about how long it took.

Two tests run a real experiment (five and three seconds of open-loop load through the
router). They are deterministic in the things that matter: the arrival schedule, the victim
order, the prompts and the mock's own draws are all seeded, and the fault schedule used in
the headline test drains a replica before killing it, so no request can be lost to a stream
that was in flight. The one test that *does* kill a replica mid-stream drives a single
request and kills at a known point, so it does not depend on timing luck either.
"""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from typer.testing import CliRunner

from turboserve.bench.loadgen import build_requests
from turboserve.bench.prompts import BenchPrompt
from turboserve.bench.records import RunResult
from turboserve.chaos.faults import FaultSchedule
from turboserve.chaos.harness import (
    ChaosHarness,
    ChaosSpec,
    _RoutedLoad,
    chaos_app,
    default_result_path,
)
from turboserve.chaos.worker import (
    InProcessWorker,
    SubprocessWorker,
    WorkerFaults,
    WorkerSpec,
    build_worker_app,
    build_workers,
)
from turboserve.engine.core.types import FinishReason, SamplingParams
from turboserve.gateway.backends.protocol import (
    BackendUnavailableError,
    GenerateRequest,
    StreamInterruptedError,
)
from turboserve.gateway.router import Router

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator

runner = CliRunner()

#: No real hardware is inspected: collecting it shells out to nvidia-smi and git, which a
#: unit test has no business doing, and none of these assertions look at it.
FAKE_HARDWARE = {"schema_version": 1, "gpu_name": None, "host": {"hostname": "test"}}


def free_port() -> int:
    """A port that was free a moment ago, for the one test that needs a real socket."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    return port


def one_request(request_id: str = "req-1", *, max_tokens: int = 4) -> GenerateRequest:
    """A single generation request aimed at the replicas' model."""
    return GenerateRequest(
        request_id=request_id,
        tenant_id="chaos",
        model="mock-model",
        prompt=[11, 12, 13],
        sampling=SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True),
    )


async def drain_stream(stream: AsyncIterator[object]) -> list[object]:
    """Collect every event of a stream."""
    return [event async for event in stream]


# -- the breakable replica ---------------------------------------------------------------


async def test_a_worker_serves_the_real_gateway_app_with_a_control_plane() -> None:
    app = build_worker_app(WorkerSpec(name="replica-0", port=9101, max_tokens=4))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://replica") as client:
        assert (await client.get("/readyz")).status_code == httpx.codes.OK
        body = {"model": "mock-model", "prompt": "hello", "max_tokens": 4, "stream": False}
        completion = await client.post("/v1/completions", json=body)
        assert completion.status_code == httpx.codes.OK
        assert completion.json()["choices"][0]["text"]

        state = (await client.get("/chaos/state")).json()
        assert state["name"] == "replica-0"
        assert state["alive"] is True
        assert state["stats"]["requests"] == 1
        assert state["mock"]["completed"] == 1


async def test_a_partition_blocks_the_data_plane_but_leaves_the_control_plane_reachable() -> None:
    app = build_worker_app(WorkerSpec(name="replica-0", port=9102, max_tokens=4))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://replica") as client:
        assert (await client.post("/chaos/partition", json={"partitioned": True})).status_code == (
            httpx.codes.OK
        )
        # Everything the gateway would touch is gone, health probes included: a partitioned
        # replica is not selectively unreachable.
        assert (await client.get("/readyz")).status_code == httpx.codes.SERVICE_UNAVAILABLE
        assert (await client.get("/healthz")).status_code == httpx.codes.SERVICE_UNAVAILABLE
        body = {"model": "mock-model", "prompt": "hello", "max_tokens": 4, "stream": False}
        assert (await client.post("/v1/completions", json=body)).status_code == (
            httpx.codes.SERVICE_UNAVAILABLE
        )
        # The operator's channel still works, which is how the partition is lifted again.
        assert (await client.post("/chaos/reset")).status_code == httpx.codes.OK
        assert (await client.get("/readyz")).status_code == httpx.codes.OK


async def test_an_injected_error_is_raised_before_the_first_token_and_is_retryable() -> None:
    worker = InProcessWorker(WorkerSpec(name="replica-0"), WorkerFaults(error_probability=1.0))
    with pytest.raises(BackendUnavailableError) as raised:
        await drain_stream(worker.backend.generate(one_request()))
    assert raised.value.retryable is True
    assert "injected" in str(raised.value)


async def test_latency_injection_delays_the_chosen_fraction_of_requests() -> None:
    worker = InProcessWorker(
        WorkerSpec(name="replica-0", max_tokens=2),
        WorkerFaults(latency_probability=1.0, latency_ms=25.0),
    )
    loop = asyncio.get_running_loop()
    start = loop.time()
    await drain_stream(worker.backend.generate(one_request(max_tokens=2)))
    assert loop.time() - start >= 0.025
    assert worker.backend.stats.delayed == 1


async def test_a_draining_replica_refuses_new_work_and_reports_unhealthy() -> None:
    worker = InProcessWorker(WorkerSpec(name="replica-0"))
    await worker.drain()
    assert await worker.backend.health() is False
    with pytest.raises(BackendUnavailableError, match="draining"):
        await drain_stream(worker.backend.generate(one_request()))
    assert worker.backend.stats.refused == 1
    # A drain is reversible: that is the whole difference between it and a kill.
    await worker.restart()
    assert await worker.backend.health() is True


async def test_the_same_request_id_gets_the_same_fate_from_the_same_replica() -> None:
    faults = WorkerFaults(error_probability=0.5)
    first = InProcessWorker(WorkerSpec(name="replica-0", seed=99), faults)
    second = InProcessWorker(WorkerSpec(name="replica-0", seed=99), faults)

    async def survives(worker: InProcessWorker, request_id: str) -> bool:
        try:
            await drain_stream(worker.backend.generate(one_request(request_id)))
        except BackendUnavailableError:
            return False
        return True

    ids = [f"req-{index}" for index in range(24)]
    outcomes = [await survives(first, request_id) for request_id in ids]
    replayed = [await survives(second, request_id) for request_id in ids]
    assert outcomes == replayed
    assert 0 < sum(outcomes) < len(ids), "a p=0.5 injection must not be all-or-nothing"


def test_a_worker_spec_renders_the_command_line_that_starts_it() -> None:
    argv = WorkerSpec(name="replica-3", port=9199, ttft_ms=12.5).argv("python3")
    assert argv[:3] == ["python3", "-m", "turboserve.chaos.worker"]
    assert "--port" in argv and "9199" in argv
    assert argv[argv.index("--ttft-ms") + 1] == "12.5"
    assert WorkerSpec(name="w", port=9199).base_url == "http://127.0.0.1:9199/v1"


def test_a_subprocess_worker_needs_a_port_and_the_mode_must_be_known() -> None:
    with pytest.raises(ValueError, match="real port"):
        SubprocessWorker(WorkerSpec(name="replica-0", port=0))
    with pytest.raises(ValueError, match="inprocess"):
        build_workers([WorkerSpec(name="replica-0")], mode="threads")


# -- the router's failure handling --------------------------------------------------------


async def test_a_dead_replica_is_retried_away_before_the_first_token() -> None:
    """The headline guarantee: a replica that dies between health probes costs no request.

    The health cache is frozen (a very long TTL) so the router still believes the killed
    replica is up, which is exactly the window a real kill happens in. Every request routed
    at the corpse must come back from the other replica instead, and the replica must be
    ejected after the first failure rather than retried into forever.
    """
    spec = ChaosSpec(
        replicas=2,
        duration_s=1.0,
        rate_rps=10.0,
        mode="inprocess",
        ttft_ms=1.0,
        itl_ms=0.0,
        health_ttl_s=3600.0,
        seed=5,
    )
    harness = ChaosHarness(spec)
    attempts: dict[str, list[str]] = {}
    router = harness.build_router(attempts)
    for worker in harness.workers:
        await worker.start()
    await router.health_report()
    await harness.workers[0].kill()

    tracker = _RoutedLoad(router, t0_ns=0)
    prompts = [
        BenchPrompt(prompt_id=f"p{index}", token_ids=[7] * 8, max_tokens=4, tenant="chaos")
        for index in range(20)
    ]
    for request in build_requests(prompts, model=spec.model):
        await drain_stream(tracker.generate(request))

    assert all(completion.ok for completion in tracker.completions)
    retries = sum(len(tried) - 1 for tried in attempts.values())
    assert retries >= 1, "at least one request must have been sent at the dead replica"
    assert attempts["p0"] == ["replica-0", "replica-1"] or retries >= 1
    served = [name for tried in attempts.values() for name in tried]
    assert served.count("replica-0") == retries, "a failed replica must be tried once, then left"
    for worker in harness.workers:
        await worker.stop()


async def test_a_hard_kill_aborts_the_stream_it_had_in_flight_and_is_not_retried() -> None:
    """The failure a perfect gateway still reports, and why an ungraceful kill costs requests.

    Once tokens have been sent, another replica cannot take over without duplicating them, so
    the stream ends with a terminating error event. That is the floor on the error rate of a
    ``grace=0`` schedule, and the reason the harness reports failures by cause.
    """
    worker = InProcessWorker(WorkerSpec(name="replica-0", ttft_ms=1.0, itl_ms=30.0, max_tokens=10))
    router = Router(health_ttl_s=3600.0, max_attempts=3)
    router.add_backend("mock-model", worker.backend)
    await router.health_report()

    seen: list[object] = []

    async def drive() -> None:
        async for routed in router.generate(one_request(max_tokens=10)):
            seen.append(routed)

    async def kill_once_it_is_streaming() -> None:
        await asyncio.sleep(0.05)
        await worker.kill()

    await asyncio.gather(drive(), kill_once_it_is_streaming())

    assert len(seen) >= 2, "the client must have received tokens before the kill"
    last = seen[-1]
    assert last.finished and last.is_error  # type: ignore[attr-defined]
    assert last.event.finish_reason is FinishReason.ABORT  # type: ignore[attr-defined]
    assert "killed with a stream in flight" in str(last.event.error)  # type: ignore[attr-defined]
    assert worker.backend.stats.aborted == 1
    assert isinstance(StreamInterruptedError("x"), Exception)


# -- whole experiments ---------------------------------------------------------------------


async def test_two_replicas_survive_repeated_kills_with_a_negligible_error_rate() -> None:
    """Five seconds of load through two replicas while one of them is killed every two seconds.

    The schedule drains for 300 ms before each kill, which is what a Kubernetes pod deletion
    with a readiness gate and a preStop hook does; with it, nothing is in flight when the
    process dies and every request is either served or retried onto the survivor. The
    threshold asserted here is the one ``deploy/kind/assert_error_rate.py`` enforces on the
    in-cluster version of the same experiment.
    """
    spec = ChaosSpec(
        faults=FaultSchedule.parse(["kill:every=2s,grace=300ms,restart=400ms"]),
        replicas=2,
        duration_s=5.0,
        rate_rps=60.0,
        mode="inprocess",
        ttft_ms=2.0,
        itl_ms=0.0,
        input_tokens=(16, 32),
        output_tokens=(4, 8),
        health_ttl_s=0.25,
        seed=7,
    )
    report = await ChaosHarness(spec, hardware=FAKE_HARDWARE).run()

    assert report.num_requests > 100, "the run must actually have applied load"
    assert report.error_rate < 0.005, report.chaos["failures_by_cause"]
    assert len(report.disruptions) == 2
    assert len(report.recoveries_s) == 2, "both replicas must have been used again"
    assert all(recovery > 0.0 for recovery in report.recoveries_s)

    chaos = report.chaos
    assert chaos["attempts"] >= report.num_requests
    assert chaos["requests_never_routed"] == 0
    assert set(chaos["attempts_per_replica"]) == {"replica-0", "replica-1"}
    assert min(chaos["attempts_per_replica"].values()) > 0, "both replicas must have served"
    assert chaos["fleet_outage_windows_s"] == []
    assert chaos["requests_during_faults"] > 0
    assert chaos["during_faults"]["num_requests"] + chaos["steady_state"]["num_requests"] == (
        report.num_requests
    )
    lifecycles = [worker["lifecycle"] for worker in chaos["workers"]]
    assert sum(lifecycle["kills"] for lifecycle in lifecycles) == 2
    assert sum(lifecycle["restarts"] for lifecycle in lifecycles) == 2


async def test_a_partition_is_recorded_as_a_disruption_and_is_lifted_again() -> None:
    spec = ChaosSpec(
        faults=FaultSchedule.parse(["partition:at=1s,for=1s,target=replica-0"]),
        replicas=2,
        duration_s=3.0,
        rate_rps=40.0,
        mode="inprocess",
        ttft_ms=2.0,
        itl_ms=0.0,
        input_tokens=(8, 16),
        output_tokens=(4, 4),
        health_ttl_s=0.25,
        seed=13,
    )
    report = await ChaosHarness(spec, hardware=FAKE_HARDWARE).run()

    assert report.error_rate < 0.005, report.chaos["failures_by_cause"]
    assert [disruption.target for disruption in report.disruptions] == ["replica-0"]
    disruption = report.disruptions[0]
    assert disruption.kind == "partition"
    assert disruption.restored_s is not None
    assert disruption.downtime_s == pytest.approx(1.0, abs=0.3)
    assert disruption.recovery_s is not None
    assert all(worker.alive for worker in ChaosHarness(spec).workers)


async def test_injected_errors_are_retried_away_across_a_fleet_of_three() -> None:
    """A replica failing one request in twenty must not fail one client request in twenty.

    Each attempt is drawn independently by the replica that serves it, so the retry path is
    what turns a per-replica failure rate into a much smaller client-visible one. Asserting
    the exact residual would be asserting the arithmetic of a seeded RNG; asserting that it
    stays below the injected rate is asserting that the retries happened at all.

    Three replicas, not two, and that is not an arbitrary choice: the router ejects a replica
    on its first failure and believes that verdict for ``health_ttl_s``, so a fleet of two
    has no margin left once one of them has failed once. The next test pins that mechanism
    down; this one shows that a fleet with a spare absorbs the same injection.
    """
    spec = ChaosSpec(
        faults=FaultSchedule.parse(["error:p=0.05"]),
        replicas=3,
        duration_s=2.0,
        rate_rps=50.0,
        mode="inprocess",
        ttft_ms=1.0,
        itl_ms=0.0,
        input_tokens=(8, 16),
        output_tokens=(4, 4),
        max_attempts=3,
        health_ttl_s=0.25,
        seed=17,
    )
    report = await ChaosHarness(spec, hardware=FAKE_HARDWARE).run()

    chaos = report.chaos
    injected = sum(worker["stats"]["injected_errors"] for worker in chaos["workers"])
    assert injected > 0, "the injection must have fired"
    assert report.retries >= injected - report.num_failed
    assert report.error_rate < 0.05, chaos["failures_by_cause"]
    assert set(chaos["failures_by_cause"]) <= {"injected_error", "no_healthy_replica"}


async def test_a_replica_is_ejected_on_its_first_failure_until_it_is_probed_again() -> None:
    """Why fleet size and the health TTL decide what a uniform error rate costs.

    The router condemns a replica that fails a request and believes that verdict for
    ``health_ttl_s`` without asking again. That is what makes a single bad replica cheap --
    and what makes a fleet with no spare expensive, because once every member has failed one
    request there is nothing left to route to until the cache expires. A chaos run surfaces
    this as ``no_healthy_replica`` in ``failures_by_cause``.
    """
    workers = [
        InProcessWorker(WorkerSpec(name=f"replica-{index}"), WorkerFaults(error_probability=1.0))
        for index in range(2)
    ]
    frozen = Router(health_ttl_s=3600.0, max_attempts=3)
    for worker in workers:
        frozen.add_backend("mock-model", worker.backend)
    await frozen.health_report()

    with pytest.raises(BackendUnavailableError):
        await drain_stream(frozen.generate(one_request("doomed")))

    for worker in workers:
        worker.faults.error_probability = 0.0
    with pytest.raises(BackendUnavailableError, match="no healthy backend"):
        await drain_stream(frozen.generate(one_request("still-ejected")))

    # The same fleet behind a router that re-probes every time serves the request happily,
    # which isolates the cause to the cached verdict rather than to the replicas.
    reprobing = Router(health_ttl_s=0.0, max_attempts=3)
    for worker in workers:
        reprobing.add_backend("mock-model", worker.backend)
    events = await drain_stream(reprobing.generate(one_request("after-reprobe")))
    assert events and events[-1].event.finished  # type: ignore[attr-defined]


async def test_a_run_writes_an_ordinary_result_file_with_a_chaos_block(tmp_path: Path) -> None:
    spec = ChaosSpec(
        faults=FaultSchedule.parse(["kill:every=1s,grace=200ms,restart=300ms"]),
        replicas=2,
        duration_s=2.0,
        rate_rps=30.0,
        mode="inprocess",
        ttft_ms=1.0,
        itl_ms=0.0,
        input_tokens=(8, 16),
        output_tokens=(4, 4),
        health_ttl_s=0.25,
        profile="dev-2060",
        seed=23,
    )
    report = await ChaosHarness(spec, hardware=FAKE_HARDWARE).run()
    path = report.save(tmp_path / "chaos" / "run.json")

    reloaded = RunResult.load(path)
    assert reloaded.scenario == "chaos"
    assert reloaded.profile == "dev-2060"
    assert reloaded.provenance == "measured"
    assert len(reloaded.requests) == report.num_requests
    # The fields deploy/kind/assert_error_rate.py reads.
    assert reloaded.summary["num_requests"] == report.num_requests
    assert reloaded.summary["num_failed"] == report.num_failed

    chaos = reloaded.summary["chaos"]
    assert chaos["schedule"]["spec"] == "kill:every=1s,grace=200ms,restart=300ms"
    assert chaos["replica_engine"] == "mock"
    assert chaos["load"]["mode"] == "open"
    assert [event["action"] for event in chaos["fault_events"]][:3] == [
        "drain",
        "kill",
        "restart",
    ]
    assert chaos["disruptions"], "a killed replica must be recorded as a disruption"
    assert reloaded.config["workload"]["faults"]["spec"].startswith("kill:")
    # Every record is attributed to the replica that served it.
    served = {record.backend for record in reloaded.requests if record.ok}
    assert served <= {"replica-0", "replica-1"}

    index = json.loads((tmp_path / "chaos" / "index.json").read_text(encoding="utf-8"))
    assert index[0]["scenario"] == "chaos"


@pytest.mark.timeout(120)
async def test_a_subprocess_replica_really_dies_and_really_comes_back() -> None:
    """The one test that crosses a socket: the HTTP path, a real SIGKILL, a real restart.

    In-process kills cannot prove that the gateway's HTTP client turns a refused connection
    into a retryable error, and that is the failure a Kubernetes pod deletion actually
    produces, so it is worth the couple of seconds two process starts cost.
    """
    worker = SubprocessWorker(WorkerSpec(name="replica-0", port=free_port(), max_tokens=4))
    router = Router(health_ttl_s=0.0, max_attempts=1)
    router.add_backend("mock-model", worker.backend)
    try:
        await worker.start()
        assert worker.alive
        assert await router.health_report() == {"mock-model": {"replica-0": True}}

        events = [routed async for routed in router.generate(one_request("live-1"))]
        assert events and events[-1].event.finished

        await worker.set_partitioned(True)
        assert await router.health_report() == {"mock-model": {"replica-0": False}}
        await worker.set_partitioned(False)

        await worker.kill()
        assert not worker.alive
        assert await router.health_report() == {"mock-model": {"replica-0": False}}
        with pytest.raises(BackendUnavailableError):
            await drain_stream(router.generate(one_request("dead-1")))

        await worker.restart()
        assert worker.alive
        assert await router.health_report() == {"mock-model": {"replica-0": True}}
        events = [routed async for routed in router.generate(one_request("live-2"))]
        assert events and events[-1].event.finished
        state = await worker.state()
        assert state["stats"]["requests"] == 1, "the restarted replica starts from scratch"
    finally:
        await worker.stop()
        await router.close()


# -- spec, CLI and paths --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replicas": 0},
        {"duration_s": 0.0},
        {"rate_rps": 0.0},
        {"mode": "threads"},
        {"input_tokens": (0, 4)},
        {"output_tokens": (8, 4)},
        {"max_attempts": 0},
        {"pool_size": -1},
    ],
)
def test_the_spec_refuses_an_impossible_experiment(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ChaosSpec(**kwargs)  # type: ignore[arg-type]


def test_the_spec_sizes_its_prompt_pool_and_names_its_replicas() -> None:
    spec = ChaosSpec(replicas=3, rate_rps=20.0, mode="subprocess", base_port=9300)
    assert spec.prompt_pool_size == 100
    specs = spec.worker_specs()
    assert [worker.name for worker in specs] == ["replica-0", "replica-1", "replica-2"]
    assert [worker.port for worker in specs] == [9300, 9301, 9302]
    assert len({worker.seed for worker in specs}) == 3, "replicas must not be identical twins"
    assert ChaosSpec(replicas=1, mode="inprocess").worker_specs()[0].port == 0


def test_the_spec_describes_itself_for_the_result_file() -> None:
    spec = ChaosSpec(faults=FaultSchedule.parse(["kill:every=10s"]), replicas=4)
    data = spec.to_dict()
    assert data["replicas"] == 4
    assert data["replica_engine"] == "mock"
    assert data["faults"]["spec"] == "kill:every=10s"


def test_the_default_result_path_lands_under_the_results_directory() -> None:
    from datetime import UTC, datetime

    stamp = datetime(2026, 9, 16, 12, 30, 45, tzinfo=UTC)
    assert default_result_path("results", now=stamp) == Path("results/chaos/20260916T123045Z.json")


def test_the_plan_command_prints_the_timeline_without_running_anything() -> None:
    result = runner.invoke(
        chaos_app,
        [
            "plan",
            "--replicas",
            "3",
            "--duration",
            "30s",
            "--faults",
            "kill:every=10s",
            "--faults",
            "error:p=0.01",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "kill" in result.stdout
    assert "restart" in result.stdout
    assert "replica-" in result.stdout
    assert "error p=0.01" in result.stdout


def test_the_plan_command_warns_about_a_schedule_that_empties_the_fleet() -> None:
    result = runner.invoke(
        chaos_app,
        [
            "plan",
            "--replicas",
            "2",
            "--duration",
            "20s",
            "--faults",
            "kill:every=5s,restart=30s",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "warning" in result.stdout.lower()


def test_the_cli_rejects_a_bad_fault_specification_and_a_bad_mode() -> None:
    bad_fault = runner.invoke(chaos_app, ["plan", "--faults", "kill:evry=10s"])
    assert bad_fault.exit_code != 0
    bad_mode = runner.invoke(chaos_app, ["run", "--mode", "threads", "--duration", "1s"])
    assert bad_mode.exit_code != 0
