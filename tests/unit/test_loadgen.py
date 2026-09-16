"""Load generator tests: arrival statistics, concurrency bounds, and failure accounting.

The streams these tests drive are synthetic: small async generators standing in for a
backend, with configurable pacing, chunk sizes and failure modes. They exercise the
:class:`~turboserve.gateway.backends.protocol.Backend` contract (an async iterator of
``TokenEvent`` whose last event is finished) without a model, a GPU or a socket.
"""

from __future__ import annotations

import asyncio
import statistics
from typing import TYPE_CHECKING, Any

import pytest

from turboserve.bench.loadgen import (
    LoadResult,
    LoadSpec,
    build_requests,
    load_config,
    poisson_offsets,
    run_closed_loop,
    run_load,
    run_open_loop,
)
from turboserve.bench.prompts import BenchPrompt
from turboserve.engine.core.types import FinishReason, SamplingParams
from turboserve.gateway.backends.protocol import (
    BackendOverloadedError,
    GenerateRequest,
    TokenEvent,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator, Callable


def make_prompts(count: int, *, tokens: int = 8, max_tokens: int = 4) -> list[BenchPrompt]:
    """Prompts with deterministic ids and known token counts (no tokenizer needed)."""
    return [
        BenchPrompt(
            prompt_id=f"p{index:03d}",
            token_ids=list(range(100, 100 + tokens)),
            text="x" * tokens,
            max_tokens=max_tokens,
            tenant=f"tenant-{index % 2}",
        )
        for index in range(count)
    ]


def make_backend(
    *,
    tokens: int = 4,
    tokens_per_chunk: int = 1,
    delay: float = 0.001,
    first_delay: float | None = None,
    usage: bool = True,
    raise_error: Exception | None = None,
    never_finish: bool = False,
    hang: bool = False,
    closed: list[str] | None = None,
) -> Callable[[GenerateRequest], AsyncIterator[TokenEvent]]:
    """Build a stand-in for ``Backend.generate`` with the behaviour a test needs.

    ``closed`` collects the ids of requests whose iterator was closed before it finished,
    which is how the "cancelling the stream aborts the work" obligation is observed.
    """

    async def generate(req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        emitted = 0
        finished_cleanly = False
        try:
            if raise_error is not None:
                raise raise_error
            if hang:
                await asyncio.sleep(3600)
            while emitted < tokens:
                await asyncio.sleep(first_delay if emitted == 0 and first_delay else delay)
                chunk = min(tokens_per_chunk, tokens - emitted)
                ids = list(range(emitted, emitted + chunk))
                emitted += chunk
                if emitted >= tokens and not never_finish:
                    yield TokenEvent.final(
                        req.request_id,
                        FinishReason.LENGTH,
                        token_ids=ids,
                        usage={"prompt_tokens": 7, "completion_tokens": tokens} if usage else None,
                    )
                    finished_cleanly = True
                else:
                    yield TokenEvent.delta(req.request_id, ids)
        finally:
            if closed is not None and not finished_cleanly:
                closed.append(req.request_id)

    return generate


# -- arrival statistics -------------------------------------------------------------------


def test_poisson_offsets_are_deterministic_and_start_at_zero() -> None:
    first = poisson_offsets(10.0, 50, seed=11)
    second = poisson_offsets(10.0, 50, seed=11)
    assert first == second
    assert first[0] == 0.0
    assert first == sorted(first)
    assert poisson_offsets(10.0, 50, seed=12) != first


def test_poisson_gaps_match_an_exponential_distribution() -> None:
    """Mean gap is 1/rate and about 63.2% of gaps fall below it -- the Exp(1) signature."""
    rate = 25.0
    offsets = poisson_offsets(rate, 20_000, seed=7)
    gaps = [later - earlier for earlier, later in zip(offsets, offsets[1:], strict=False)]
    mean = statistics.fmean(gaps)
    assert mean == pytest.approx(1.0 / rate, rel=0.03)
    # For Exp(lambda), P(X < 1/lambda) = 1 - e^-1 = 0.632.
    below_mean = sum(1 for gap in gaps if gap < 1.0 / rate) / len(gaps)
    assert below_mean == pytest.approx(0.632, abs=0.02)
    # The standard deviation of an exponential equals its mean.
    assert statistics.stdev(gaps) == pytest.approx(mean, rel=0.05)


def test_poisson_offsets_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="rate_rps"):
        poisson_offsets(0.0, 10)


# -- request construction -----------------------------------------------------------------


def test_build_requests_sends_token_ids_and_greedy_sampling_by_default() -> None:
    prompts = make_prompts(3, tokens=5, max_tokens=9)
    requests = build_requests(prompts, model="m", id_prefix="run1-")
    assert [req.request_id for req in requests] == ["run1-p000", "run1-p001", "run1-p002"]
    assert requests[0].prompt_is_tokens
    assert requests[0].prompt_token_ids == prompts[0].token_ids
    assert requests[0].tenant_id == "tenant-0"
    assert requests[0].sampling.max_tokens == 9
    assert requests[0].sampling.is_greedy
    assert requests[0].sampling.ignore_eos


def test_build_requests_can_send_text_and_per_prompt_adapters() -> None:
    prompts = make_prompts(4)
    requests = build_requests(
        prompts,
        model="m",
        send_text=True,
        lora=lambda prompt: f"adapter-{prompt.prompt_id[-1]}",
        sampling=SamplingParams(max_tokens=3),
    )
    assert requests[0].prompt_text == prompts[0].text
    assert [req.lora for req in requests] == ["adapter-0", "adapter-1", "adapter-2", "adapter-3"]
    assert all(req.sampling.max_tokens == 3 for req in requests)


def test_build_requests_rejects_an_empty_token_prompt() -> None:
    prompt = BenchPrompt(prompt_id="empty", token_ids=[], text="hello")
    with pytest.raises(ValueError, match="no token ids"):
        build_requests([prompt], model="m")


# -- closed loop --------------------------------------------------------------------------


async def test_closed_loop_records_every_request_and_respects_its_bound() -> None:
    requests = build_requests(make_prompts(20), model="m")
    spec = LoadSpec(mode="closed", concurrency=4, backend_name="mock")
    result = await run_closed_loop(requests, make_backend(tokens=3, delay=0.002), spec=spec)

    assert len(result.records) == 20
    assert result.max_in_flight_observed == 4
    assert all(record.ok for record in result.records)
    assert {record.backend for record in result.records} == {"mock"}
    assert {record.lane for record in result.records} == {"stable"}
    record = result.records[0]
    assert record.output_tokens == 3
    assert record.prompt_tokens == 7  # from the backend's usage block
    assert record.ttft_ms is not None and record.ttft_ms > 0
    assert len(record.itl_ns) == 2
    assert record.e2e_ms is not None and record.e2e_ms >= record.ttft_ms
    assert result.wall_seconds > 0


async def test_closed_loop_falls_back_to_the_request_length_without_usage() -> None:
    requests = build_requests(make_prompts(2, tokens=11), model="m")
    spec = LoadSpec(concurrency=2)
    result = await run_closed_loop(requests, make_backend(tokens=2, usage=False), spec=spec)
    assert {record.prompt_tokens for record in result.records} == {11}
    assert {record.output_tokens for record in result.records} == {2}


async def test_closed_loop_stops_at_the_duration_deadline() -> None:
    requests = build_requests(make_prompts(200), model="m")
    spec = LoadSpec(concurrency=2, duration_s=0.05)
    result = await run_closed_loop(requests, make_backend(tokens=2, delay=0.01), spec=spec)
    assert 0 < len(result.records) < 200


async def test_empty_request_list_produces_an_empty_result() -> None:
    result = await run_closed_loop([], make_backend(), spec=LoadSpec())
    assert result == LoadResult()


# -- open loop ----------------------------------------------------------------------------


async def test_open_loop_follows_the_planned_poisson_schedule() -> None:
    requests = build_requests(make_prompts(40), model="m")
    spec = LoadSpec(mode="open", rate_rps=200.0, seed=3, concurrency=1)
    result = await run_open_loop(requests, make_backend(tokens=1, delay=0.0), spec=spec)

    assert len(result.records) == 40
    planned = poisson_offsets(200.0, 40, seed=3)
    origin = min(record.t_send_ns for record in result.records)
    actual = sorted((record.t_send_ns - origin) / 1e9 for record in result.records)
    # The driver sleeps to absolute offsets, so it may be late but never early.
    pairs = zip(actual, planned, strict=True)
    assert all(observed >= expected - 0.01 for observed, expected in pairs)
    assert actual[-1] == pytest.approx(planned[-1], abs=0.5)


async def test_open_loop_lets_concurrency_grow_beyond_one() -> None:
    """The point of an open loop: arrivals do not wait for completions."""
    requests = build_requests(make_prompts(30), model="m")
    spec = LoadSpec(mode="open", rate_rps=500.0, seed=5)
    result = await run_open_loop(requests, make_backend(tokens=2, delay=0.02), spec=spec)
    assert result.max_in_flight_observed > 1
    assert len(result.records) == 30


async def test_open_loop_honours_max_in_flight() -> None:
    requests = build_requests(make_prompts(30), model="m")
    spec = LoadSpec(mode="open", rate_rps=1000.0, seed=5, max_in_flight=3)
    result = await run_open_loop(requests, make_backend(tokens=2, delay=0.005), spec=spec)
    assert result.max_in_flight_observed <= 3
    assert len(result.records) == 30


async def test_open_loop_repeats_the_pool_with_distinct_ids() -> None:
    requests = build_requests(make_prompts(3), model="m")
    spec = LoadSpec(mode="open", rate_rps=400.0, duration_s=0.15, repeat_requests=True, seed=1)
    result = await run_open_loop(requests, make_backend(tokens=1, delay=0.0), spec=spec)
    ids = [record.request_id for record in result.records]
    assert len(ids) > 3
    assert len(set(ids)) == len(ids)


# -- failure accounting -------------------------------------------------------------------


async def test_backend_errors_become_failed_records_not_exceptions() -> None:
    requests = build_requests(make_prompts(4), model="m")
    generate = make_backend(raise_error=BackendOverloadedError("queue full", backend="mock"))
    result = await run_closed_loop(requests, generate, spec=LoadSpec(concurrency=2))
    assert len(result.records) == 4
    assert result.num_failed == 4
    assert all("queue full" in (record.error or "") for record in result.records)


async def test_a_stream_without_a_terminator_is_a_failure() -> None:
    requests = build_requests(make_prompts(1), model="m")
    generate = make_backend(tokens=2, never_finish=True)
    result = await run_closed_loop(requests, generate, spec=LoadSpec())
    record = result.records[0]
    assert not record.ok
    assert record.error == "stream ended without a terminating event"


async def test_a_timeout_aborts_the_stream_and_records_the_failure() -> None:
    closed: list[str] = []
    requests = build_requests(make_prompts(2), model="m")
    generate = make_backend(hang=True, closed=closed)
    spec = LoadSpec(concurrency=2, request_timeout_s=0.05)
    result = await run_closed_loop(requests, generate, spec=spec)
    assert result.num_failed == 2
    assert all("timed out" in (record.error or "") for record in result.records)
    # The obligation from the Backend contract: closing the iterator aborts the work.
    assert sorted(closed) == sorted(req.request_id for req in requests)


async def test_an_in_band_error_event_marks_the_record_failed() -> None:
    async def generate(req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        yield TokenEvent.delta(req.request_id, [1])
        yield TokenEvent.failure(req.request_id, "engine died mid-stream")

    requests = build_requests(make_prompts(1), model="m")
    result = await run_closed_loop(requests, generate, spec=LoadSpec())
    record = result.records[0]
    assert not record.ok
    assert record.error == "engine died mid-stream"
    assert record.output_tokens == 1


# -- run_load orchestration ---------------------------------------------------------------


async def test_run_load_keeps_warmup_records_out_of_the_measured_set() -> None:
    warmup = build_requests(make_prompts(2), model="m", id_prefix="warm-")
    measured = build_requests(make_prompts(6), model="m", id_prefix="run-")
    result = await run_load(
        measured,
        make_backend(tokens=2),
        spec=LoadSpec(concurrency=3),
        warmup=warmup,
    )
    assert len(result.warmup_records) == 2
    assert len(result.records) == 6
    assert all(record.request_id.startswith("run-") for record in result.records)


async def test_run_load_dispatches_to_the_open_driver() -> None:
    requests = build_requests(make_prompts(10), model="m")
    spec = LoadSpec(mode="open", rate_rps=500.0, seed=2)
    result = await run_load(requests, make_backend(tokens=1, delay=0.01), spec=spec)
    assert len(result.records) == 10


async def test_records_can_be_folded_into_a_run_result() -> None:
    from turboserve.bench.records import RunResult

    requests = build_requests(make_prompts(5), model="m")
    result = await run_closed_loop(requests, make_backend(tokens=3), spec=LoadSpec(concurrency=2))
    run = RunResult(scenario="unit", profile="dev", hardware={})
    result.into(run)
    summary = run.summarize()
    assert summary["num_requests"] == 5
    assert summary["num_ok"] == 5
    assert summary["output_tokens"] == 15


# -- configuration ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "sideways"}, "mode must be"),
        ({"concurrency": 0}, "concurrency"),
        ({"mode": "open"}, "rate_rps"),
        ({"duration_s": 0.0}, "duration_s"),
        ({"max_in_flight": 0}, "max_in_flight"),
        ({"request_timeout_s": -1.0}, "request_timeout_s"),
        ({"repeat_requests": True}, "repeat_requests"),
    ],
)
def test_load_spec_rejects_incoherent_settings(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LoadSpec(**kwargs)


def test_load_config_describes_the_spec_for_the_result_file() -> None:
    spec = LoadSpec(mode="open", rate_rps=12.5, concurrency=8, backend_name="reference")
    block = load_config(spec, note="unit")
    assert block["mode"] == "open"
    assert block["rate_rps"] == 12.5
    assert block["concurrency"] == 8
    assert block["backend"] == "reference"
    assert block["note"] == "unit"
