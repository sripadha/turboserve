"""Tests for the asyncio surface: streaming order, cancellation and failure propagation.

Most of these drive a *fake* synchronous engine rather than a real model. The async layer's
job is queueing, ordering and cancellation, and a fake makes every one of those observable:
a step that returns exactly the deltas the test dictates, a step that raises on demand, a
step that blocks until the test lets it finish. One test at the end runs the real engine on
the cached tiny checkpoint to show the two halves fit together.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from turboserve.engine.core.types import (
    EngineConfig,
    FinishReason,
    RequestOutput,
    SamplingParams,
    SchedulerConfig,
)
from turboserve.engine.runtime.async_engine import AsyncEngineDeadError, AsyncLLMEngine
from turboserve.engine.runtime.engine import LLMEngine


class FakeEngine:
    """A synchronous engine stand-in whose steps the test writes by hand.

    Implements only what :class:`AsyncLLMEngine` calls: ``add_request``, ``step``, ``abort``,
    ``has_unfinished``, ``stats`` and ``close``. Each request emits ``tokens_per_request``
    single-token deltas and then a terminator.
    """

    def __init__(
        self,
        *,
        tokens_per_request: int = 3,
        fail_on_step: int | None = None,
        step_delay_s: float = 0.0,
    ) -> None:
        self.tokens_per_request = tokens_per_request
        self.fail_on_step = fail_on_step
        self.step_delay_s = step_delay_s
        self.added: list[str] = []
        self.aborted: list[str] = []
        self.rejected: set[str] = set()
        self.closed = False
        self.num_steps = 0
        self.step_gate: asyncio.Event | None = None
        self._progress: dict[str, int] = {}
        self._order: list[str] = []

    # -- LLMEngine surface ---------------------------------------------------------------

    def add_request(self, request_id: str, prompt: Any, sampling: Any = None, **kw: Any) -> None:
        if request_id in self.rejected:
            raise ValueError(f"request {request_id!r} rejected by the fake engine")
        self.added.append(request_id)
        self._progress[request_id] = 0
        self._order.append(request_id)

    def abort(self, request_id: str, **kw: Any) -> bool:
        self.aborted.append(request_id)
        if request_id in self._progress:
            del self._progress[request_id]
            self._order.remove(request_id)
            return True
        return False

    def has_unfinished(self) -> bool:
        return bool(self._order)

    def step(self) -> list[RequestOutput]:
        if self.step_delay_s:
            time.sleep(self.step_delay_s)  # stands in for a forward pass
        self.num_steps += 1
        if self.fail_on_step is not None and self.num_steps >= self.fail_on_step:
            raise RuntimeError("synthetic engine failure")
        outputs: list[RequestOutput] = []
        for request_id in list(self._order):
            done = self._progress[request_id] + 1
            self._progress[request_id] = done
            finished = done >= self.tokens_per_request
            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    new_token_ids=[100 + done],
                    text_delta=f"t{done}",
                    finished=finished,
                    finish_reason=FinishReason.LENGTH if finished else None,
                    output_tokens=done,
                    prompt_tokens=4,
                )
            )
            if finished:
                del self._progress[request_id]
                self._order.remove(request_id)
        return outputs

    def stats(self) -> dict[str, int | float | str]:
        return {"num_running": len(self._order), "num_steps": self.num_steps}

    def close(self) -> None:
        self.closed = True


def _async_engine(**kwargs: Any) -> tuple[AsyncLLMEngine, FakeEngine]:
    fake = FakeEngine(**kwargs)
    return AsyncLLMEngine(fake, idle_timeout_s=0.01), fake  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------
# streaming
# ----------------------------------------------------------------------------------------


async def test_generate_streams_deltas_in_order() -> None:
    """Deltas arrive in the order the engine produced them, ending with ``finished``."""
    engine, fake = _async_engine(tokens_per_request=4)
    async with engine:
        events = [event async for event in engine.generate("r1", "hello")]
    assert [event.text_delta for event in events] == ["t1", "t2", "t3", "t4"]
    assert [event.new_token_ids for event in events] == [[101], [102], [103], [104]]
    assert events[-1].finished
    assert events[-1].finish_reason is FinishReason.LENGTH
    assert not any(event.finished for event in events[:-1])
    assert fake.added == ["r1"]


async def test_concurrent_streams_do_not_interleave_their_events() -> None:
    """Two streams share one step loop but each sees only its own request's deltas."""
    engine, _ = _async_engine(tokens_per_request=3)

    async def drain(request_id: str) -> list[str]:
        return [event.request_id async for event in engine.generate(request_id, f"p-{request_id}")]

    async with engine:
        first, second = await asyncio.gather(drain("a"), drain("b"))
    assert first == ["a", "a", "a"]
    assert second == ["b", "b", "b"]


async def test_stats_expose_the_async_layer() -> None:
    """``stats()`` merges the engine's counters with queue depths from this layer."""
    engine, _ = _async_engine(tokens_per_request=2)
    async with engine:
        async for _ in engine.generate("r1", "hello"):
            pass
        stats = engine.stats()
    assert stats["loop_running"] == 1
    assert stats["num_streams"] == 0
    assert stats["num_pending"] == 0


async def test_duplicate_request_id_is_refused() -> None:
    """Two live streams with the same id would be indistinguishable, so the second fails."""
    engine, _ = _async_engine(tokens_per_request=8)
    async with engine:
        first = engine.generate("dup", "hello")
        await first.__anext__()
        with pytest.raises(ValueError, match="already streaming"):
            await engine.generate("dup", "hello").__anext__()
        await first.aclose()


async def test_engine_rejection_fails_only_that_stream() -> None:
    """A request the engine refuses raises to its own caller and leaves the loop running."""
    engine, fake = _async_engine(tokens_per_request=2)
    fake.rejected.add("bad")
    async with engine:
        with pytest.raises(ValueError, match="rejected by the fake engine"):
            async for _ in engine.generate("bad", "hello"):
                pass
        good = [event.text_delta async for event in engine.generate("good", "hello")]
    assert good == ["t1", "t2"]


# ----------------------------------------------------------------------------------------
# cancellation
# ----------------------------------------------------------------------------------------


async def test_closing_the_stream_early_aborts_the_request() -> None:
    """Closing the iterator early aborts the request so its KV blocks come back.

    ``aclose()`` is explicit here rather than relying on a ``break``: Python finalises an
    abandoned async generator through the loop's asyncgen hooks at an unspecified later
    point, so a server that wants the blocks back promptly must close the stream itself.
    That is what :class:`~turboserve.gateway.backends.local_engine.LocalEngineBackend` does.
    """
    engine, fake = _async_engine(tokens_per_request=50, step_delay_s=0.01)
    async with engine:
        stream = engine.generate("r1", "hello")
        first = await stream.__anext__()
        assert not first.finished
        await stream.aclose()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if "r1" in fake.aborted:
                break
    assert "r1" in fake.aborted


async def test_cancelling_the_consumer_task_aborts_the_request() -> None:
    """An ``asyncio.CancelledError`` in the consumer reaches the engine as an abort.

    This is the client-disconnect path: the HTTP server cancels the handler task, the
    generator's ``finally`` runs, and the request must not keep its blocks.
    """
    engine, fake = _async_engine(tokens_per_request=100, step_delay_s=0.01)

    async def consume() -> None:
        async for _ in engine.generate("r1", "hello"):
            pass

    async with engine:
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(20):
            await asyncio.sleep(0.01)
            if "r1" in fake.aborted:
                break
    assert "r1" in fake.aborted


async def test_abort_delivers_a_terminating_event_to_a_waiting_consumer() -> None:
    """An abort requested from elsewhere ends the stream with an ``ABORT`` terminator."""
    engine, _ = _async_engine(tokens_per_request=1000, step_delay_s=0.005)
    async with engine:
        stream = engine.generate("r1", "hello")
        first = await stream.__anext__()
        assert not first.finished
        await engine.abort("r1")
        last = first
        for _ in range(20):
            last = await stream.__anext__()
            if last.finished:
                break
        assert last.finished
        assert last.finish_reason is FinishReason.ABORT
        await stream.aclose()


# ----------------------------------------------------------------------------------------
# failure propagation
# ----------------------------------------------------------------------------------------


async def test_a_dying_step_loop_fails_every_open_stream() -> None:
    """When a step raises, every waiting consumer learns about it instead of hanging."""
    engine, _ = _async_engine(tokens_per_request=100, fail_on_step=2)
    try:
        with pytest.raises(AsyncEngineDeadError, match="step loop died"):
            async for _ in engine.generate("r1", "hello"):
                pass
        with pytest.raises(AsyncEngineDeadError):
            async for _ in engine.generate("r2", "hello"):
                pass
    finally:
        await engine.close()


async def test_close_is_idempotent_and_closes_the_engine() -> None:
    """``close()`` stops the loop, releases the engine, and can be called twice."""
    engine, fake = _async_engine(tokens_per_request=2)
    await engine.start()
    await engine.close()
    await engine.close()
    assert fake.closed
    assert engine.is_running is False
    assert engine.is_healthy is False
    with pytest.raises(AsyncEngineDeadError, match="has been closed"):
        await engine.start()


# ----------------------------------------------------------------------------------------
# with the real engine
# ----------------------------------------------------------------------------------------


async def test_real_engine_streams_the_same_tokens_it_would_batch(
    tiny_qwen2_path: Path,
) -> None:
    """The async wrapper does not change what the engine produces, only how it arrives."""
    config = EngineConfig(
        model=str(tiny_qwen2_path),
        device="cpu",
        dtype="float32",
        scheduler=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=128, block_size=16, num_blocks=32
        ),
    )
    sampling = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    with LLMEngine(config, local_files_only=True) as reference:
        expected = reference.generate(["Hello world"], sampling)[0].new_token_ids

    async with AsyncLLMEngine(
        LLMEngine(config, local_files_only=True), idle_timeout_s=0.01
    ) as engine:
        streamed: list[int] = []
        async for event in engine.generate("r1", "Hello world", sampling):
            streamed.extend(event.new_token_ids)
            if event.finished:
                assert event.usage()["completion_tokens"] == len(streamed)
    assert streamed == expected
