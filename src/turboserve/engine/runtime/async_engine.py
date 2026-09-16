"""The asyncio surface: one background step loop feeding one queue per request.

A serving process has two very different kinds of work. Accepting HTTP requests, applying
chat templates and writing SSE frames is I/O bound and belongs on an event loop; a scheduler
step is a dense GPU call that holds the GIL for as long as it runs. Running the step on the
event loop would stall every open stream for the duration of every forward pass, which is
exactly the tail-latency pathology a serving engine exists to avoid.

So :class:`AsyncLLMEngine` runs the loop like this:

* one background ``asyncio.Task`` owns the engine and is the *only* thing that touches it;
* each :meth:`AsyncLLMEngine.generate` call registers an ``asyncio.Queue`` and drops a
  request description into a thread-safe ``deque``, which the loop drains before its next
  step -- so no lock is ever taken and no caller can observe the engine mid-step;
* the step itself runs in ``asyncio.to_thread``, releasing the event loop for the whole
  forward pass. torch drops the GIL inside its kernels, so the loop really does make
  progress rather than merely appearing to.

Cancellation is the other half of the contract. Closing the async generator -- a client
disconnecting, ``asyncio.timeout`` firing, a ``break`` in the consumer -- aborts the request
in the engine, which frees its KV blocks on the next step. A dropped connection that kept
its blocks would be an outage waiting to happen.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from turboserve.engine.core.sequence import DEFAULT_TENANT
from turboserve.engine.core.types import NO_LORA, EngineConfig, FinishReason, RequestOutput
from turboserve.engine.runtime.engine import LLMEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncGenerator
    from collections.abc import Sequence as SequenceABC

    from turboserve.config import Settings
    from turboserve.engine.core.types import SamplingParams

logger = logging.getLogger(__name__)

__all__ = ["AsyncEngineDeadError", "AsyncLLMEngine"]

#: How long the loop waits for new work before checking again. Only reached when the engine
#: is completely idle: any arrival sets an event that wakes it immediately, so this is a
#: liveness backstop, not a polling interval that costs latency.
_IDLE_TIMEOUT_S = 0.5


class AsyncEngineDeadError(RuntimeError):
    """The background loop stopped because the engine raised.

    Raised to every waiting stream and to every later :meth:`AsyncLLMEngine.generate`. The
    engine is not restarted automatically: a step that raised has left the KV pool in a
    state nobody has reasoned about, and serving on top of that would turn one bad request
    into silently wrong tokens for everybody.
    """


@dataclass(slots=True)
class _PendingRequest:
    """A request accepted by the API but not yet handed to the engine."""

    request_id: str
    prompt: str | SequenceABC[int]
    sampling: SamplingParams | None
    tenant_id: str
    priority: int
    lora_id: int
    arrival: float
    queue: asyncio.Queue[RequestOutput | BaseException]


@dataclass(slots=True)
class _Stream:
    """The consumer side of one in-flight request."""

    queue: asyncio.Queue[RequestOutput | BaseException] = field(default_factory=asyncio.Queue)
    finished: bool = False


class AsyncLLMEngine:
    """Asynchronous wrapper around :class:`~turboserve.engine.runtime.engine.LLMEngine`.

    Start it with :meth:`start` (or ``async with``), stream with :meth:`generate`, stop it
    with :meth:`close`. The wrapped engine is available as :attr:`engine` for stats and for
    tests, but must not be stepped from outside while the loop is running.
    """

    def __init__(self, engine: LLMEngine, *, idle_timeout_s: float = _IDLE_TIMEOUT_S) -> None:
        self.engine = engine
        self._idle_timeout_s = idle_timeout_s
        self._pending: deque[_PendingRequest] = deque()
        self._aborting: deque[str] = deque()
        self._streams: dict[str, _Stream] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._closed = False

    # -- construction --------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: EngineConfig, **kwargs: Any) -> AsyncLLMEngine:
        """Build the underlying engine from a config and wrap it."""
        idle_timeout_s = kwargs.pop("idle_timeout_s", _IDLE_TIMEOUT_S)
        return cls(LLMEngine(config, **kwargs), idle_timeout_s=idle_timeout_s)

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> AsyncLLMEngine:
        """Build from the process-wide ``TURBOSERVE_*`` settings."""
        return cls.from_config(EngineConfig.from_settings(settings), **kwargs)

    # -- lifecycle -----------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the background loop is alive."""
        return self._task is not None and not self._task.done()

    @property
    def is_healthy(self) -> bool:
        """Whether the engine can still accept work.

        ``False`` once the loop has died or the engine has been closed. A *stopped* engine
        that has never been started is healthy: it starts on the first request.
        """
        return self._error is None and not self._closed

    async def start(self) -> None:
        """Start the background step loop. Idempotent."""
        if self._closed:
            raise AsyncEngineDeadError("engine has been closed")
        if self.is_running:
            return
        self._error = None
        self._task = asyncio.create_task(self._run(), name="turboserve-engine-loop")

    async def close(self) -> None:
        """Stop the loop, fail every open stream and release the engine. Idempotent."""
        if self._closed:
            return
        self._closed = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._fail_all(AsyncEngineDeadError("engine was closed"))
        self.engine.close()

    async def __aenter__(self) -> AsyncLLMEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- the loop ------------------------------------------------------------------------

    async def _run(self) -> None:
        """Drain arrivals, step, fan out -- forever, until cancelled or the engine raises."""
        try:
            while True:
                self._drain_pending()
                self._drain_aborts()
                if not self.engine.has_unfinished():
                    await self._wait_for_work()
                    continue
                outputs = await asyncio.to_thread(self.engine.step)
                self._dispatch(outputs)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # re-raised below, after every stream is told
            logger.exception("engine step loop died")
            self._error = exc
            self._fail_all(AsyncEngineDeadError(f"engine step loop died: {exc}"))
            raise

    async def _wait_for_work(self) -> None:
        """Sleep until something arrives, or until the backstop timeout expires."""
        self._wake.clear()
        if self._pending or self._aborting:
            return
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=self._idle_timeout_s)
        except TimeoutError:
            return

    def _drain_pending(self) -> None:
        """Hand every queued arrival to the engine.

        A request that the engine rejects (duplicate id, prompt larger than the whole KV
        pool) fails only its own stream: one bad request must not take the loop down.
        """
        while self._pending:
            item = self._pending.popleft()
            try:
                self.engine.add_request(
                    item.request_id,
                    item.prompt,
                    item.sampling,
                    tenant_id=item.tenant_id,
                    priority=item.priority,
                    lora_id=item.lora_id,
                    arrival=item.arrival,
                )
            except (ValueError, RuntimeError, TypeError) as exc:
                logger.warning("rejected request %s: %s", item.request_id, exc)
                self._streams.pop(item.request_id, None)
                item.queue.put_nowait(exc)

    def _drain_aborts(self) -> None:
        """Abort every request whose consumer went away, and close its stream."""
        while self._aborting:
            request_id = self._aborting.popleft()
            if self.engine.abort(request_id):
                logger.debug("aborted %s", request_id)
            stream = self._streams.pop(request_id, None)
            if stream is not None and not stream.finished:
                stream.queue.put_nowait(
                    RequestOutput(
                        request_id=request_id,
                        finished=True,
                        finish_reason=FinishReason.ABORT,
                    )
                )

    def _dispatch(self, outputs: SequenceABC[RequestOutput]) -> None:
        """Put each delta on its request's queue, in the order the engine produced it."""
        for output in outputs:
            stream = self._streams.get(output.request_id)
            if stream is None:
                continue  # consumer already gone; the abort is queued or already applied
            stream.queue.put_nowait(output)
            if output.finished:
                stream.finished = True
                self._streams.pop(output.request_id, None)

    def _fail_all(self, error: BaseException) -> None:
        """Push ``error`` into every open stream and forget them."""
        for stream in list(self._streams.values()):
            if not stream.finished:
                stream.queue.put_nowait(error)
        self._streams.clear()

    # -- public API ----------------------------------------------------------------------

    async def generate(
        self,
        request_id: str,
        prompt: str | SequenceABC[int],
        sampling: SamplingParams | None = None,
        *,
        tenant_id: str = DEFAULT_TENANT,
        priority: int = 0,
        lora_id: int = NO_LORA,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Stream one request's output deltas, in order, ending with ``finished=True``.

        Closing the iterator early aborts the request. The final event is always delivered:
        either the engine's own terminator, or an ``ABORT`` terminator synthesised when the
        consumer disappeared first.
        """
        if self._closed:
            raise AsyncEngineDeadError("engine has been closed")
        if self._error is not None:
            raise AsyncEngineDeadError(f"engine step loop died: {self._error}")
        await self.start()
        if request_id in self._streams:
            raise ValueError(f"request {request_id!r} is already streaming")
        stream = _Stream()
        self._streams[request_id] = stream
        self._pending.append(
            _PendingRequest(
                request_id=request_id,
                prompt=prompt,
                sampling=sampling,
                tenant_id=tenant_id,
                priority=priority,
                lora_id=lora_id,
                arrival=time.perf_counter(),
                queue=stream.queue,
            )
        )
        self._wake.set()
        finished = False
        try:
            while True:
                item = await stream.queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    finished = True
                    return
        finally:
            if not finished:
                self.abort_nowait(request_id)

    def abort_nowait(self, request_id: str) -> None:
        """Queue an abort without waiting for the loop to apply it.

        Safe from a ``finally`` block during cancellation, where awaiting is not allowed.
        """
        self._aborting.append(request_id)
        self._wake.set()

    async def abort(self, request_id: str) -> None:
        """Queue an abort and yield control so the loop can pick it up promptly."""
        self.abort_nowait(request_id)
        await asyncio.sleep(0)

    def stats(self) -> dict[str, int | float | str]:
        """The wrapped engine's statistics, plus the async layer's own queue depths."""
        out = self.engine.stats()
        out["num_streams"] = len(self._streams)
        out["num_pending"] = len(self._pending)
        out["loop_running"] = int(self.is_running)
        return out

    def __repr__(self) -> str:
        state = "running" if self.is_running else ("closed" if self._closed else "stopped")
        return f"AsyncLLMEngine({state}, streams={len(self._streams)}, engine={self.engine!r})"
