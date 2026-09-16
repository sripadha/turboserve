"""The contract every gateway backend implements, and the errors it may raise.

The gateway routes a request to *a* backend without knowing what is behind it: an
in-process :class:`~turboserve.engine.runtime.engine.AsyncLLMEngine`, a vLLM server over
HTTP, or a mock used by the chaos harness and the kind end-to-end test. Keeping that
surface to four methods is what makes those three interchangeable in routing, canary
weighting and fault injection.

Two details of this contract carry weight:

* **Streaming is the only mode.** :meth:`Backend.generate` always returns an async
  iterator of :class:`TokenEvent`, even for a non-streaming client request, which the
  route then collects. A backend that buffered internally would destroy time-to-first-token
  and inter-token-latency -- the two numbers the whole gateway is measured on.
* **Events carry ``t_ns``, stamped by the producer.** Each event records
  ``time.monotonic_ns()`` at the moment the backend had the token, not the moment the
  client deserialised it. Nanosecond integers, because inter-token gaps under load are
  routinely below a microsecond of resolution once batched, and because integers survive
  JSON round-trips without the rounding a float second would suffer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from turboserve.engine.core.types import FinishReason, SamplingParams

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator

__all__ = [
    "AdapterNotFoundError",
    "Backend",
    "BackendError",
    "BackendOverloadedError",
    "BackendRequestError",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "GenerateRequest",
    "ModelNotFoundError",
    "NonRetryableBackendError",
    "RetryableBackendError",
    "StreamInterruptedError",
    "TokenEvent",
]


class GenerateRequest(BaseModel):
    """One generation request, normalised out of whatever OpenAI-shaped body arrived.

    The prompt is either text or token ids: the chat route applies the tokenizer's chat
    template and passes ids, while the completions route may pass text straight through.
    Backends must handle both, because an HTTP backend cannot send ids to every server and
    an in-process engine should not re-tokenise text it already has ids for.

    ``tenant_id`` is resolved by :mod:`turboserve.gateway.auth` before this object exists,
    so every backend, metric and log line downstream is attributable to a tenant.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt: str | list[int]
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    lora: str | None = None
    """Adapter *name* as the tenant knows it; the backend maps it to a slot or a model id."""

    priority: int = 0
    """Higher runs first where the backend supports it; equal priorities are FCFS."""

    stream: bool = True
    arrival_ts: float = Field(default_factory=time.perf_counter)
    """``time.perf_counter()`` seconds at admission, the ``t_arrival`` of TTFT."""

    @field_validator("prompt")
    @classmethod
    def _prompt_not_empty(cls, value: str | list[int]) -> str | list[int]:
        if len(value) == 0:
            raise ValueError("prompt must not be empty")
        return value

    @property
    def prompt_is_tokens(self) -> bool:
        """Whether the prompt is already tokenised."""
        return isinstance(self.prompt, list)

    @property
    def prompt_token_ids(self) -> list[int] | None:
        """The prompt's token ids, or ``None`` if it is text."""
        return self.prompt if isinstance(self.prompt, list) else None

    @property
    def prompt_text(self) -> str | None:
        """The prompt's text, or ``None`` if it is already tokenised."""
        return self.prompt if isinstance(self.prompt, str) else None


@dataclass(slots=True)
class TokenEvent:
    """One chunk of a streaming response, or its terminator, or its failure.

    Exactly one of three shapes:

    * a *delta*: ``token_ids`` and/or ``text`` non-empty, ``finished`` false;
    * a *terminator*: ``finished`` true, ``finish_reason`` set, ``usage`` filled in;
    * a *failure*: ``error`` set and ``finished`` true.

    A terminator may also carry the last delta, so a backend never has to emit an empty
    event just to say it is done.
    """

    request_id: str
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    t_ns: int = field(default_factory=time.monotonic_ns)
    """``time.monotonic_ns()`` when the producing backend had this chunk."""

    finished: bool = False
    finish_reason: FinishReason | None = None
    usage: dict[str, Any] | None = None
    """Token accounting, present on the terminator; see ``RequestOutput.usage()``."""

    error: str | None = None

    @property
    def is_error(self) -> bool:
        """Whether this event reports a failure rather than progress."""
        return self.error is not None

    @property
    def num_tokens(self) -> int:
        """Tokens carried by this event."""
        return len(self.token_ids)

    @classmethod
    def delta(
        cls,
        request_id: str,
        token_ids: list[int],
        text: str = "",
        *,
        t_ns: int | None = None,
    ) -> TokenEvent:
        """Build a progress event."""
        return cls(
            request_id=request_id,
            token_ids=token_ids,
            text=text,
            t_ns=time.monotonic_ns() if t_ns is None else t_ns,
        )

    @classmethod
    def final(
        cls,
        request_id: str,
        finish_reason: FinishReason,
        *,
        token_ids: list[int] | None = None,
        text: str = "",
        usage: dict[str, Any] | None = None,
        t_ns: int | None = None,
    ) -> TokenEvent:
        """Build a terminator, optionally carrying the final delta."""
        return cls(
            request_id=request_id,
            token_ids=list(token_ids or []),
            text=text,
            t_ns=time.monotonic_ns() if t_ns is None else t_ns,
            finished=True,
            finish_reason=finish_reason,
            usage=usage,
        )

    @classmethod
    def failure(
        cls,
        request_id: str,
        error: str,
        *,
        finish_reason: FinishReason | None = None,
        t_ns: int | None = None,
    ) -> TokenEvent:
        """Build a failure terminator.

        Used where raising is impossible -- once the response body has started, an
        exception cannot become an HTTP status code, so the failure travels in-band and
        the gateway records it as an errored request.
        """
        return cls(
            request_id=request_id,
            t_ns=time.monotonic_ns() if t_ns is None else t_ns,
            finished=True,
            finish_reason=finish_reason,
            error=error,
        )


@runtime_checkable
class Backend(Protocol):
    """A source of token streams that the router can treat like any other.

    ``generate`` is declared as a plain method returning an ``AsyncIterator`` rather than
    as ``async def``: implementations are async generator functions (``async def`` with
    ``yield``), which return the iterator directly and are *not* awaitable. Callers write
    ``async for event in backend.generate(req):``. The other three methods are coroutines.
    """

    name: str
    """Stable identifier used in metric labels, routing tables and log lines."""

    supports_lora: bool
    """Whether :attr:`GenerateRequest.lora` is honoured; the router rejects it otherwise."""

    def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Stream the response to ``req``.

        Must yield at least one event, and the last event it yields must have
        ``finished=True`` (a terminator or a failure). Cancelling the iterator (the client
        disconnected) must abort the work behind it rather than leave it running.
        """
        ...

    async def health(self) -> bool:
        """Whether this backend is ready to take traffic right now.

        Must not raise: the router polls it on a timer and treats an exception the same as
        ``False``, but a backend that raises makes the health loop noisy for no gain.
        """
        ...

    async def models(self) -> list[str]:
        """Model names this backend serves, for ``GET /v1/models`` and routing checks."""
        ...

    async def close(self) -> None:
        """Release connections, worker tasks and device memory. Idempotent."""
        ...


class BackendError(Exception):
    """Base class for backend failures, carrying what the gateway needs to respond.

    ``retryable`` is the single question the router asks: may this request be sent to
    another replica? It is a class-level property of the error type rather than a
    case-by-case judgement at the raise site, so the rule stays auditable.
    """

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.backend = backend
        self.status_code = status_code

    def __str__(self) -> str:
        where = f" [backend={self.backend}]" if self.backend else ""
        code = f" (status {self.status_code})" if self.status_code is not None else ""
        return f"{self.message}{where}{code}"


class RetryableBackendError(BackendError):
    """The request never reached a model, so another replica may serve it.

    Retrying is only safe before the first byte of the response has been written; see
    :class:`StreamInterruptedError` for what happens after.
    """

    retryable = True


class NonRetryableBackendError(BackendError):
    """Retrying would fail identically, or would duplicate work already streamed."""

    retryable = False


class BackendUnavailableError(RetryableBackendError):
    """The backend could not be reached at all (connection refused, DNS, not ready)."""


class BackendTimeoutError(RetryableBackendError):
    """The backend accepted the connection but produced nothing within the deadline."""


class BackendOverloadedError(RetryableBackendError):
    """The backend refused the request because its queue is full (HTTP 429/503)."""


class BackendRequestError(NonRetryableBackendError):
    """The request itself is invalid for this backend (bad sampling params, too long)."""


class ModelNotFoundError(NonRetryableBackendError):
    """The requested model is not served by this backend."""


class AdapterNotFoundError(NonRetryableBackendError):
    """The requested LoRA adapter is unknown to this backend."""


class StreamInterruptedError(NonRetryableBackendError):
    """The stream died after tokens had already been sent to the client.

    Non-retryable by construction: the client has part of a completion, and restarting on
    another replica would either duplicate or contradict it. The gateway ends the stream
    and counts the request as failed.
    """
