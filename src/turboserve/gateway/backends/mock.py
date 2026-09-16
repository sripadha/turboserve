"""A backend that behaves like a model server without needing a model.

Three consumers need to exercise the gateway's *serving* behaviour -- routing, retries,
quotas, canary weighting, chaos, the Kubernetes end-to-end test -- without a GPU, a
checkpoint or a multi-second warm-up. Giving them a real engine would make those tests slow,
flaky and dependent on hardware that CI does not have; giving them a stub that returns
instantly would make them meaningless, because every interesting behaviour in a gateway is a
*timing* behaviour.

So this backend is configurable along exactly the axes that matter for those tests:

* ``ttft_ms`` and ``itl_ms`` -- the two latencies every serving metric is built from;
* ``error_probability`` -- failure *before* the first token, the only kind a router may
  retry;
* ``drop_probability`` -- failure *after* tokens were sent, which must never be retried;
* ``healthy`` -- what the health probe says, flippable at runtime so a chaos schedule can
  take a replica out and put it back;
* ``adapters`` and ``models`` -- so adapter and model routing can be refused realistically.

Everything is driven by a seeded RNG derived from the request id, so a given request always
gets the same latencies, the same text and the same fate. That is what lets a chaos test
assert an error rate instead of hoping for one.

The same object is also the engine behind :func:`serve_mock`, which puts an
OpenAI-compatible HTTP server in front of it. The chaos harness launches several of those as
subprocesses to get real sockets, real connection failures and real process kills.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field

from turboserve.engine.core.types import FinishReason
from turboserve.gateway.backends import register_backend
from turboserve.gateway.backends.protocol import (
    AdapterNotFoundError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
    TokenEvent,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

__all__ = [
    "MOCK_VOCABULARY",
    "MockBackend",
    "MockConfig",
    "MockStats",
    "build_mock_app",
    "serve_mock",
]

#: Words the mock emits. Deliberately about the system under test rather than lorem ipsum,
#: so a captured transcript is recognisably synthetic and nobody mistakes it for a real
#: model's output.
MOCK_VOCABULARY: Final[tuple[str, ...]] = (
    "synthetic",
    "token",
    "from",
    "the",
    "turboserve",
    "mock",
    "backend",
    "stream",
    "batch",
    "prefix",
    "cache",
    "adapter",
    "lane",
    "tenant",
    "gateway",
    "replica",
)

#: Token ids the mock reports. Offset well clear of the small ids real tokenizers use for
#: control tokens, so a mock id showing up where a real one belongs is obvious.
_TOKEN_ID_BASE: Final = 100_000

#: Characters per token assumed when counting a *text* prompt. The mock has no tokenizer;
#: this is an approximation and the usage it reports is approximate with it.
_CHARS_PER_TOKEN: Final = 4


class MockConfig(BaseModel):
    """Everything the mock backend can be told to do.

    Mutable on purpose (``validate_assignment`` keeps it honest): the chaos harness applies a
    fault schedule by assigning to ``error_probability``, ``itl_ms`` or ``healthy`` on a live
    backend, which is how an in-process fault injection works without restarting anything.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = "mock"
    models: list[str] = Field(default_factory=lambda: ["mock-model"])
    adapters: list[str] = Field(default_factory=list)
    max_tokens: int = Field(default=32, ge=1)
    """Upper bound on the completion length, regardless of the request's ``max_tokens``."""

    ttft_ms: float = Field(default=0.0, ge=0.0)
    itl_ms: float = Field(default=0.0, ge=0.0)
    jitter: float = Field(default=0.0, ge=0.0, le=1.0)
    """Relative spread applied to both latencies, uniform in ``[-jitter, +jitter]``."""

    error_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    """Chance the request fails before producing anything (retryable by the router)."""

    drop_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    """Chance the stream is cut after some tokens (never retryable: bytes are already out)."""

    healthy: bool = True
    tokens_per_event: int = Field(default=1, ge=1)
    """Tokens per streamed event; >1 imitates a server that batches chunks on the wire."""

    seed: int = 1234
    accept_any_model: bool = False
    """When true, do not refuse unknown model names -- useful for a drop-in stand-in."""

    vocabulary: list[str] = Field(default_factory=lambda: list(MOCK_VOCABULARY))


class MockStats(BaseModel):
    """Counters a test or a chaos run reads back off the backend."""

    model_config = ConfigDict(extra="forbid")

    requests: int = 0
    completed: int = 0
    failed: int = 0
    dropped: int = 0
    cancelled: int = 0
    tokens: int = 0
    in_flight: int = 0


@register_backend("mock")
class MockBackend:
    """A :class:`~turboserve.gateway.backends.protocol.Backend` with no model behind it."""

    supports_lora = True

    def __init__(self, config: MockConfig | None = None, **overrides: Any) -> None:
        self.config = config.model_copy(deep=True) if config is not None else MockConfig()
        for key, value in overrides.items():
            setattr(self.config, key, value)
        self.name = self.config.name
        """Fixed at construction: the name is a metric label and a routing key, and a label
        that changed under a running query would silently split a time series in two."""

        self.stats = MockStats()
        self._closed = False

    def set_healthy(self, healthy: bool) -> None:
        """Flip what the health probe reports; the chaos harness uses this to fail a replica."""
        self.config.healthy = healthy

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    # -- generation ---------------------------------------------------------------------

    def _rng(self, request_id: str) -> random.Random:
        """Deterministic RNG for one request.

        Seeding from a *string* is what makes this reproducible across processes: Python
        derives the state from a hash of the bytes, not from ``hash()``, so it does not move
        with ``PYTHONHASHSEED``. Two workers given the same request id therefore behave
        identically, which is what a chaos run needs to be replayable.
        """
        return random.Random(f"{self.config.seed}:{request_id}")

    def _delay(self, base_ms: float, rng: random.Random) -> float:
        """Seconds to sleep for a configured millisecond latency, with optional jitter."""
        if base_ms <= 0.0:
            return 0.0
        jitter = self.config.jitter
        factor = 1.0 + rng.uniform(-jitter, jitter) if jitter else 1.0
        return max(0.0, base_ms * factor) / 1000.0

    def _prompt_tokens(self, req: GenerateRequest) -> int:
        """Prompt length in tokens; approximate for text, because there is no tokenizer."""
        ids = req.prompt_token_ids
        if ids is not None:
            return len(ids)
        text = req.prompt_text or ""
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def _check_request(self, req: GenerateRequest) -> None:
        """Refuse the request the way a real server would, before any work."""
        if self._closed:
            raise BackendUnavailableError("backend is closed", backend=self.name)
        if not self.config.accept_any_model and req.model not in self.config.models:
            raise ModelNotFoundError(
                f"model {req.model!r} is not served by {self.name!r}", backend=self.name
            )
        if req.lora is not None and req.lora not in self.config.adapters:
            raise AdapterNotFoundError(
                f"adapter {req.lora!r} is not loaded on {self.name!r}", backend=self.name
            )

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Stream a synthetic completion, obeying the configured latencies and faults."""
        self._check_request(req)
        rng = self._rng(req.request_id)
        num_tokens = max(1, min(req.sampling.max_tokens, self.config.max_tokens))
        fails_early = rng.random() < self.config.error_probability
        drops = (not fails_early) and rng.random() < self.config.drop_probability
        # Cut point drawn up front so the drop rate equals drop_probability exactly instead
        # of compounding over the length of the completion. It lands strictly inside the
        # completion, so a dropped stream always has sent at least one token and never all of
        # them; a single-token completion has no inside and is therefore never dropped.
        cut_at = rng.randrange(1, num_tokens) if drops and num_tokens > 1 else num_tokens + 1

        self.stats.requests += 1
        self.stats.in_flight += 1
        prompt_tokens = self._prompt_tokens(req)
        emitted = 0
        try:
            await asyncio.sleep(self._delay(self.config.ttft_ms, rng))
            if fails_early:
                self.stats.failed += 1
                raise BackendUnavailableError(
                    f"{self.name}: synthetic pre-first-token failure", backend=self.name
                )

            pending_ids: list[int] = []
            pending_text: list[str] = []
            while emitted < num_tokens:
                if emitted and self.config.itl_ms:
                    await asyncio.sleep(self._delay(self.config.itl_ms, rng))
                if emitted >= cut_at:
                    self.stats.dropped += 1
                    self.stats.failed += 1
                    yield TokenEvent.failure(
                        req.request_id,
                        f"{self.name}: synthetic mid-stream drop",
                        finish_reason=FinishReason.ABORT,
                    )
                    return
                index = rng.randrange(len(self.config.vocabulary))
                pending_ids.append(_TOKEN_ID_BASE + index)
                pending_text.append(
                    self.config.vocabulary[index]
                    if emitted == 0
                    else f" {self.config.vocabulary[index]}"
                )
                emitted += 1
                self.stats.tokens += 1
                is_last = emitted >= num_tokens
                if len(pending_ids) >= self.config.tokens_per_event and not is_last:
                    yield TokenEvent.delta(req.request_id, list(pending_ids), "".join(pending_text))
                    pending_ids.clear()
                    pending_text.clear()

            self.stats.completed += 1
            yield TokenEvent.final(
                req.request_id,
                FinishReason.LENGTH,
                token_ids=list(pending_ids),
                text="".join(pending_text),
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": emitted,
                    "total_tokens": prompt_tokens + emitted,
                    "cached_prompt_tokens": 0,
                },
            )
        except asyncio.CancelledError:
            self.stats.cancelled += 1
            raise
        finally:
            self.stats.in_flight -= 1

    # -- protocol remainder -------------------------------------------------------------

    async def health(self) -> bool:
        """Configured health. Never raises, as the protocol requires."""
        return self.config.healthy and not self._closed

    async def models(self) -> list[str]:
        """Model names this mock claims to serve."""
        return list(self.config.models)

    async def close(self) -> None:
        """Mark the backend closed; idempotent. Later requests fail as unavailable."""
        self._closed = True

    def __repr__(self) -> str:
        return (
            f"MockBackend(name={self.name!r}, models={self.config.models}, "
            f"ttft_ms={self.config.ttft_ms:g}, itl_ms={self.config.itl_ms:g})"
        )


def build_mock_app(config: MockConfig | None = None, **overrides: Any) -> Any:
    """Build a standalone OpenAI-compatible FastAPI app served by one :class:`MockBackend`.

    It is the real gateway app with authentication off and a single-backend router, not a
    second implementation: a fault injected here therefore travels the same auth, routing,
    accounting and SSE code a production request would, which is the only way the chaos
    results mean anything.

    The gateway app is imported inside the function to keep this module importable (and the
    backend registry populated) without dragging FastAPI in.
    """
    from turboserve.config import Settings
    from turboserve.gateway.app import GatewayOptions, create_app
    from turboserve.gateway.router import Router
    from turboserve.gateway.tenants import TenantRegistry

    backend = MockBackend(config, **overrides)
    router = Router()
    for model in backend.config.models:
        router.add_backend(model, backend)
    app = create_app(
        Settings(),
        router=router,
        tenants=TenantRegistry.default(),
        options=GatewayOptions(require_auth=False),
    )
    app.state.mock_backend = backend
    return app


def serve_mock(
    port: int,
    *,
    host: str = "127.0.0.1",
    log_level: str = "warning",
    **cfg: Any,
) -> None:
    """Run :func:`build_mock_app` under uvicorn until the process is stopped.

    This is the entry point the chaos harness launches as a subprocess (directly, or via
    ``python -m turboserve.gateway.backends.mock``), so that killing a "replica" is a real
    process kill and the gateway sees a real connection failure.
    """
    import uvicorn

    app = build_mock_app(**cfg)
    logger.info("serving mock backend on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def _main(argv: list[str] | None = None) -> None:
    """``python -m turboserve.gateway.backends.mock`` -- argparse, not typer, to stay cheap."""
    import argparse

    from turboserve.logging_utils import configure_logging

    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible mock model server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--name", default="mock")
    parser.add_argument("--model", dest="models", action="append", default=None)
    parser.add_argument("--adapter", dest="adapters", action="append", default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ttft-ms", type=float, default=0.0)
    parser.add_argument("--itl-ms", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--error-probability", type=float, default=0.0)
    parser.add_argument("--drop-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--accept-any-model", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    serve_mock(
        args.port,
        host=args.host,
        log_level=args.log_level.lower(),
        name=args.name,
        models=args.models or ["mock-model"],
        adapters=args.adapters or [],
        max_tokens=args.max_tokens,
        ttft_ms=args.ttft_ms,
        itl_ms=args.itl_ms,
        jitter=args.jitter,
        error_probability=args.error_probability,
        drop_probability=args.drop_probability,
        seed=args.seed,
        accept_any_model=args.accept_any_model,
    )


if __name__ == "__main__":  # pragma: no cover - module entry point
    _main()
