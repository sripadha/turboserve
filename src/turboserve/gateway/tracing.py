"""OpenTelemetry tracing for the gateway: one span per generation, and nothing else.

A serving stack is a distributed system whose interesting failures are *between* the
components: a request that was fast at the gateway and slow at the engine, a retry that
crossed replicas, a tenant whose tail latency comes from one backend and nobody noticed.
Metrics (:mod:`turboserve.gateway.metrics`) aggregate those away by construction -- that is
what makes them cheap -- so this module adds the other half: a trace that follows one
request from the client's connection, through auth, quotas and routing, into whichever
engine answered it.

Three decisions shape everything below.

**Off unless asked for.** ``TURBOSERVE_OTEL_ENDPOINT`` is the switch. With it unset,
:func:`configure_tracing` returns a disabled :class:`Tracing`, imports nothing from
``opentelemetry`` and hands the request path a null span whose methods do nothing -- so the
feature costs an attribute lookup per request, not an exporter, a queue and four packages.
The packages themselves are an optional extra (``uv sync --extra otel``); with the endpoint
set but the extra missing, the gateway logs one warning and serves exactly as before rather
than refusing to start, because tracing is observability and observability must never be the
reason a fleet cannot boot.

**No global state.** Nothing here calls ``trace.set_tracer_provider``. The provider lives on
the app's :class:`~turboserve.gateway.app.GatewayState` and is handed to the FastAPI
instrumentation explicitly, so two gateways in one process (the real one and the in-process
mock upstream the chaos harness builds) get two providers instead of fighting over one
global that only the first caller may set. Only *propagation* uses the library's global, and
that is a stateless codec over the current context rather than configuration.

**Ids, never content.** A span carries the tenant id, the model name, the adapter name, the
backend and lane that served it and the token counts -- the things an operator needs to find
the request again. It never carries prompt text, completion text, messages, API keys or
anything derived from them: a trace backend is usually a different trust domain from the
gateway, frequently retains data for months, and a prompt is the one thing in this system
that is unambiguously the tenant's.

The span's shape::

    turboserve.generate                       (kind INTERNAL, child of the FastAPI span)
      turboserve.tenant       = "acme"
      turboserve.model        = "Qwen/Qwen2.5-7B-Instruct"
      turboserve.backend      = "vllm-a"      set once the router has chosen
      turboserve.lane         = "stable"
      turboserve.lora         = "acme-support" (absent when no adapter was resolved)
      turboserve.prompt_tokens     = <int>
      turboserve.completion_tokens = <int>
      event first_token  { turboserve.ttft_ms = <float>, milliseconds since arrival }
      event finished     { turboserve.finish_reason = "stop", turboserve.status = "ok" }

``first_token`` is where time-to-first-token lands in a trace, and it is the event that makes
a trace comparable with the ``ttft_seconds`` histogram: both are measured from the request's
arrival, not from the moment the backend was called.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Mapping

    from fastapi import FastAPI

    from turboserve.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "ATTR_BACKEND",
    "use_tracing",
    "ATTR_COMPLETION_TOKENS",
    "ATTR_FINISH_REASON",
    "ATTR_LANE",
    "ATTR_LORA",
    "ATTR_MODEL",
    "ATTR_PROMPT_TOKENS",
    "ATTR_STATUS",
    "ATTR_TENANT",
    "ATTR_TTFT_MS",
    "EVENT_FINISHED",
    "EVENT_FIRST_TOKEN",
    "SPAN_NAME",
    "GenerationSpan",
    "NO_SPAN",
    "Tracing",
    "configure_tracing",
    "inject_trace_context",
    "instrument_app",
]

#: The span one generation produces. One name, so a query for it finds every request this
#: gateway served whatever route, engine or lane answered it.
SPAN_NAME: Final = "turboserve.generate"

EVENT_FIRST_TOKEN: Final = "first_token"
EVENT_FINISHED: Final = "finished"

# Attribute keys. Namespaced, because OpenTelemetry's semantic conventions own the
# unprefixed namespace and a future `model` or `status` there would silently collide with
# ours. The suffixes are the label names `GatewayMetrics` already uses, so a span and a
# metric series are joined on the same words.
ATTR_TENANT: Final = "turboserve.tenant"
ATTR_MODEL: Final = "turboserve.model"
ATTR_BACKEND: Final = "turboserve.backend"
ATTR_LANE: Final = "turboserve.lane"
ATTR_LORA: Final = "turboserve.lora"
ATTR_PROMPT_TOKENS: Final = "turboserve.prompt_tokens"
ATTR_COMPLETION_TOKENS: Final = "turboserve.completion_tokens"
ATTR_TTFT_MS: Final = "turboserve.ttft_ms"
ATTR_FINISH_REASON: Final = "turboserve.finish_reason"
ATTR_STATUS: Final = "turboserve.status"

#: Routes excluded from the FastAPI instrumentation. Liveness, readiness and the metrics
#: scrape are polled every few seconds forever; tracing them would bury the requests that
#: matter under machine traffic and cost real money in a hosted trace backend.
EXCLUDED_URLS: Final = "healthz,readyz,metrics"


class GenerationSpan:
    """One in-flight ``turboserve.generate`` span, fed by the request path.

    Wraps the OpenTelemetry span rather than exposing it, for two reasons. The request path
    then never branches on whether tracing is on -- :data:`NO_SPAN` answers the same calls
    with nothing -- and every attribute this repository records is written in one place, so
    "no prompt text in spans" is a property of this file instead of a rule each call site is
    trusted to remember.
    """

    __slots__ = ("_ended", "_saw_first_token", "_span")

    def __init__(self, span: Any) -> None:
        self._span = span
        self._saw_first_token = False
        self._ended = False

    def activate(self) -> Any:
        """Make this span current for the duration of a ``with`` block.

        Used around the *first* pull from the backend, which is where the outgoing HTTP
        request is built and therefore the only moment at which
        :func:`inject_trace_context` can put a ``traceparent`` on it. Deliberately not held
        across the whole stream: the streaming body runs in a different task from the route
        handler, and a context token attached in one task and detached in another is how an
        instrumented server starts reporting other people's spans as parents.
        """
        from opentelemetry import trace

        return trace.use_span(self._span, end_on_exit=False, record_exception=False)

    def set_route(self, *, backend: str, lane: str) -> None:
        """Record which replica and lane answered, once the router has chosen one."""
        if backend:
            self._span.set_attribute(ATTR_BACKEND, backend)
        if lane:
            self._span.set_attribute(ATTR_LANE, lane)

    def first_token(self, ttft_s: float | None) -> None:
        """Mark the first output token, at most once, with the TTFT it implies."""
        if self._saw_first_token:
            return
        self._saw_first_token = True
        attributes = {} if ttft_s is None else {ATTR_TTFT_MS: round(ttft_s * 1000.0, 3)}
        self._span.add_event(EVENT_FIRST_TOKEN, attributes=attributes)

    def finished(self, *, reason: str | None, status: str) -> None:
        """Mark the terminating event, with why the completion ended and how it went."""
        attributes: dict[str, Any] = {ATTR_STATUS: status}
        if reason:
            attributes[ATTR_FINISH_REASON] = reason
        self._span.add_event(EVENT_FINISHED, attributes=attributes)

    def set_usage(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        """Record the authoritative token counts, known only when the stream ends."""
        self._span.set_attribute(ATTR_PROMPT_TOKENS, prompt_tokens)
        self._span.set_attribute(ATTR_COMPLETION_TOKENS, completion_tokens)

    def record_error(self, message: str) -> None:
        """Mark the span failed, with the message a client was given.

        The message, not the exception: an exception's traceback can quote a request body,
        and the status description is the one field of a span a UI shows by default.
        """
        from opentelemetry.trace import Status, StatusCode

        self._span.set_status(Status(StatusCode.ERROR, message))

    def end(self) -> None:
        """Close the span. Idempotent: the request path ends it on every exit."""
        if self._ended:
            return
        self._ended = True
        self._span.end()


class _NullGenerationSpan:
    """What the request path gets when tracing is off: every call, no effect.

    A null object rather than ``None`` so that ``session.span.first_token(...)`` needs no
    guard at any of the half-dozen places the stream is folded into the accounting. Its
    :meth:`activate` returns a plain ``nullcontext``, so the disabled path imports nothing
    from ``opentelemetry`` at all.
    """

    __slots__ = ()

    def activate(self) -> Any:
        return nullcontext()

    def set_route(self, *, backend: str, lane: str) -> None:
        return None

    def first_token(self, ttft_s: float | None) -> None:
        return None

    def finished(self, *, reason: str | None, status: str) -> None:
        return None

    def set_usage(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        return None

    def record_error(self, message: str) -> None:
        return None

    def end(self) -> None:
        return None


#: The span object a disabled :class:`Tracing` hands out. One instance: it holds no state.
NO_SPAN: Final = _NullGenerationSpan()


@dataclass(frozen=True, slots=True)
class Tracing:
    """A configured tracer, or a disabled stand-in for one.

    Always a real object, never ``None``, so :class:`~turboserve.gateway.app.GatewayState`
    has one field with one type and the routes have no branch. ``enabled`` is the only
    question anything outside this module ever asks, and only the app's start-up log asks it.
    """

    tracer: Any = None
    provider: Any = None
    endpoint: str | None = None
    service_name: str = ""

    @property
    def enabled(self) -> bool:
        """Whether spans are actually being produced."""
        return self.tracer is not None

    def generation(
        self,
        *,
        tenant: str,
        model: str,
        lora: str | None = None,
        prompt_tokens: int = 0,
    ) -> GenerationSpan | _NullGenerationSpan:
        """Open the span for one generation.

        Started rather than entered: the span outlives the function that creates it -- a
        streaming response is closed by the SSE generator, in another task -- so its lifetime
        is managed by the session that owns it and ended in that session's ``finally``.
        """
        if self.tracer is None:
            return NO_SPAN
        attributes: dict[str, Any] = {
            ATTR_TENANT: tenant,
            ATTR_MODEL: model,
            ATTR_PROMPT_TOKENS: prompt_tokens,
        }
        if lora:
            attributes[ATTR_LORA] = lora
        return GenerationSpan(self.tracer.start_span(SPAN_NAME, attributes=attributes))

    def shutdown(self) -> None:
        """Flush and stop the exporter. Safe to call when disabled, and idempotent."""
        if self.provider is not None:
            self.provider.shutdown()


def configure_tracing(settings: Settings, *, exporter: Any = None) -> Tracing:
    """Build the gateway's tracer from ``settings``, or a disabled :class:`Tracing`.

    Disabled -- and importing nothing -- unless ``settings.otel_endpoint`` is set, which is
    ``TURBOSERVE_OTEL_ENDPOINT`` in the environment. That is the whole switch: a deployment
    with no collector says nothing and gets nothing.

    ``exporter`` replaces the OTLP/HTTP exporter with any ``SpanExporter``, which is how the
    tests drive the real SDK against an ``InMemorySpanExporter`` instead of a socket. It is
    exported through a ``SimpleSpanProcessor`` rather than the batching one, so a test can
    read a finished span without flushing; the OTLP path keeps ``BatchSpanProcessor``,
    because a serving request must never wait on a trace backend.

    The endpoint is still required with an ``exporter``: "tracing is on" is a deployment
    decision and must have exactly one answer, or a test would be exercising a code path
    production cannot reach.
    """
    endpoint = (settings.otel_endpoint or "").strip()
    if not endpoint:
        return Tracing()
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except ImportError as exc:
        logger.warning(
            "TURBOSERVE_OTEL_ENDPOINT is set to %s but OpenTelemetry is not installed (%s); "
            "serving without tracing. Install the extra:  uv sync --extra otel",
            endpoint,
            exc,
        )
        return Tracing()

    from turboserve._version import __version__

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.namespace": settings.otel_service_namespace,
            "service.version": __version__,
            # The two settings that decide what these spans are *about*: which engine served
            # them and which checkpoint. Resource attributes rather than span attributes
            # because they are true of the process for its whole life, and repeating them on
            # every span is bytes on the wire for no extra information.
            "turboserve.engine": settings.engine,
            "turboserve.model": settings.model,
        }
    )
    # ParentBased: a request that arrives already sampled is recorded whatever the ratio
    # says, because half of somebody else's trace is worse than none of it. The ratio only
    # decides what happens to traces this gateway starts itself.
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(settings.otel_sample_ratio)),
    )
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    logger.info("tracing enabled: exporting %s spans to %s", settings.otel_service_name, endpoint)
    return Tracing(
        tracer=provider.get_tracer("turboserve.gateway", __version__),
        provider=provider,
        endpoint=endpoint,
        service_name=settings.otel_service_name,
    )


def instrument_app(app: FastAPI, tracing: Tracing) -> bool:
    """Attach the FastAPI auto-instrumentation to ``app``; report whether it was attached.

    The server span it creates is the parent of every ``turboserve.generate`` span and the
    thing that reads a client's incoming ``traceparent``, so without it a gateway would
    start a fresh trace per request and lose the caller. It is given this app's provider
    explicitly -- ``instrument_app`` rather than the global ``instrument()`` -- so that two
    apps in one process stay separate, and it is a no-op when tracing is disabled.
    """
    if not tracing.enabled:
        return False
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError as exc:  # pragma: no cover - the extra installs all four together
        logger.warning("FastAPI instrumentation unavailable (%s); spans will have no parent", exc)
        return False
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=tracing.provider, excluded_urls=EXCLUDED_URLS
    )
    return True


def inject_trace_context(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    """``headers`` plus the W3C ``traceparent`` of the current context, if there is one.

    This is what makes the engine's own spans children of the gateway's rather than roots of
    their own trace: vLLM and SGLang both run the OpenTelemetry Python instrumentation when
    they are configured for it, and both read the standard header. A copy is returned rather
    than the argument mutated, because the caller's mapping is usually a client's shared
    default headers and injecting into it would pin one request's trace id to every later
    request on that connection pool.

    Returns the headers unchanged when tracing is off, when OpenTelemetry is not installed,
    or when no span is current -- the propagator writes nothing for an invalid context, so
    there is no branch here for the disabled case.
    """
    carrier = dict(headers or {})
    try:
        from opentelemetry.propagate import inject
    except ImportError:
        return carrier
    inject(carrier)
    return carrier


@contextmanager
def use_tracing(settings: Settings, *, exporter: Any = None) -> Iterator[Tracing]:
    """A :func:`configure_tracing` that always shuts its provider down again.

    For anything that builds a gateway for a bounded time -- the tests, the chaos harness's
    in-process upstream -- where leaving a ``BatchSpanProcessor`` thread behind would keep
    exporting after the thing it describes is gone.
    """
    tracing = configure_tracing(settings, exporter=exporter)
    try:
        yield tracing
    finally:
        tracing.shutdown()
