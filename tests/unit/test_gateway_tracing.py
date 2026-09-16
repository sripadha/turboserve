"""Tracing: what a generation span carries, and what it costs when nobody asked for one.

Driven against the real OpenTelemetry SDK with an ``InMemorySpanExporter``, not against a
mock of it. A mock would assert that this code calls the methods this code calls; the
exporter asserts that a span with these attributes and these events actually reaches a
collector, which is the thing an operator is going to query.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from turboserve.config import Settings
from turboserve.gateway.app import GatewayOptions, create_app
from turboserve.gateway.auth import hash_api_key
from turboserve.gateway.backends.mock import MockBackend
from turboserve.gateway.metrics import GatewayMetrics
from turboserve.gateway.router import Router
from turboserve.gateway.tenants import Tenant, TenantRegistry
from turboserve.gateway.tracing import (
    ATTR_BACKEND,
    ATTR_COMPLETION_TOKENS,
    ATTR_FINISH_REASON,
    ATTR_LANE,
    ATTR_LORA,
    ATTR_MODEL,
    ATTR_PROMPT_TOKENS,
    ATTR_STATUS,
    ATTR_TENANT,
    ATTR_TTFT_MS,
    EVENT_FINISHED,
    EVENT_FIRST_TOKEN,
    NO_SPAN,
    SPAN_NAME,
    Tracing,
    configure_tracing,
    inject_trace_context,
    use_tracing,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

KEY = "sk-test-acme"
MODEL = "mock-model"
AUTH = {"Authorization": f"Bearer {KEY}"}
ENDPOINT = "http://127.0.0.1:4318/v1/traces"


def build_settings(**overrides: Any) -> Settings:
    """Settings pointed at files that do not exist, so no repository config is read."""
    return Settings(
        tenants_file="/nonexistent/tenants.yaml",
        models_file="/nonexistent/models.yaml",
        **overrides,
    )


def build_tenants() -> TenantRegistry:
    return TenantRegistry(
        [
            Tenant(
                tenant_id="acme",
                keys_sha256=[hash_api_key(KEY)],
                max_concurrency=4,
                adapters={"support": "acme-support-r16"},
            )
        ]
    )


def build_app(tracing: Tracing, settings: Settings) -> Any:
    """A gateway around one mock backend, with the tracer under test injected."""
    router = Router()
    router.add_backend(
        MODEL,
        MockBackend(name="mock-a", models=[MODEL], adapters=["acme-support-r16"], max_tokens=4),
    )
    return create_app(
        settings,
        router,
        tenants=build_tenants(),
        metrics=GatewayMetrics(),
        options=GatewayOptions(),
        tracing=tracing,
    )


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def traced(exporter: InMemorySpanExporter) -> Iterator[tuple[TestClient, InMemorySpanExporter]]:
    """A client whose gateway exports spans into ``exporter``."""
    settings = build_settings(otel_endpoint=ENDPOINT, otel_service_name="turboserve-test")
    with use_tracing(settings, exporter=exporter) as tracing:
        assert tracing.enabled
        with TestClient(build_app(tracing, settings)) as client:
            yield client, exporter


def generation_spans(exporter: InMemorySpanExporter) -> list[Any]:
    """Every ``turboserve.generate`` span exported so far, oldest first."""
    return [span for span in exporter.get_finished_spans() if span.name == SPAN_NAME]


def event_named(span: Any, name: str) -> Any:
    """The one event of ``span`` called ``name``; fails the test when there is not exactly one."""
    matches = [event for event in span.events if event.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} event, got {len(matches)}"
    return matches[0]


# --- disabled by default ------------------------------------------------------------------


def test_no_endpoint_means_no_tracer_and_no_span() -> None:
    """The switch is the endpoint, and with it unset the gateway builds nothing."""
    tracing = configure_tracing(build_settings())
    assert not tracing.enabled
    assert tracing.provider is None
    assert tracing.endpoint is None
    assert tracing.generation(tenant="acme", model=MODEL) is NO_SPAN
    # Shutting a disabled tracer down is legal: create_app's caller does not branch either.
    tracing.shutdown()


def test_blank_endpoint_is_treated_as_unset() -> None:
    """An empty TURBOSERVE_OTEL_ENDPOINT is how a chart renders "no collector here"."""
    assert not configure_tracing(build_settings(otel_endpoint="   ")).enabled


def test_a_gateway_without_tracing_still_serves(exporter: InMemorySpanExporter) -> None:
    """The disabled path is the default one, so it is the one that must not regress."""
    settings = build_settings()
    with TestClient(build_app(configure_tracing(settings), settings)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert exporter.get_finished_spans() == ()


# --- the generation span ------------------------------------------------------------------


def test_generation_span_carries_the_request_identity(
    traced: tuple[TestClient, InMemorySpanExporter],
) -> None:
    client, exporter = traced
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there"}],
            "lora": "support",
        },
    )
    assert response.status_code == 200

    spans = generation_spans(exporter)
    assert len(spans) == 1
    attributes = spans[0].attributes
    assert attributes[ATTR_TENANT] == "acme"
    assert attributes[ATTR_MODEL] == MODEL
    assert attributes[ATTR_BACKEND] == "mock-a"
    assert attributes[ATTR_LANE] == "stable"
    # The adapter is recorded as the *resolved* name the backend was asked for, not the
    # alias the tenant used, because that is what identifies the weights that served it.
    assert attributes[ATTR_LORA] == "acme-support-r16"
    assert attributes[ATTR_PROMPT_TOKENS] > 0
    assert attributes[ATTR_COMPLETION_TOKENS] == response.json()["usage"]["completion_tokens"]


def test_generation_span_records_first_token_and_finish(
    traced: tuple[TestClient, InMemorySpanExporter],
) -> None:
    """``first_token`` is where TTFT lands in a trace; ``finished`` says why it stopped."""
    client, exporter = traced
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert body.endswith("data: [DONE]\n\n")

    span = generation_spans(exporter)[0]
    first_token = event_named(span, EVENT_FIRST_TOKEN)
    assert first_token.attributes[ATTR_TTFT_MS] > 0.0
    finished = event_named(span, EVENT_FINISHED)
    assert finished.attributes[ATTR_STATUS] == "ok"
    assert finished.attributes[ATTR_FINISH_REASON] in {"stop", "length"}
    # The events are ordered as they happened, and both fall inside the span.
    assert first_token.timestamp <= finished.timestamp
    assert span.start_time <= first_token.timestamp <= span.end_time


def test_no_prompt_or_completion_text_reaches_a_span(
    traced: tuple[TestClient, InMemorySpanExporter],
) -> None:
    """The rule this module exists to keep: ids and counts, never content."""
    client, exporter = traced
    secret = "the-tenants-private-prompt"
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": MODEL, "messages": [{"role": "user", "content": secret}]},
    )
    assert response.status_code == 200

    span = generation_spans(exporter)[0]
    rendered = repr(span.attributes) + repr([repr(event.attributes) for event in span.events])
    assert secret not in rendered
    assert KEY not in rendered


def test_a_failed_request_produces_no_orphan_span(
    traced: tuple[TestClient, InMemorySpanExporter],
) -> None:
    """A refusal before the backend is reached must not leave a span open forever."""
    client, exporter = traced
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "not-served", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert generation_spans(exporter) == []


def test_every_request_gets_its_own_span(
    traced: tuple[TestClient, InMemorySpanExporter],
) -> None:
    client, exporter = traced
    for _ in range(3):
        assert (
            client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
            ).status_code
            == 200
        )
    spans = generation_spans(exporter)
    assert len(spans) == 3
    assert len({span.context.span_id for span in spans}) == 3


def test_the_scrape_and_probe_routes_are_not_traced(
    traced: tuple[TestClient, InMemorySpanExporter],
) -> None:
    """Liveness, readiness and /metrics are polled forever; tracing them buries the rest."""
    client, exporter = traced
    for path in ("/healthz", "/readyz", "/metrics"):
        client.get(path)
    assert exporter.get_finished_spans() == ()


# --- propagation ---------------------------------------------------------------------------


def test_inject_writes_a_traceparent_inside_a_span(exporter: InMemorySpanExporter) -> None:
    """What puts the engine's spans under the gateway's rather than in a trace of their own."""
    settings = build_settings(otel_endpoint=ENDPOINT)
    with use_tracing(settings, exporter=exporter) as tracing:
        span = tracing.generation(tenant="acme", model=MODEL)
        with span.activate():
            headers = inject_trace_context({"accept": "text/event-stream"})
        span.end()
    assert headers["accept"] == "text/event-stream"
    assert "traceparent" in headers
    exported = generation_spans(exporter)[0]
    assert format(exported.context.trace_id, "032x") in headers["traceparent"]


def test_inject_does_not_mutate_the_caller_headers(exporter: InMemorySpanExporter) -> None:
    """The caller's mapping is a client's shared defaults; a traceparent there pins one
    request's trace id to every later request on the same connection pool."""
    settings = build_settings(otel_endpoint=ENDPOINT)
    shared = {"accept": "text/event-stream"}
    with use_tracing(settings, exporter=exporter) as tracing:
        span = tracing.generation(tenant="acme", model=MODEL)
        with span.activate():
            injected = inject_trace_context(shared)
        span.end()
    assert shared == {"accept": "text/event-stream"}
    assert injected is not shared


def test_inject_outside_a_span_returns_the_headers_unchanged() -> None:
    """No active context is not an error: the propagator writes nothing and nothing breaks."""
    assert inject_trace_context({"accept": "application/json"}) == {"accept": "application/json"}
    assert inject_trace_context() == {}


def test_the_openai_backend_puts_a_traceparent_on_its_request(
    exporter: InMemorySpanExporter,
) -> None:
    """The propagation the engine actually sees, asserted on the outgoing request itself.

    Driven through ``httpx.MockTransport`` against a synthetic transcript: no network, no
    server, and the header is read off the request the backend built rather than inferred
    from the fact that ``inject`` was called.
    """
    import httpx

    from turboserve.gateway.backends.openai_compat import OpenAICompatBackend
    from turboserve.gateway.backends.protocol import GenerateRequest

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        body = "data: " + '{"choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}]}'
        return httpx.Response(
            200,
            content=(body + "\n\ndata: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://upstream/v1"
    )
    backend = OpenAICompatBackend("http://upstream/v1", client=client, name="upstream")
    request = GenerateRequest.model_validate(
        {"request_id": "req-1", "tenant_id": "acme", "model": MODEL, "prompt": "hello"}
    )

    settings = build_settings(otel_endpoint=ENDPOINT)
    with use_tracing(settings, exporter=exporter) as tracing:
        span = tracing.generation(tenant="acme", model=MODEL)

        async def drive() -> None:
            with span.activate():
                async for _ in backend.generate(request):
                    break
            await backend.close()

        asyncio.run(drive())
        span.end()

    exported = generation_spans(exporter)[0]
    assert format(exported.context.trace_id, "032x") in seen["traceparent"]
    assert format(exported.context.span_id, "016x") in seen["traceparent"]
