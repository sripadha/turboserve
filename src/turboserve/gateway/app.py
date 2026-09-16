"""The OpenAI-compatible multi-tenant HTTP gateway.

One request travels a fixed sequence of gates, and the order is the design:

``authenticate`` -> ``authorize model`` -> ``resolve adapter`` -> ``quota`` -> ``route`` ->
``stream`` -> ``account``

Each gate can only refuse with a status that is actionable for the client, and the cheap
refusals come first: a request from an unknown key never reaches the token bucket, and a
request over quota never reaches a GPU. The one gate that is *not* free -- routing -- is
entered before the response is opened, because the router may retry a replica and that is
only legal while nothing has been sent (see :mod:`turboserve.gateway.router`).

Streaming uses the OpenAI wire format exactly: ``data: {json}\\n\\n`` per chunk and a final
``data: [DONE]\\n\\n``. The separator is pinned to ``\\n`` rather than the CRLF an SSE library
defaults to, because that is what the reference clients emit and what the strict parser in
this repository's tests asserts on.

Everything the app needs is held in :class:`GatewayState` on ``app.state`` and injected
through ``create_app``, so a test builds an app around a :class:`MockBackend`, a fake clock
and a private metrics registry without patching a single module-level name.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from fastapi import FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from turboserve.config import Settings
from turboserve.gateway.auth import Authenticator, AuthError, Principal, hash_api_key
from turboserve.gateway.backends.protocol import (
    BackendError,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
    RetryableBackendError,
    TokenEvent,
)
from turboserve.gateway.chat_template import ChatTemplateCache
from turboserve.gateway.limits import LimiterRegistry, RateLimitExceeded, retry_after_header
from turboserve.gateway.metrics import GatewayMetrics
from turboserve.gateway.openai_types import (
    SSE_DONE,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    UsageInfo,
    created_timestamp,
    new_chat_completion_id,
    new_completion_id,
    openai_finish_reason,
)
from turboserve.gateway.router import ModelsFile, Router, RouterConfigError, build_router
from turboserve.gateway.tenants import TenantConfigError, TenantRegistry
from turboserve.gateway.tracing import Tracing, configure_tracing, instrument_app
from turboserve.gateway.usage import PriceTable, UsageAccumulator, UsageRecord, UsageTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncGenerator, AsyncIterator

    from turboserve.gateway.router import CanaryWeightSource, RoutedEvent
    from turboserve.gateway.tracing import GenerationSpan, _NullGenerationSpan

logger = logging.getLogger(__name__)

__all__ = [
    "GatewayOptions",
    "GatewayState",
    "create_app",
    "gateway_app",
    "router_for_engine",
]

_HTTP_BAD_REQUEST = 400
_HTTP_NOT_FOUND = 404
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_BAD_GATEWAY = 502
_HTTP_SERVICE_UNAVAILABLE = 503
_HTTP_GATEWAY_TIMEOUT = 504


@dataclass(slots=True)
class GatewayOptions:
    """Behavioural switches that are not fleet data and not per-process tuning.

    They are constructor arguments rather than :class:`~turboserve.config.Settings` fields
    because each one is a property of *this app object*: two apps in one process (the real
    gateway and an in-process mock upstream, which is exactly the chaos-harness layout) need
    different values, and environment variables cannot give them different values.
    """

    require_auth: bool = True
    anonymous_tenant_id: str = "default"
    default_max_tokens: int = 128
    sse_ping_interval_s: float = 900.0
    """Keep-alive comment interval for SSE. Long by default: a comment line is legal SSE but
    surprising in a captured transcript, and a completion normally finishes well inside it."""

    local_files_only: bool = True
    """Whether chat-template tokenizers may be downloaded. Off by default: a server start
    must never turn into a multi-gigabyte download."""

    include_process_metrics: bool = False
    served_model_names: list[str] = field(default_factory=list)
    """Extra names advertised by ``GET /v1/models`` beyond the router's pools."""


@dataclass(slots=True)
class GatewayState:
    """Everything a request handler needs, assembled once at app creation."""

    settings: Settings
    router: Router
    auth: Authenticator
    tenants: TenantRegistry
    limiters: LimiterRegistry
    metrics: GatewayMetrics
    usage: UsageTracker
    templates: ChatTemplateCache
    options: GatewayOptions
    tracing: Tracing = field(default_factory=Tracing)
    """Spans for this app, or a disabled stand-in. Never ``None``: the request path calls
    it unconditionally and a disabled one answers every call with nothing."""

    def served_models(self) -> list[str]:
        """Model names this gateway advertises."""
        return sorted({*self.router.models(), *self.options.served_model_names})


def _state(request: Request) -> GatewayState:
    """Pull the gateway state off the app; a plain function, so routes stay easy to read."""
    state: GatewayState = request.app.state.gateway
    return state


def _error_response(
    message: str,
    *,
    status_code: int,
    type_: str = "invalid_request_error",
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Render an OpenAI-shaped error body, which is what clients unwrap into exceptions."""
    body = ErrorResponse.of(message, type_=type_, code=code)
    return JSONResponse(body.model_dump(), status_code=status_code, headers=headers)


def _backend_status(exc: BackendError) -> int:
    """Map a backend failure onto the status the client should see.

    A retryable failure the router could not work around is a 503 (come back later); a
    rejected request is a 400 (fix it); an unknown model is a 404. The distinction is what
    makes a client's retry policy correct without it having to parse messages.
    """
    if isinstance(exc, ModelNotFoundError):
        return _HTTP_NOT_FOUND
    if isinstance(exc, BackendTimeoutError):
        return _HTTP_GATEWAY_TIMEOUT
    if isinstance(exc, BackendUnavailableError | RetryableBackendError):
        return _HTTP_SERVICE_UNAVAILABLE
    if isinstance(exc, BackendRequestError):
        return _HTTP_BAD_REQUEST
    return _HTTP_BAD_GATEWAY


# --------------------------------------------------------------------------------------
# Request preparation
# --------------------------------------------------------------------------------------


def _count_prompt_tokens(
    state: GatewayState, model: str, prompt: str | list[int]
) -> tuple[int, bool]:
    """Prompt length in tokens and whether that number is an estimate.

    Token ids are exact. Text is tokenised when a local tokenizer for the model exists, and
    otherwise estimated from its length -- the gateway still has to charge the tenant's token
    bucket *before* it sends the request, and the backend's authoritative count arrives only
    with the response.
    """
    if isinstance(prompt, list):
        return len(prompt), False
    counted = state.templates.get(model).count_tokens(prompt)
    if counted is not None:
        return counted, False
    from turboserve.gateway.usage import estimate_text_tokens

    return estimate_text_tokens(prompt), True


def _admit(state: GatewayState, principal: Principal, model: str, prompt_tokens: int) -> None:
    """Charge the tenant's rate quotas, or raise :class:`RateLimitExceeded`."""
    limiter = state.limiters.for_tenant(principal.tenant)
    try:
        limiter.check_request(prompt_tokens=prompt_tokens)
    except RateLimitExceeded as exc:
        state.metrics.record_rate_limited(tenant=principal.tenant_id, limit=exc.limit)
        state.metrics.record_request(tenant=principal.tenant_id, model=model, status="rate_limited")
        raise


def _generate_request(
    state: GatewayState,
    principal: Principal,
    body: CompletionRequest | ChatCompletionRequest,
    prompt: str | list[int],
    adapter: str | None,
) -> GenerateRequest:
    """Normalise a validated HTTP body into the backend-facing request."""
    return GenerateRequest(
        request_id=f"req-{uuid.uuid4().hex}",
        tenant_id=principal.tenant_id,
        model=body.model,
        prompt=prompt,
        sampling=body.sampling_params(default_max_tokens=state.options.default_max_tokens),
        lora=adapter,
        priority=body.priority if body.priority is not None else principal.tenant.priority,
        stream=body.stream,
    )


@dataclass(slots=True)
class _Session:
    """One admitted request, its open stream, its accounting and its span."""

    state: GatewayState
    principal: Principal
    request: GenerateRequest
    accumulator: UsageAccumulator
    stream: AsyncGenerator[RoutedEvent, None]
    first: RoutedEvent
    span: GenerationSpan | _NullGenerationSpan
    finished: bool = False

    @property
    def model(self) -> str:
        """The model name the client asked for."""
        return self.request.model

    def feed(self, routed: RoutedEvent) -> TokenEvent:
        """Fold one routed event into the accounting and the span; return the inner event.

        The span is fed from the accumulator rather than from the event, so a trace's
        ``first_token`` event carries the same time-to-first-token the metrics histogram and
        the usage record carry -- measured from arrival, through auth, quotas and routing.
        """
        event = routed.event
        self.accumulator.on_event(event)
        if self.accumulator.t_first_token is not None:
            self.span.first_token(self.accumulator.ttft_s)
        if event.is_error:
            self.span.record_error(event.error or "stream failed")
        if event.finished:
            self.span.finished(
                reason=str(event.finish_reason) if event.finish_reason is not None else None,
                status="error" if event.is_error else "ok",
            )
        return event

    def complete(self, *, status: str = "ok") -> UsageRecord:
        """Close the accounting exactly once, end the span, and publish the record."""
        if self.finished:
            return self.state.usage.record(self.accumulator.finish(status=status))
        self.finished = True
        limiter = self.state.limiters.for_tenant(self.principal.tenant)
        limiter.charge_tokens(self.accumulator.completion_tokens)
        record = self.state.usage.complete(self.accumulator, status=status)
        self.state.metrics.dec_inflight(tenant=record.tenant_id, model=record.model)
        self.state.metrics.set_queue_depth(
            model=record.model, value=self.state.router.queue_depth()
        )
        self.span.set_usage(
            prompt_tokens=record.prompt_tokens, completion_tokens=record.completion_tokens
        )
        self.span.end()
        return record


async def _open_session(
    state: GatewayState,
    principal: Principal,
    gen_request: GenerateRequest,
    *,
    prompt_tokens: int,
    estimated: bool,
) -> _Session:
    """Start routing and pull the first event, before any response body exists.

    This is where the gateway's retry semantics are realised: the router may still switch
    replicas while this coroutine runs, and it may still raise a status-bearing error. From
    the moment it returns, the outcome can only be delivered inside the stream.
    """
    accumulator = state.usage.start(
        request_id=gen_request.request_id,
        tenant_id=gen_request.tenant_id,
        model=gen_request.model,
        prompt_tokens=prompt_tokens,
        estimated=estimated,
        arrival_ts=gen_request.arrival_ts,
    )
    span = state.tracing.generation(
        tenant=gen_request.tenant_id,
        model=gen_request.model,
        lora=gen_request.lora,
        prompt_tokens=prompt_tokens,
    )
    stream = state.router.generate(gen_request)
    try:
        # The span is made current only for this first pull, which is where an HTTP backend
        # builds its outgoing request and therefore the only moment a `traceparent` can be
        # put on it. Holding it across the whole stream would mean attaching a context in
        # this task and detaching it in the SSE generator's, which is a different task.
        with span.activate():
            first = await anext(stream)
    except StopAsyncIteration as exc:
        await stream.aclose()
        message = f"backend produced no events for request {gen_request.request_id}"
        span.record_error(message)
        span.end()
        raise BackendUnavailableError(message) from exc
    except BaseException as exc:
        await stream.aclose()
        span.record_error(str(exc))
        span.end()
        raise
    accumulator.backend = first.backend
    accumulator.lane = first.lane
    span.set_route(backend=first.backend, lane=first.lane)
    session = _Session(
        state=state,
        principal=principal,
        request=gen_request,
        accumulator=accumulator,
        stream=stream,
        first=first,
        span=span,
    )
    state.metrics.inc_inflight(tenant=gen_request.tenant_id, model=gen_request.model)
    state.metrics.set_queue_depth(model=gen_request.model, value=state.router.queue_depth())
    session.feed(first)
    return session


async def _remaining(session: _Session) -> AsyncIterator[TokenEvent]:
    """Yield the events after the first, folding each into the accounting."""
    async for routed in session.stream:
        yield session.feed(routed)


def _usage_info(record: UsageRecord) -> UsageInfo:
    """Render a usage record as the response body's ``usage`` object."""
    return UsageInfo.model_validate(record.to_openai_usage())


# --------------------------------------------------------------------------------------
# Response rendering
# --------------------------------------------------------------------------------------


def _chunk_json(payload: Any) -> str:
    """Serialise one SSE payload.

    Nulls are kept: OpenAI's chunks carry ``"finish_reason": null`` on every non-final
    chunk, and clients treat the key as required. The only object that hides its unset
    members is :class:`ChatCompletionDelta`, which does so in its own serialiser.
    """
    return payload.model_dump_json()


async def _completion_sse(session: _Session, body: CompletionRequest) -> AsyncIterator[str]:
    """Stream a ``/v1/completions`` response as OpenAI text-completion chunks."""
    completion_id = new_completion_id()
    created = created_timestamp()
    status = "ok"
    try:
        first_event = session.first.event
        async for event in _prepended(first_event, _remaining(session)):
            if event.is_error:
                status = "error"
                yield _chunk_json(
                    ErrorResponse.of(event.error or "stream failed", type_="server_error")
                )
                break
            if event.text or event.finished:
                yield _chunk_json(
                    CompletionResponse(
                        id=completion_id,
                        created=created,
                        model=body.model,
                        choices=[
                            CompletionChoice(
                                text=event.text,
                                finish_reason=openai_finish_reason(event.finish_reason)
                                if event.finished
                                else None,
                            )
                        ],
                    )
                )
    except BackendError as exc:  # a failure the router could not turn into an event
        status = "error"
        logger.warning("completion stream %s failed: %s", session.request.request_id, exc)
        yield _chunk_json(ErrorResponse.of(str(exc), type_="server_error"))
    finally:
        record = session.complete(status=status)
        if body.include_usage:
            yield _chunk_json(
                CompletionResponse(
                    id=completion_id,
                    created=created,
                    model=body.model,
                    choices=[],
                    usage=_usage_info(record),
                )
            )
        yield SSE_DONE


async def _chat_sse(session: _Session, body: ChatCompletionRequest) -> AsyncIterator[str]:
    """Stream a ``/v1/chat/completions`` response as OpenAI chat chunks.

    The first chunk carries only ``delta.role``; content chunks carry only
    ``delta.content``. Clients concatenate ``delta.content`` and would otherwise see the
    role repeated inside the message text.
    """
    completion_id = new_chat_completion_id()
    created = created_timestamp()
    status = "ok"

    def chunk(delta: ChatCompletionDelta, finish_reason: str | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=body.model,
            choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason)],
        )

    try:
        yield _chunk_json(chunk(ChatCompletionDelta(role="assistant")))
        async for event in _prepended(session.first.event, _remaining(session)):
            if event.is_error:
                status = "error"
                yield _chunk_json(
                    ErrorResponse.of(event.error or "stream failed", type_="server_error")
                )
                break
            if event.text:
                yield _chunk_json(chunk(ChatCompletionDelta(content=event.text)))
            if event.finished:
                yield _chunk_json(
                    chunk(
                        ChatCompletionDelta(),
                        openai_finish_reason(event.finish_reason),
                    )
                )
    except BackendError as exc:
        status = "error"
        logger.warning("chat stream %s failed: %s", session.request.request_id, exc)
        yield _chunk_json(ErrorResponse.of(str(exc), type_="server_error"))
    finally:
        record = session.complete(status=status)
        if body.include_usage:
            yield _chunk_json(
                ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=body.model,
                    choices=[],
                    usage=_usage_info(record),
                )
            )
        yield SSE_DONE


async def _prepended(
    first: TokenEvent, rest: AsyncIterator[TokenEvent]
) -> AsyncIterator[TokenEvent]:
    """Put an already-pulled event back at the head of a stream."""
    yield first
    async for event in rest:
        yield event


async def _collect(session: _Session) -> tuple[str, TokenEvent]:
    """Drain a stream into one string and return it with the terminating event."""
    parts: list[str] = []
    last = session.first.event
    async for event in _prepended(session.first.event, _remaining(session)):
        last = event
        if event.text:
            parts.append(event.text)
    return "".join(parts), last


# --------------------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------------------


def _default_router(settings: Settings, canary: CanaryWeightSource | None) -> Router:
    """Build a router when the caller supplied none.

    Prefers the models file; falls back to a single mock replica serving
    ``settings.model``. The fallback exists so that a fresh checkout, the kind end-to-end
    test and ``turboserve gateway serve --engine mock`` all start without configuration --
    and it logs loudly, because a production process reaching it is a misconfiguration.
    """
    models_file = Path(settings.models_file)
    if models_file.is_file():
        config = ModelsFile.from_yaml(models_file)
        if config.models:
            return build_router(config, canary=canary)
    from turboserve.gateway.backends.mock import MockBackend

    logger.warning(
        "no usable models file at %s; serving %r from an in-process mock backend",
        models_file,
        settings.model,
    )
    router = Router(canary=canary)
    router.add_backend(settings.model, MockBackend(name="mock", models=[settings.model]))
    return router


def create_app(
    settings: Settings | None = None,
    router: Router | None = None,
    *,
    tenants: TenantRegistry | None = None,
    prices: PriceTable | None = None,
    metrics: GatewayMetrics | None = None,
    limiters: LimiterRegistry | None = None,
    templates: ChatTemplateCache | None = None,
    options: GatewayOptions | None = None,
    canary: CanaryWeightSource | None = None,
    tracing: Tracing | None = None,
) -> FastAPI:
    """Build the gateway application.

    Every collaborator is injectable and every one has a sensible default, so production
    calls this with a settings object and tests call it with a router full of mocks. Nothing
    is read from module-level state, which is what allows two gateways to coexist in one
    process without sharing quotas or metrics -- tracing included: the provider is this
    app's, handed to the FastAPI instrumentation explicitly rather than installed globally.
    """
    settings = settings or Settings()
    options = options or GatewayOptions()
    resolved_router = router if router is not None else _default_router(settings, canary)
    if router is not None and canary is not None:
        resolved_router.set_canary(canary)

    tenants_file = Path(settings.tenants_file)
    if tenants is None:
        tenants = (
            TenantRegistry.from_yaml(tenants_file)
            if tenants_file.is_file()
            else TenantRegistry.default()
        )
    if not options.require_auth and tenants.get(options.anonymous_tenant_id) is None:
        tenants = TenantRegistry([*tenants, *TenantRegistry.default()])
    if options.require_auth:
        # The example keys are published in configs/tenants.yaml's own comments, so a
        # deployment that kept them is accepting credentials anyone can read off GitHub.
        # Refusing to start would break the quickstart, so this is loud rather than fatal.
        example_tenants = tenants.tenants_with_example_keys()
        if example_tenants:
            logger.warning(
                "tenants %s still accept the example API keys shipped in "
                "configs/tenants.yaml; their plaintext is public. Replace the digests "
                "(turboserve gateway hash-key) before exposing this gateway.",
                ", ".join(example_tenants),
            )

    resolved_tracing = tracing if tracing is not None else configure_tracing(settings)
    resolved_metrics = metrics or GatewayMetrics(
        include_process_metrics=options.include_process_metrics
    )
    resolved_limiters = limiters or LimiterRegistry(tenants)
    resolved_prices = prices if prices is not None else _prices_from_settings(settings)
    state = GatewayState(
        settings=settings,
        router=resolved_router,
        auth=Authenticator(
            tenants,
            require_auth=options.require_auth,
            anonymous_tenant_id=options.anonymous_tenant_id,
        ),
        tenants=tenants,
        limiters=resolved_limiters,
        metrics=resolved_metrics,
        usage=UsageTracker(prices=resolved_prices, metrics=resolved_metrics),
        templates=templates or ChatTemplateCache(local_files_only=options.local_files_only),
        options=options,
        tracing=resolved_tracing,
    )
    resolved_router.attach(limiters=resolved_limiters, metrics=resolved_metrics)

    app = FastAPI(
        title="turboserve gateway",
        description="OpenAI-compatible multi-tenant inference gateway.",
        version=_package_version(),
    )
    app.state.gateway = state
    _install_routes(app)
    _install_exception_handlers(app)
    # After the routes, so every one of them is wrapped: the server span this creates is the
    # parent of each `turboserve.generate` span and the thing that adopts a client's
    # incoming traceparent. A no-op when tracing is disabled.
    instrument_app(app, resolved_tracing)
    return app


def _prices_from_settings(settings: Settings) -> PriceTable:
    """Read the configured price table out of the models file, if there is one."""
    models_file = Path(settings.models_file)
    if not models_file.is_file():
        return PriceTable()
    try:
        return ModelsFile.from_yaml(models_file).price_table()
    except RouterConfigError as exc:
        logger.warning("ignoring price table in %s: %s", models_file, exc)
        return PriceTable()


def _package_version() -> str:
    """The installed package version, for the OpenAPI document."""
    from turboserve._version import __version__

    return __version__


def _install_exception_handlers(app: FastAPI) -> None:
    """Render every refusal as an OpenAI error body with the right status."""

    async def on_auth_error(request: Request, exc: Exception) -> Response:
        error = cast("AuthError", exc)
        headers = {"WWW-Authenticate": "Bearer"} if error.status_code == 401 else None
        _state(request).metrics.record_request(
            tenant="unknown",
            model="",
            status="unauthorized" if error.status_code == 401 else "forbidden",
        )
        return _error_response(
            error.message,
            status_code=error.status_code,
            type_=error.error_type,
            code=error.code,
            headers=headers,
        )

    async def on_rate_limited(request: Request, exc: Exception) -> Response:
        error = cast("RateLimitExceeded", exc)
        return _error_response(
            error.message,
            status_code=_HTTP_TOO_MANY_REQUESTS,
            type_="rate_limit_error",
            code=f"{error.limit}_exceeded",
            headers={"Retry-After": retry_after_header(error.retry_after_s)},
        )

    async def on_backend_error(request: Request, exc: Exception) -> Response:
        error = cast("BackendError", exc)
        status = _backend_status(error)
        logger.warning("backend error (%d): %s", status, error)
        return _error_response(
            str(error),
            status_code=status,
            type_="invalid_request_error" if status < 500 else "server_error",
        )

    async def on_validation_error(request: Request, exc: Exception) -> Response:
        # OpenAI answers a malformed body with 400, and clients branch on that; FastAPI's
        # default 422 makes them treat a bad request as an unknown protocol error.
        return _error_response(
            _first_validation_message(cast("RequestValidationError", exc)),
            status_code=_HTTP_BAD_REQUEST,
        )

    app.add_exception_handler(AuthError, on_auth_error)
    app.add_exception_handler(RateLimitExceeded, on_rate_limited)
    app.add_exception_handler(BackendError, on_backend_error)
    app.add_exception_handler(RequestValidationError, on_validation_error)


def _first_validation_message(exc: RequestValidationError) -> str:
    """Flatten pydantic's error list into the one sentence a client can act on."""
    errors = exc.errors()
    if not errors:
        return "invalid request body"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = str(first.get("msg", "invalid value"))
    return f"{location}: {message}" if location else message


def _install_routes(app: FastAPI) -> None:
    """Register every route on ``app``."""

    @app.post("/v1/completions")
    async def create_completion(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        body: CompletionRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Text completion, streaming or buffered."""
        state = _state(request)
        principal = state.auth.authenticate(authorization)
        state.auth.authorize_model(principal, body.model)
        _require_served(state, body.model)
        adapter = state.auth.resolve_adapter(principal, body.lora)
        prompt = body.single_prompt()
        prompt_tokens, estimated = _count_prompt_tokens(state, body.model, prompt)
        _admit(state, principal, body.model, prompt_tokens)
        gen_request = _generate_request(state, principal, body, prompt, adapter)
        session = await _open_session(
            state, principal, gen_request, prompt_tokens=prompt_tokens, estimated=estimated
        )
        if body.stream:
            return _sse(state, _completion_sse(session, body))
        text, last = await _collect(session)
        record = session.complete(status="error" if last.is_error else "ok")
        if last.is_error:
            return _error_response(
                last.error or "stream failed",
                status_code=_HTTP_BAD_GATEWAY,
                type_="server_error",
            )
        return JSONResponse(
            CompletionResponse(
                id=new_completion_id(),
                model=body.model,
                choices=[
                    CompletionChoice(
                        text=text, finish_reason=openai_finish_reason(last.finish_reason)
                    )
                ],
                usage=_usage_info(record),
            ).model_dump()
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        body: ChatCompletionRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Chat completion; the messages are rendered with the model's own chat template."""
        state = _state(request)
        principal = state.auth.authenticate(authorization)
        state.auth.authorize_model(principal, body.model)
        _require_served(state, body.model)
        adapter = state.auth.resolve_adapter(principal, body.lora)
        template = state.templates.get(body.model)
        prompt = template.render(body.messages, add_generation_prompt=body.add_generation_prompt)
        prompt_tokens, estimated = _count_prompt_tokens(state, body.model, prompt)
        _admit(state, principal, body.model, prompt_tokens)
        gen_request = _generate_request(state, principal, body, prompt, adapter)
        session = await _open_session(
            state, principal, gen_request, prompt_tokens=prompt_tokens, estimated=estimated
        )
        if body.stream:
            return _sse(state, _chat_sse(session, body))
        text, last = await _collect(session)
        record = session.complete(status="error" if last.is_error else "ok")
        if last.is_error:
            return _error_response(
                last.error or "stream failed",
                status_code=_HTTP_BAD_GATEWAY,
                type_="server_error",
            )
        return JSONResponse(
            ChatCompletionResponse(
                id=new_chat_completion_id(),
                model=body.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionMessage(content=text),
                        finish_reason=openai_finish_reason(last.finish_reason),
                    )
                ],
                usage=_usage_info(record),
            ).model_dump()
        )

    @app.get("/v1/models")
    async def list_models(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> ModelList:
        """Models this caller may address.

        Filtered per tenant: a model the caller would be refused for must not appear in the
        picker of an OpenAI client, and listing it would leak another tenant's fleet.
        """
        state = _state(request)
        principal = state.auth.authenticate(authorization)
        names = state.auth.visible_models(principal, state.served_models())
        return ModelList(data=[ModelCard(id=name) for name in names])

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Liveness: the process is up and serving HTTP. Never depends on a backend."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        """Readiness: at least one healthy replica for every served model.

        Separate from liveness on purpose -- a gateway whose only engine pod is restarting
        should leave the Service, not be killed and restarted itself.
        """
        state = _state(request)
        report = await state.router.health_report()
        ready = bool(report) and all(any(entry.values()) for entry in report.values())
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "backends": report},
            status_code=200 if ready else _HTTP_SERVICE_UNAVAILABLE,
        )

    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        """Prometheus exposition of this app's registry."""
        body, content_type = _state(request).metrics.render()
        return Response(content=body, media_type=content_type)


def _require_served(state: GatewayState, model: str) -> None:
    """Raise :class:`ModelNotFoundError` when no pool serves ``model``."""
    if not state.router.has_model(model):
        raise ModelNotFoundError(
            f"model {model!r} is not served by this gateway; "
            f"available: {', '.join(state.served_models()) or 'none'}"
        )


def _sse(state: GatewayState, generator: AsyncIterator[str]) -> Response:
    """Wrap a payload generator in an OpenAI-shaped SSE response."""
    return EventSourceResponse(
        generator,
        sep="\n",
        ping=int(state.options.sse_ping_interval_s),
        headers={"Cache-Control": "no-store"},
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

gateway_app = typer.Typer(
    name="gateway",
    help="Run and inspect the multi-tenant OpenAI-compatible gateway.",
    no_args_is_help=True,
)


@gateway_app.command("serve")
def serve_command(
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            help=(
                "'config' to build the pools from --models, 'mock' for an in-process "
                "synthetic backend, or the base URL of an OpenAI-compatible server such as "
                "http://127.0.0.1:8000/v1."
            ),
        ),
    ] = "config",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name to serve; defaults to the configured model."),
    ] = None,
    tenants: Annotated[Path, typer.Option("--tenants", help="Tenant directory YAML.")] = Path(
        "configs/tenants.yaml"
    ),
    models: Annotated[
        Path, typer.Option("--models", help="Model pool YAML, used when --engine is 'config'.")
    ] = Path("configs/models.yaml"),
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    require_auth: Annotated[
        bool, typer.Option("--require-auth/--no-require-auth", help="Demand a bearer key.")
    ] = True,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Serve the gateway over HTTP."""
    import uvicorn

    overrides: dict[str, Any] = {"model": model} if model else {}
    settings = Settings(tenants_file=tenants, models_file=models, **overrides)
    router = router_for_engine(engine, settings, models)
    app = create_app(
        settings,
        router,
        options=GatewayOptions(require_auth=require_auth, include_process_metrics=True),
    )
    typer.echo(f"turboserve gateway on http://{host}:{port} serving {', '.join(router.models())}")
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def router_for_engine(engine: str, settings: Settings, models_path: Path) -> Router:
    """Build the router named by ``--engine``.

    ``config`` is the default and reads the model pools from ``--models``; it falls back to
    an in-process mock when that file is absent, so the command always starts something.
    """
    if engine == "config":
        return _default_router(settings, None)
    router = Router()
    if engine == "mock":
        from turboserve.gateway.backends.mock import MockBackend

        router.add_backend(settings.model, MockBackend(name="mock", models=[settings.model]))
        return router
    if engine.startswith(("http://", "https://")):
        from turboserve.gateway.backends.openai_compat import OpenAICompatBackend

        router.add_backend(settings.model, OpenAICompatBackend(engine, name="upstream"))
        return router
    raise typer.BadParameter(
        f"--engine must be 'config', 'mock' or an http(s) base URL, got {engine!r}",
        param_hint="--engine",
    )


@gateway_app.command("hash-key")
def hash_key_command(
    key: Annotated[str, typer.Argument(help="The API key to hash.")],
) -> None:
    """Print the SHA-256 digest to paste into ``configs/tenants.yaml``."""
    typer.echo(hash_api_key(key))


@gateway_app.command("config-check")
def config_check_command(
    tenants: Annotated[Path, typer.Option("--tenants")] = Path("configs/tenants.yaml"),
    models: Annotated[Path, typer.Option("--models")] = Path("configs/models.yaml"),
) -> None:
    """Validate the tenant and model configuration without starting a server."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    try:
        registry = TenantRegistry.from_yaml(tenants)
    except TenantConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="--tenants") from exc
    try:
        models_file = ModelsFile.from_yaml(models)
    except RouterConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="--models") from exc

    tenant_table = Table(title=f"tenants ({tenants})")
    for column in ("id", "keys", "rpm", "tpm", "concurrency", "models", "adapters"):
        tenant_table.add_column(column)
    for tenant in registry:
        tenant_table.add_row(
            tenant.tenant_id,
            str(len(tenant.keys_sha256)),
            "-" if tenant.rpm is None else str(tenant.rpm),
            "-" if tenant.tpm is None else str(tenant.tpm),
            "-" if tenant.max_concurrency is None else str(tenant.max_concurrency),
            ", ".join(tenant.model_patterns),
            ", ".join(sorted(tenant.adapters)) or "-",
        )
    console.print(tenant_table)

    example_tenants = registry.tenants_with_example_keys()
    if example_tenants:
        console.print(
            f"[yellow]warning[/yellow]: {', '.join(example_tenants)} still accept the "
            "example API keys shipped with this repository, whose plaintext is public. "
            "Replace those digests before serving real traffic."
        )

    model_table = Table(title=f"model pools ({models})")
    for column in ("model", "backend", "type", "lane", "weight", "priced"):
        model_table.add_column(column)
    for pool in models_file.models:
        for backend in pool.backends:
            model_table.add_row(
                pool.name,
                backend.name,
                backend.backend,
                backend.lane,
                f"{backend.weight:g}",
                "yes" if pool.price is not None else "no",
            )
    console.print(model_table)
