"""Unit tests for the gateway backend contract and the backend registry.

The fake backend below is a synthetic stand-in for a real one: it emits a fixed, made-up
token stream so that the shape of the contract (streaming, terminator, cancellation,
error classification) can be tested without a model.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from turboserve.engine.core.types import FinishReason, SamplingParams
from turboserve.gateway import backends
from turboserve.gateway.backends import (
    BACKENDS,
    AdapterNotFoundError,
    Backend,
    BackendError,
    BackendOverloadedError,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
    NonRetryableBackendError,
    RetryableBackendError,
    StreamInterruptedError,
    TokenEvent,
    available_backends,
    get_backend_cls,
    load_builtin_backends,
    register_backend,
    unregister_backend,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class FakeBackend:
    """Synthetic backend: yields two deltas and a terminator, never touching a model."""

    name = "fake"
    supports_lora = True

    def __init__(self) -> None:
        self.closed = False
        self.cancelled = False

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        try:
            yield TokenEvent.delta(req.request_id, [11], "he")
            yield TokenEvent.delta(req.request_id, [12], "llo")
            yield TokenEvent.final(
                req.request_id,
                FinishReason.STOP,
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            )
        except GeneratorExit:
            self.cancelled = True
            raise

    async def health(self) -> bool:
        return True

    async def models(self) -> list[str]:
        return ["fake-model"]

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Snapshot and restore the module-global registry around a test."""
    saved = dict(BACKENDS)
    saved_loaded = backends._builtins_loaded
    BACKENDS.clear()
    backends._builtins_loaded = False
    try:
        yield
    finally:
        BACKENDS.clear()
        BACKENDS.update(saved)
        backends._builtins_loaded = saved_loaded


def _request(**overrides: Any) -> GenerateRequest:
    payload: dict[str, Any] = {
        "request_id": "req-1",
        "tenant_id": "acme",
        "model": "fake-model",
        "prompt": "hello",
    }
    payload.update(overrides)
    return GenerateRequest(**payload)


# --------------------------------------------------------------------------------------
# GenerateRequest
# --------------------------------------------------------------------------------------


def test_generate_request_defaults() -> None:
    req = _request()
    assert req.sampling == SamplingParams()
    assert req.lora is None
    assert req.priority == 0
    assert req.stream is True
    assert isinstance(req.arrival_ts, float)


def test_generate_request_text_prompt_helpers() -> None:
    req = _request(prompt="hi there")
    assert not req.prompt_is_tokens
    assert req.prompt_text == "hi there"
    assert req.prompt_token_ids is None


def test_generate_request_token_prompt_helpers() -> None:
    req = _request(prompt=[1, 2, 3])
    assert req.prompt_is_tokens
    assert req.prompt_token_ids == [1, 2, 3]
    assert req.prompt_text is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": ""},
        {"tenant_id": ""},
        {"model": ""},
        {"prompt": ""},
        {"prompt": []},
        {"unexpected": 1},
    ],
)
def test_generate_request_rejects_invalid(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _request(**overrides)


def test_generate_request_carries_sampling_validation() -> None:
    with pytest.raises(ValidationError):
        _request(sampling={"temperature": -1.0})


def test_generate_request_accepts_a_sampling_dict() -> None:
    req = _request(sampling={"max_tokens": 7, "temperature": 0.0})
    assert req.sampling.max_tokens == 7
    assert req.sampling.is_greedy


# --------------------------------------------------------------------------------------
# TokenEvent
# --------------------------------------------------------------------------------------


def test_token_event_delta_shape() -> None:
    before = time.monotonic_ns()
    event = TokenEvent.delta("r", [5, 6], "ab")
    assert event.request_id == "r"
    assert event.token_ids == [5, 6]
    assert event.num_tokens == 2
    assert event.text == "ab"
    assert not event.finished
    assert not event.is_error
    assert event.t_ns >= before


def test_token_event_final_carries_usage_and_reason() -> None:
    event = TokenEvent.final("r", FinishReason.LENGTH, usage={"total_tokens": 9}, t_ns=123)
    assert event.finished
    assert event.finish_reason is FinishReason.LENGTH
    assert event.usage == {"total_tokens": 9}
    assert event.t_ns == 123
    assert not event.is_error


def test_token_event_final_may_carry_the_last_delta() -> None:
    event = TokenEvent.final("r", FinishReason.STOP, token_ids=[3], text="!")
    assert event.token_ids == [3]
    assert event.text == "!"
    assert event.finished


def test_token_event_failure_is_a_finished_error() -> None:
    event = TokenEvent.failure("r", "connection reset")
    assert event.is_error
    assert event.error == "connection reset"
    assert event.finished
    assert event.token_ids == []


def test_token_event_defaults_are_not_shared() -> None:
    first = TokenEvent(request_id="a")
    first.token_ids.append(1)
    assert TokenEvent(request_id="b").token_ids == []


# --------------------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------------------


def test_fake_backend_satisfies_the_protocol() -> None:
    assert isinstance(FakeBackend(), Backend)


def test_incomplete_backend_does_not_satisfy_the_protocol() -> None:
    class Missing:
        name = "x"
        supports_lora = False

        async def health(self) -> bool:
            return True

    assert not isinstance(Missing(), Backend)


async def test_backend_streams_deltas_then_a_terminator() -> None:
    backend = FakeBackend()
    events = [event async for event in backend.generate(_request())]
    assert [event.text for event in events] == ["he", "llo", ""]
    assert [event.finished for event in events] == [False, False, True]
    assert events[-1].finish_reason is FinishReason.STOP
    assert events[-1].usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert [event.t_ns for event in events] == sorted(event.t_ns for event in events)


async def test_cancelling_the_stream_reaches_the_backend() -> None:
    backend = FakeBackend()
    stream = backend.generate(_request())
    assert (await anext(stream)).text == "he"
    await stream.aclose()
    assert backend.cancelled


async def test_backend_health_models_and_close() -> None:
    backend = FakeBackend()
    assert await backend.health() is True
    assert await backend.models() == ["fake-model"]
    await backend.close()
    assert backend.closed


# --------------------------------------------------------------------------------------
# Error hierarchy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_cls",
    [BackendUnavailableError, BackendTimeoutError, BackendOverloadedError],
)
def test_retryable_errors_are_marked_retryable(error_cls: type[BackendError]) -> None:
    error = error_cls("boom")
    assert error.retryable
    assert isinstance(error, RetryableBackendError)
    assert isinstance(error, BackendError)


@pytest.mark.parametrize(
    "error_cls",
    [BackendRequestError, ModelNotFoundError, AdapterNotFoundError, StreamInterruptedError],
)
def test_non_retryable_errors_are_marked_non_retryable(error_cls: type[BackendError]) -> None:
    error = error_cls("boom")
    assert not error.retryable
    assert isinstance(error, NonRetryableBackendError)
    assert isinstance(error, BackendError)


def test_backend_error_message_includes_backend_and_status() -> None:
    error = BackendOverloadedError("queue full", backend="vllm-a", status_code=429)
    assert error.message == "queue full"
    assert error.backend == "vllm-a"
    assert error.status_code == 429
    assert str(error) == "queue full [backend=vllm-a] (status 429)"


def test_backend_error_message_without_context() -> None:
    assert str(BackendError("plain")) == "plain"


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


@pytest.mark.usefixtures("clean_registry")
def test_register_and_look_up_a_backend() -> None:
    register_backend("fake")(FakeBackend)
    assert available_backends() == ["fake"]
    assert get_backend_cls("fake") is FakeBackend


@pytest.mark.usefixtures("clean_registry")
def test_registering_the_same_class_twice_is_allowed() -> None:
    register_backend("fake")(FakeBackend)
    register_backend("fake")(FakeBackend)
    assert BACKENDS["fake"] is FakeBackend


@pytest.mark.usefixtures("clean_registry")
def test_duplicate_name_with_a_different_class_raises() -> None:
    class Other(FakeBackend):
        pass

    register_backend("fake")(FakeBackend)
    with pytest.raises(ValueError, match="already registered"):
        register_backend("fake")(Other)


@pytest.mark.usefixtures("clean_registry")
def test_override_replaces_the_registration() -> None:
    class Other(FakeBackend):
        pass

    register_backend("fake")(FakeBackend)
    register_backend("fake", override=True)(Other)
    assert get_backend_cls("fake") is Other


@pytest.mark.usefixtures("clean_registry")
def test_registering_a_class_missing_contract_methods_raises() -> None:
    class Broken:
        name = "broken"
        supports_lora = False

    with pytest.raises(TypeError, match="missing generate, health, models, close"):
        register_backend("broken")(Broken)


def test_empty_backend_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        register_backend("")


@pytest.mark.usefixtures("clean_registry")
def test_unknown_backend_reports_the_known_names() -> None:
    register_backend("fake")(FakeBackend)
    with pytest.raises(KeyError, match="unknown backend 'nope'"):
        get_backend_cls("nope")


@pytest.mark.usefixtures("clean_registry")
def test_unregister_removes_a_backend_and_is_idempotent() -> None:
    register_backend("fake")(FakeBackend)
    unregister_backend("fake")
    unregister_backend("fake")
    assert available_backends() == []


@pytest.mark.usefixtures("clean_registry")
def test_load_builtin_backends_skips_absent_modules_and_is_cached() -> None:
    register_backend("fake")(FakeBackend)
    assert load_builtin_backends() == ["fake"]
    # Second call short-circuits on the cached flag and still reports the registry.
    assert load_builtin_backends() == ["fake"]
