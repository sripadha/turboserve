"""The two HTTP-free backends: the synthetic mock and the OpenAI-compatible client.

The OpenAI-compatible backend is exercised against an ``httpx.MockTransport`` that replays
*synthetic* server transcripts written here by hand -- no network, no server, and nothing
recorded from a real deployment.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from turboserve.engine.core.types import FinishReason, SamplingParams
from turboserve.gateway.backends import get_backend_cls
from turboserve.gateway.backends.mock import MockBackend, MockConfig
from turboserve.gateway.backends.openai_compat import OpenAICompatBackend
from turboserve.gateway.backends.protocol import (
    AdapterNotFoundError,
    Backend,
    BackendOverloadedError,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
    TokenEvent,
)


def make_request(**overrides: object) -> GenerateRequest:
    """A minimal valid generate request."""
    data: dict[str, object] = {
        "request_id": "req-1",
        "tenant_id": "acme",
        "model": "m",
        "prompt": "hello there",
    }
    data.update(overrides)
    return GenerateRequest.model_validate(data)


async def drain(backend: Backend, req: GenerateRequest) -> list[TokenEvent]:
    """Collect every event a backend yields for one request."""
    return [event async for event in backend.generate(req)]


# -- MockBackend ------------------------------------------------------------------------


def test_mock_is_registered_under_its_name() -> None:
    assert get_backend_cls("mock") is MockBackend


async def test_mock_streams_and_terminates_with_usage() -> None:
    backend = MockBackend(models=["m"], max_tokens=8)
    events = await drain(backend, make_request(sampling=SamplingParams(max_tokens=5)))
    assert len(events) == 5
    assert all(not event.finished for event in events[:-1])
    last = events[-1]
    assert last.finished and last.finish_reason is FinishReason.LENGTH
    assert last.usage == {
        "prompt_tokens": 2,
        "completion_tokens": 5,
        "total_tokens": 7,
        "cached_prompt_tokens": 0,
    }
    assert backend.stats.completed == 1
    assert backend.stats.in_flight == 0


async def test_mock_output_is_reproducible_per_request_id() -> None:
    # Reproducibility is what lets a chaos run be replayed and a flake be distinguished
    # from a regression.
    a = MockBackend(models=["m"], seed=7)
    b = MockBackend(models=["m"], seed=7)
    left = await drain(a, make_request(request_id="same"))
    right = await drain(b, make_request(request_id="same"))
    assert [event.token_ids for event in left] == [event.token_ids for event in right]
    other = await drain(a, make_request(request_id="different"))
    assert [event.text for event in other] != [event.text for event in left]


async def test_mock_respects_the_configured_token_ceiling() -> None:
    backend = MockBackend(models=["m"], max_tokens=3)
    events = await drain(backend, make_request(sampling=SamplingParams(max_tokens=1000)))
    assert sum(event.num_tokens for event in events) == 3


async def test_mock_batches_tokens_per_event() -> None:
    backend = MockBackend(models=["m"], max_tokens=6, tokens_per_event=3)
    events = await drain(backend, make_request(sampling=SamplingParams(max_tokens=6)))
    assert [event.num_tokens for event in events] == [3, 3]


async def test_mock_pre_first_token_failure_is_retryable() -> None:
    backend = MockBackend(models=["m"], error_probability=1.0)
    with pytest.raises(BackendUnavailableError) as exc:
        await drain(backend, make_request())
    assert exc.value.retryable is True
    assert backend.stats.failed == 1


async def test_mock_mid_stream_drop_arrives_as_an_in_band_failure() -> None:
    # After the first token a failure cannot be an HTTP status any more, so the contract is
    # a terminating error event rather than an exception.
    backend = MockBackend(models=["m"], max_tokens=8, drop_probability=1.0, seed=3)
    events = await drain(backend, make_request(sampling=SamplingParams(max_tokens=8)))
    assert events[-1].finished and events[-1].is_error
    assert events[-1].finish_reason is FinishReason.ABORT
    assert backend.stats.dropped == 1


async def test_mock_drop_rate_matches_the_configured_probability() -> None:
    # The cut point is drawn once per request rather than per token, so the observed rate is
    # the configured one and does not depend on how long the completions are.
    runs = 1200
    backend = MockBackend(models=["m"], max_tokens=4, drop_probability=0.25, seed=11)
    dropped = 0
    for index in range(runs):
        events = await drain(backend, make_request(request_id=f"req-{index}"))
        dropped += events[-1].is_error
    assert 0.22 <= dropped / runs <= 0.28


async def test_mock_refuses_unknown_models_and_adapters() -> None:
    backend = MockBackend(models=["m"], adapters=["support"])
    with pytest.raises(ModelNotFoundError):
        await drain(backend, make_request(model="other"))
    with pytest.raises(AdapterNotFoundError):
        await drain(backend, make_request(lora="nope"))
    assert await drain(backend, make_request(lora="support"))


async def test_mock_accept_any_model_disables_the_check() -> None:
    backend = MockBackend(models=["m"], accept_any_model=True)
    assert await drain(backend, make_request(model="whatever"))


async def test_mock_health_is_flippable_and_never_raises() -> None:
    backend = MockBackend(models=["m"])
    assert await backend.health() is True
    backend.set_healthy(False)
    assert await backend.health() is False
    backend.set_healthy(True)
    await backend.close()
    assert await backend.health() is False  # a closed backend is never healthy


async def test_mock_close_is_idempotent_and_refuses_later_work() -> None:
    backend = MockBackend(models=["m"])
    await backend.close()
    await backend.close()
    with pytest.raises(BackendUnavailableError):
        await drain(backend, make_request())


async def test_mock_latency_configuration_is_honoured() -> None:
    backend = MockBackend(models=["m"], max_tokens=3, ttft_ms=20.0, itl_ms=10.0)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await drain(backend, make_request(sampling=SamplingParams(max_tokens=3)))
    # 20 ms before the first token plus 10 ms before each of the other two.
    assert loop.time() - start >= 0.04


async def test_mock_cancellation_is_counted_and_releases_the_slot() -> None:
    backend = MockBackend(models=["m"], max_tokens=50, itl_ms=50.0)
    stream = backend.generate(make_request(sampling=SamplingParams(max_tokens=50)))
    await anext(stream)
    await stream.aclose()
    assert backend.stats.in_flight == 0


def test_mock_config_is_mutable_so_faults_can_be_injected_live() -> None:
    backend = MockBackend(MockConfig(models=["m"]))
    backend.config.error_probability = 0.5
    backend.config.itl_ms = 12.0
    assert backend.config.error_probability == 0.5
    with pytest.raises(ValueError, match="less than or equal to 1"):
        backend.config.error_probability = 2.0


def test_mock_config_is_copied_not_shared() -> None:
    config = MockConfig(models=["m"])
    backend = MockBackend(config)
    backend.config.itl_ms = 5.0
    assert config.itl_ms == 0.0


# -- OpenAICompatBackend ----------------------------------------------------------------


def sse(*chunks: dict[str, object], done: bool = True) -> bytes:
    """Render a synthetic OpenAI-compatible SSE body."""
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


def text_chunk(text: str, finish_reason: str | None = None) -> dict[str, object]:
    """One synthetic completions chunk."""
    return {"choices": [{"index": 0, "text": text, "finish_reason": finish_reason}]}


def backend_with(handler: object, **kwargs: object) -> OpenAICompatBackend:
    """An OpenAI-compatible backend wired to a mock transport."""
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        base_url="http://upstream/v1",
    )
    return OpenAICompatBackend("http://upstream/v1", client=client, name="upstream", **kwargs)  # type: ignore[arg-type]


async def test_openai_compat_streams_text_and_final_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200,
            content=sse(
                text_chunk("Hello"),
                text_chunk(" world"),
                text_chunk("", "length"),
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 2,
                        "total_tokens": 13,
                        "prompt_tokens_details": {"cached_tokens": 8},
                    },
                },
            ),
            headers={"content-type": "text/event-stream"},
        )

    backend = backend_with(handler)
    events = await drain(backend, make_request())
    assert [event.text for event in events] == ["Hello", " world", ""]
    last = events[-1]
    assert last.finished and last.finish_reason is FinishReason.LENGTH
    assert last.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "total_tokens": 13,
        "cached_prompt_tokens": 8,
    }
    await backend.close()


async def test_openai_compat_reads_the_chat_delta_shape_too() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ),
            headers={"content-type": "text/event-stream"},
        )

    backend = backend_with(handler)
    events = await drain(backend, make_request())
    assert "".join(event.text for event in events) == "hi"
    assert events[-1].finish_reason is FinishReason.STOP
    await backend.close()


async def test_openai_compat_sends_token_ids_and_extra_sampling_knobs() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse(text_chunk("x", "stop")))

    backend = backend_with(handler, extra_body={"min_tokens": 1})
    sampling = SamplingParams(
        max_tokens=9, top_k=40, repetition_penalty=1.1, ignore_eos=True, seed=5, stop=["\n"]
    )
    await drain(backend, make_request(prompt=[1, 2, 3], sampling=sampling, priority=4))
    assert captured["prompt"] == [1, 2, 3]
    assert captured["top_k"] == 40
    assert captured["repetition_penalty"] == pytest.approx(1.1)
    assert captured["ignore_eos"] is True
    assert captured["seed"] == 5
    assert captured["stop"] == ["\n"]
    assert captured["priority"] == 4
    assert captured["min_tokens"] == 1
    await backend.close()


async def test_openai_compat_omits_neutral_options() -> None:
    # A stricter upstream rejects unknown fields; neutral values must not be sent at all.
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse(text_chunk("x", "stop")))

    backend = backend_with(handler)
    await drain(backend, make_request())
    assert set(captured) == {
        "model",
        "stream",
        "stream_options",
        "max_tokens",
        "temperature",
        "top_p",
        "prompt",
    }
    await backend.close()


async def test_openai_compat_addresses_a_lora_adapter_through_the_model_field() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse(text_chunk("x", "stop")))

    backend = backend_with(handler, adapter_models={"support": "acme-support-r16"})
    await drain(backend, make_request(lora="support"))
    assert captured["model"] == "acme-support-r16"
    await backend.close()


async def test_openai_compat_maps_model_aliases() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse(text_chunk("x", "stop")))

    backend = backend_with(handler, model_map={"m": "upstream-name"})
    await drain(backend, make_request())
    assert captured["model"] == "upstream-name"
    await backend.close()


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [
        (429, BackendOverloadedError, True),
        (503, BackendOverloadedError, True),
        (504, BackendTimeoutError, True),
        (404, ModelNotFoundError, False),
        (400, BackendRequestError, False),
        (500, BackendUnavailableError, True),
    ],
)
async def test_openai_compat_maps_status_codes_to_error_classes(
    status: int, expected: type[Exception], retryable: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "nope"})

    backend = backend_with(handler)
    with pytest.raises(expected) as exc:
        await drain(backend, make_request())
    assert exc.value.retryable is retryable
    await backend.close()


async def test_openai_compat_transport_failure_before_the_first_token_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = backend_with(handler)
    with pytest.raises(BackendUnavailableError) as exc:
        await drain(backend, make_request())
    assert exc.value.retryable is True
    await backend.close()


async def test_openai_compat_timeout_is_its_own_class() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    backend = backend_with(handler)
    with pytest.raises(BackendTimeoutError):
        await drain(backend, make_request())
    await backend.close()


async def test_openai_compat_discards_a_malformed_chunk_rather_than_dying() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = b"data: {not json}\n\n" + sse(text_chunk("ok", "stop"))
        return httpx.Response(200, content=payload)

    backend = backend_with(handler)
    events = await drain(backend, make_request())
    assert "".join(event.text for event in events) == "ok"
    await backend.close()


async def test_openai_compat_lists_models_and_reports_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}, {}]})
        return httpx.Response(404)

    backend = backend_with(handler)
    assert await backend.models() == ["a", "b"]
    assert await backend.health() is True
    await backend.close()


async def test_openai_compat_health_falls_back_to_the_model_list() -> None:
    # Not every OpenAI-compatible server exposes /health.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "a"}]})

    backend = backend_with(handler)
    assert await backend.health() is True
    await backend.close()


async def test_openai_compat_health_is_false_and_silent_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    backend = backend_with(handler)
    assert await backend.health() is False
    await backend.close()
    assert await backend.health() is False


def test_openai_compat_requires_a_base_url() -> None:
    with pytest.raises(ValueError, match="base_url is required"):
        OpenAICompatBackend("")


# -- recognising an SGLang server --------------------------------------------------------
#
# Nothing on the request path distinguishes the two production engines: the transcripts
# above are the same bytes whichever of them sent them, which is why adding SGLang as a
# second backend needed no change to `generate`. What differs is what a server will say
# about itself, and that is what these tests pin.


async def test_openai_compat_reads_an_sglang_servers_version_and_settings() -> None:
    """The two native endpoints, at the server root rather than under /v1."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.5.3"})
        if request.url.path == "/get_server_info":
            return httpx.Response(
                200,
                json={
                    "server_args": {
                        "model_path": "Qwen/Qwen2.5-7B-Instruct",
                        "dtype": "bfloat16",
                        "disable_radix_cache": False,
                        "max_loras_per_batch": 8,
                        # Neither recorded: one is not in the whitelist, the other is not a
                        # scalar, and a result file is not a copy of an argument parser.
                        "log_level": "info",
                        "lora_paths": ["acme=/adapters/acme"],
                    },
                    "context_length": 8192,
                },
            )
        return httpx.Response(404)

    backend = backend_with(handler)
    info = await backend.server_info()
    assert seen == ["/version", "/get_server_info"]
    assert info["version"] == "0.5.3"
    assert info["settings"] == {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": "bfloat16",
        "disable_radix_cache": False,
        "max_loras_per_batch": 8,
        "context_length": 8192,
    }
    await backend.close()


async def test_openai_compat_falls_back_to_get_server_info_for_the_version() -> None:
    """SGLang reports its version inside `/get_server_info`, and not every build routes
    `/version`. A server whose settings are recorded without a version is a server whose
    settings cannot be looked up, so the document's own field is the fallback."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_server_info":
            return httpx.Response(
                200,
                json={
                    "version": "0.5.3",
                    "model_path": "Qwen/Qwen2.5-7B-Instruct",
                    "disable_radix_cache": True,
                },
            )
        return httpx.Response(404, json={"detail": "Not Found"})

    backend = backend_with(handler)
    info = await backend.server_info()
    assert info == {
        "version": "0.5.3",
        "settings": {
            "model_path": "Qwen/Qwen2.5-7B-Instruct",
            "disable_radix_cache": True,
        },
    }
    await backend.close()


async def test_openai_compat_server_info_keeps_only_the_version_from_a_vllm_server() -> None:
    """vLLM answers /version and has no /get_server_info; a 404 there is not an error."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        return httpx.Response(404, json={"detail": "Not Found"})

    backend = backend_with(handler)
    assert await backend.server_info() == {"version": "0.11.0"}
    await backend.close()


async def test_openai_compat_server_info_is_empty_and_silent_when_nothing_answers() -> None:
    """A proxy, an older build or an unreachable server contributes no block at all."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/version":
            return httpx.Response(200, content=b"not json")
        raise httpx.ConnectError("down")

    backend = backend_with(handler)
    assert await backend.server_info() == {}
    await backend.close()
    assert await backend.server_info() == {}
