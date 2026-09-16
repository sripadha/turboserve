"""Tests for the in-process gateway backend.

The backend's job is translation, so the tests are about translation: model names, adapter
names, the ``TokenEvent`` shapes the gateway's SSE writer depends on, the retry rule encoded
in which failures are raised and which are yielded, and lazy construction (a backend built
from ``configs/models.yaml`` must not load a checkpoint until someone sends a request).

Only the last two tests build a real engine, on the cached tiny checkpoint.
"""

from __future__ import annotations

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
from turboserve.gateway.backends import Backend, get_backend_cls
from turboserve.gateway.backends.local_engine import LocalEngineBackend
from turboserve.gateway.backends.protocol import (
    AdapterNotFoundError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
)


class FakeAsyncEngine:
    """Stands in for :class:`AsyncLLMEngine`: records calls, replays scripted deltas."""

    def __init__(self, outputs: list[RequestOutput] | None = None) -> None:
        self.engine = type("_Inner", (), {"config": EngineConfig(model="fake/model")})()
        self.outputs = outputs if outputs is not None else []
        self.started = 0
        self.closed = 0
        self.calls: list[dict[str, Any]] = []
        self.is_healthy = True
        self.raise_after: int | None = None

    @property
    def is_running(self) -> bool:
        return self.started > 0 and self.closed == 0

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    async def generate(self, request_id: str, prompt: Any, sampling: Any = None, **kw: Any):
        self.calls.append({"request_id": request_id, "prompt": prompt, "sampling": sampling, **kw})
        for index, output in enumerate(self.outputs):
            if self.raise_after is not None and index == self.raise_after:
                raise RuntimeError("upstream exploded")
            yield output

    def stats(self) -> dict[str, int | float | str]:
        return {"num_running": 0}


def _request(**overrides: Any) -> GenerateRequest:
    data: dict[str, Any] = {
        "request_id": "req-1",
        "tenant_id": "acme",
        "model": "fake/model",
        "prompt": "hello",
        "sampling": SamplingParams(max_tokens=4, temperature=0.0),
    }
    data.update(overrides)
    return GenerateRequest(**data)


def _delta(text: str, token: int) -> RequestOutput:
    return RequestOutput(request_id="req-1", new_token_ids=[token], text_delta=text)


def _final(text: str, token: int, *, output_tokens: int) -> RequestOutput:
    return RequestOutput(
        request_id="req-1",
        new_token_ids=[token],
        text_delta=text,
        finished=True,
        finish_reason=FinishReason.STOP,
        prompt_tokens=3,
        output_tokens=output_tokens,
        cached_prompt_tokens=2,
    )


# ----------------------------------------------------------------------------------------
# registration and construction
# ----------------------------------------------------------------------------------------


def test_registered_under_the_name_local() -> None:
    """``configs/models.yaml`` says ``backend: local``; the registry must resolve it."""
    assert get_backend_cls("local") is LocalEngineBackend


def test_satisfies_the_backend_protocol() -> None:
    """Structural check, so a renamed method is caught here rather than in production."""
    backend = LocalEngineBackend(model="fake/model")
    assert isinstance(backend, Backend)


def test_construction_does_not_load_anything() -> None:
    """A backend built from config must be cheap: no weights until the first request."""
    backend = LocalEngineBackend(name="ref", model="fake/model", adapters={"legal": 3})
    assert backend.is_loaded is False
    assert backend.name == "ref"
    assert backend.config.model == "fake/model"
    assert backend.supports_lora is True


def test_options_from_the_models_file_shape() -> None:
    """The exact call ``build_router`` makes: ``cls(name=..., **options)``."""
    backend = LocalEngineBackend(
        name="reference",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        served_models=["Qwen/Qwen2.5-0.5B-Instruct", "default"],
        local_files_only=True,
    )
    assert backend.name == "reference"
    assert backend.config.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert backend.supports_lora is False


def test_config_mapping_is_validated() -> None:
    """``options.config`` is an :class:`EngineConfig` body, and a bad one fails loudly."""
    backend = LocalEngineBackend(
        config={"model": "x/y", "device": "cpu", "scheduler": {"block_size": 32}}
    )
    assert backend.config.block_size == 32
    with pytest.raises(ValueError):
        LocalEngineBackend(config={"model": "x/y", "device": "not-a-device"})
    with pytest.raises(ValueError, match="needs a model"):
        LocalEngineBackend()


def test_explicit_model_overrides_the_config() -> None:
    """``model`` next to a ``config`` block wins, so one config can serve several pools."""
    backend = LocalEngineBackend(
        model="override/model", config=EngineConfig(model="base/model", device="cpu")
    )
    assert backend.config.model == "override/model"
    assert backend.config.device == "cpu"


# ----------------------------------------------------------------------------------------
# translation
# ----------------------------------------------------------------------------------------


def test_adapter_names_map_to_engine_slots() -> None:
    """A configured adapter becomes a slot; an unknown one is a non-retryable 403."""
    backend = LocalEngineBackend(model="fake/model", adapters={"legal": 3, "support": 7})
    assert backend.adapter_slot(None) == 0
    assert backend.adapter_slot("legal") == 3
    with pytest.raises(AdapterNotFoundError) as excinfo:
        backend.adapter_slot("marketing")
    assert excinfo.value.retryable is False
    assert excinfo.value.status_code == 403
    assert "legal, support" in str(excinfo.value)


async def test_unknown_model_is_refused_before_anything_loads() -> None:
    """A model this backend does not serve is a 404, not a silent redirect."""
    backend = LocalEngineBackend(model="fake/model")
    with pytest.raises(ModelNotFoundError) as excinfo:
        async for _ in backend.generate(_request(model="other/model")):
            pass
    assert excinfo.value.retryable is False
    assert backend.is_loaded is False


async def test_streams_deltas_then_a_terminator_with_usage() -> None:
    """Deltas carry tokens and text; the terminator carries the finish reason and usage."""
    fake = FakeAsyncEngine([_delta("he", 1), _delta("llo", 2), _final("!", 3, output_tokens=3)])
    backend = LocalEngineBackend(model="fake/model", engine=fake)  # type: ignore[arg-type]
    events = [event async for event in backend.generate(_request())]
    assert [event.text for event in events] == ["he", "llo", "!"]
    assert [event.token_ids for event in events] == [[1], [2], [3]]
    assert not any(event.finished for event in events[:-1])
    last = events[-1]
    assert last.finished
    assert last.finish_reason is FinishReason.STOP
    assert last.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 3,
        "total_tokens": 6,
        "cached_prompt_tokens": 2,
    }
    assert all(event.t_ns > 0 for event in events)
    await backend.close()


async def test_request_fields_reach_the_engine() -> None:
    """Tenant, priority, sampling and the adapter slot are forwarded, not dropped."""
    fake = FakeAsyncEngine([_final("x", 1, output_tokens=1)])
    backend = LocalEngineBackend(
        model="fake/model",
        engine=fake,
        adapters={"legal": 5},  # type: ignore[arg-type]
    )
    request = _request(tenant_id="acme", priority=4, lora="legal", prompt=[9, 8, 7])
    async for _ in backend.generate(request):
        pass
    call = fake.calls[0]
    assert call["request_id"] == "req-1"
    assert call["prompt"] == [9, 8, 7]
    assert call["tenant_id"] == "acme"
    assert call["priority"] == 4
    assert call["lora_id"] == 5
    assert call["sampling"].max_tokens == 4


async def test_empty_stream_still_terminates() -> None:
    """A stream that ends without a terminator gets a synthesised ``ABORT`` one.

    The protocol requires a last event with ``finished=True``; a gateway that never saw one
    would hold the connection open forever.
    """
    backend = LocalEngineBackend(model="fake/model", engine=FakeAsyncEngine([]))  # type: ignore[arg-type]
    events = [event async for event in backend.generate(_request())]
    assert len(events) == 1
    assert events[0].finished
    assert events[0].finish_reason is FinishReason.ABORT


# ----------------------------------------------------------------------------------------
# failure semantics
# ----------------------------------------------------------------------------------------


async def test_a_failure_before_the_first_token_is_raised() -> None:
    """Nothing has been sent, so the router may still retry another replica: raise."""
    fake = FakeAsyncEngine([_delta("a", 1)])
    fake.raise_after = 0
    backend = LocalEngineBackend(model="fake/model", engine=fake)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="upstream exploded"):
        async for _ in backend.generate(_request()):
            pass


async def test_a_failure_after_the_first_token_becomes_an_event() -> None:
    """Bytes are already on the wire, so the failure travels in-band and is never retried."""
    fake = FakeAsyncEngine([_delta("a", 1), _delta("b", 2)])
    fake.raise_after = 1
    backend = LocalEngineBackend(model="fake/model", engine=fake)  # type: ignore[arg-type]
    events = [event async for event in backend.generate(_request())]
    assert events[0].text == "a"
    assert events[-1].is_error
    assert events[-1].finished
    assert events[-1].finish_reason is FinishReason.ABORT
    assert "upstream exploded" in (events[-1].error or "")


async def test_health_models_and_close() -> None:
    """Health never raises, ``models()`` lists what the pool advertises, close is idempotent."""
    fake = FakeAsyncEngine([_final("x", 1, output_tokens=1)])
    backend = LocalEngineBackend(
        name="ref",
        model="fake/model",
        engine=fake,  # type: ignore[arg-type]
        served_models=["fake/model", "alias"],
    )
    assert await backend.health() is True
    assert await backend.models() == ["fake/model", "alias"]
    assert backend.stats()["backend"] == "ref"
    await backend.close()
    await backend.close()
    assert fake.closed == 1
    assert backend.closed is True
    assert await backend.health() is False
    with pytest.raises(BackendUnavailableError, match="closed"):
        async for _ in backend.generate(_request()):
            pass


async def test_unhealthy_engine_is_reported_unhealthy() -> None:
    """A dead step loop must fail readiness so the router stops sending traffic."""
    fake = FakeAsyncEngine([])
    backend = LocalEngineBackend(model="fake/model", engine=fake)  # type: ignore[arg-type]
    assert await backend.health() is True
    fake.is_healthy = False
    assert await backend.health() is False


async def test_unloaded_backend_reports_healthy_and_cold_stats() -> None:
    """A cold backend is ready: it loads on the first request rather than failing readiness."""
    backend = LocalEngineBackend(name="cold", model="fake/model")
    assert await backend.health() is True
    assert backend.stats() == {"backend": "cold", "model": "fake/model", "loaded": 0}


# ----------------------------------------------------------------------------------------
# with a real engine
# ----------------------------------------------------------------------------------------


def _real_config(model_path: Path) -> EngineConfig:
    return EngineConfig(
        model=str(model_path),
        device="cpu",
        dtype="float32",
        scheduler=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=128, block_size=16, num_blocks=32
        ),
    )


async def test_real_engine_end_to_end(tiny_qwen2_path: Path) -> None:
    """A real engine behind the backend produces well-formed events with correct usage."""
    backend = LocalEngineBackend(
        name="reference",
        config=_real_config(tiny_qwen2_path),
        served_models=[str(tiny_qwen2_path)],
        local_files_only=True,
    )
    try:
        request = _request(
            model=str(tiny_qwen2_path),
            prompt="Hello world",
            sampling=SamplingParams(max_tokens=6, temperature=0.0, ignore_eos=True),
        )
        events = [event async for event in backend.generate(request)]
        assert backend.is_loaded is True
        assert len(events) == 6
        assert sum(event.num_tokens for event in events) == 6
        assert events[-1].finished
        assert events[-1].usage is not None
        assert events[-1].usage["completion_tokens"] == 6
        assert events[-1].usage["prompt_tokens"] > 0
        assert "".join(event.text for event in events) != ""
        assert await backend.health() is True
        assert backend.stats()["loaded"] == 1
    finally:
        await backend.close()


async def test_a_failure_to_load_is_retryable(tmp_path: Path) -> None:
    """A checkpoint that will not load is a retryable backend failure, not a crash."""
    backend = LocalEngineBackend(model=str(tmp_path / "does-not-exist"), local_files_only=True)
    with pytest.raises(BackendUnavailableError) as excinfo:
        async for _ in backend.generate(_request(model=str(tmp_path / "does-not-exist"))):
            pass
    assert excinfo.value.retryable is True
    await backend.close()
