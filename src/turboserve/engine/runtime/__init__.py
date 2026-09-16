"""Runtime layer: model runner, synchronous and async engines, baselines, streaming.

This package is the seam between the pure scheduling logic of
:mod:`turboserve.engine.core` and the tensor code of :mod:`turboserve.engine.model`. It
exports four things worth importing from elsewhere:

:class:`~turboserve.engine.runtime.engine.LLMEngine`
    The continuous-batching engine. ``add_request`` / ``step`` / ``abort`` / ``stats``.
:class:`~turboserve.engine.runtime.async_engine.AsyncLLMEngine`
    The same engine driven by a background task, with one ``asyncio.Queue`` per request.
:class:`~turboserve.engine.runtime.naive.NaiveHFEngine` and
:class:`~turboserve.engine.runtime.naive.StaticBatchHFEngine`
    ``transformers``-based baselines with the same interface, so the benchmark can drive all
    three identically.
:data:`engine_app`
    The ``turboserve engine`` CLI sub-app, with a ``generate`` smoke command.

Imports are resolved lazily through :pep:`562` module ``__getattr__``. Touching any name
here pulls in torch and the model code, and the gateway's mock backend, the canary
controller and ``turboserve hwinfo`` all import parts of this package's parent without
needing any of that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from turboserve.engine.runtime.async_engine import AsyncEngineDeadError, AsyncLLMEngine
    from turboserve.engine.runtime.engine import DecodeStepHook, EngineStats, LLMEngine
    from turboserve.engine.runtime.memory import (
        KVCacheSizing,
        MemoryProbe,
        MemoryProfileError,
        build_kv_cache,
        probe_memory,
        size_kv_cache,
    )
    from turboserve.engine.runtime.naive import (
        BaselineEngine,
        NaiveHFEngine,
        StaticBatchHFEngine,
    )
    from turboserve.engine.runtime.streaming import (
        DecodedDelta,
        IncrementalDetokenizer,
        StopStringMatcher,
        StreamingDecoder,
        get_tokenizer,
    )
    from turboserve.engine.runtime.worker import (
        BatchTensors,
        LoRAContextBuilder,
        ModelRunner,
        StepOutput,
    )

__all__ = [
    "AsyncEngineDeadError",
    "AsyncLLMEngine",
    "BaselineEngine",
    "BatchTensors",
    "DecodeStepHook",
    "DecodedDelta",
    "EngineStats",
    "IncrementalDetokenizer",
    "KVCacheSizing",
    "LLMEngine",
    "LoRAContextBuilder",
    "MemoryProbe",
    "MemoryProfileError",
    "ModelRunner",
    "NaiveHFEngine",
    "StaticBatchHFEngine",
    "StepOutput",
    "StopStringMatcher",
    "StreamingDecoder",
    "build_kv_cache",
    "engine_app",
    "get_tokenizer",
    "probe_memory",
    "size_kv_cache",
]

_MODULES: dict[str, str] = {
    "AsyncEngineDeadError": "async_engine",
    "AsyncLLMEngine": "async_engine",
    "BaselineEngine": "naive",
    "BatchTensors": "worker",
    "DecodeStepHook": "engine",
    "DecodedDelta": "streaming",
    "EngineStats": "engine",
    "IncrementalDetokenizer": "streaming",
    "KVCacheSizing": "memory",
    "LLMEngine": "engine",
    "LoRAContextBuilder": "worker",
    "MemoryProbe": "memory",
    "MemoryProfileError": "memory",
    "ModelRunner": "worker",
    "NaiveHFEngine": "naive",
    "StaticBatchHFEngine": "naive",
    "StepOutput": "worker",
    "StopStringMatcher": "streaming",
    "StreamingDecoder": "streaming",
    "build_kv_cache": "memory",
    "engine_app": "engine",
    "get_tokenizer": "streaming",
    "probe_memory": "memory",
    "size_kv_cache": "memory",
}


def __getattr__(name: str) -> Any:
    """Import the module that owns ``name`` on first access (:pep:`562`)."""
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
