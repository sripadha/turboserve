"""The in-process backend: the gateway talking to :class:`AsyncLLMEngine` directly.

This is the backend that makes ``turboserve serve --engine reference`` a single process.
There is no socket, no serialisation and no second copy of the tokenizer between the HTTP
handler and the scheduler: a token sampled in the step loop reaches the SSE writer through
an ``asyncio.Queue``. That matters for measurement as much as for latency -- the gap between
what the engine produced and what the client saw is the gateway's own overhead, with nothing
else mixed in.

The work of this module is therefore not generation but *translation*, in three places:

* **Construction.** The backend is built from ``configs/models.yaml`` options, long before
  anyone asks it to generate. Loading a multi-gigabyte checkpoint at construction time would
  make ``turboserve gateway config-check`` load a model, so the engine is built on first use,
  once, behind an ``asyncio.Lock``, in a worker thread.
* **Naming.** A tenant asks for a model by the name the gateway advertises; the engine knows
  one model. Anything else is a :class:`ModelNotFoundError`, never a silent redirect to
  whatever happens to be loaded.
* **Adapters.** ``GenerateRequest.lora`` carries an adapter *name*; the engine takes an
  integer slot. The mapping is configuration, supplied in ``options.adapters``.

Failures follow the retry rule the backend protocol encodes: anything that goes wrong before
the first token is raised, so the router can try another replica, and anything after it
becomes a terminating ``TokenEvent.failure`` -- bytes have already reached the client, and a
retry would duplicate them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from turboserve.engine.core.types import EngineConfig, FinishReason
from turboserve.gateway.backends import register_backend
from turboserve.gateway.backends.protocol import (
    AdapterNotFoundError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
    TokenEvent,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator, Mapping, Sequence

    from turboserve.engine.runtime.async_engine import AsyncLLMEngine

logger = logging.getLogger(__name__)

__all__ = ["LocalEngineBackend"]


@register_backend("local")
class LocalEngineBackend:
    """Serve requests from an :class:`AsyncLLMEngine` running inside the gateway process.

    Constructor options are what ``configs/models.yaml`` puts under a backend's ``options``
    block and are forwarded verbatim by ``build_router``; every one of them is optional
    except ``model``.
    """

    def __init__(
        self,
        *,
        name: str = "local",
        model: str | None = None,
        engine: AsyncLLMEngine | None = None,
        config: EngineConfig | Mapping[str, Any] | None = None,
        served_models: Sequence[str] | None = None,
        adapters: Mapping[str, int] | None = None,
        local_files_only: bool = False,
        engine_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Describe the engine to build; nothing is loaded here.

        Args:
            name: the backend's name in metrics, logs and the router's pool.
            model: repo id or path of the model to serve. Required unless ``engine`` or a
                ``config`` carrying a model is supplied.
            engine: an already-built engine, which the backend then does not own the
                construction of (it still closes it, so the caller must not share one).
            config: an :class:`EngineConfig`, or a mapping of its fields.
            served_models: names this backend answers to. Defaults to the engine's model
                name, which is what an OpenAI client will send.
            adapters: adapter name to engine LoRA slot. An empty mapping means this backend
                serves the base model only and rejects any request naming an adapter.
            local_files_only: never contact the Hub; the offline path used by tests and by
                air-gapped deployments.
            engine_kwargs: extra keyword arguments forwarded to
                :class:`~turboserve.engine.runtime.engine.LLMEngine` (injected collaborators
                in tests, the speculative and LoRA hooks in production).
        """
        self.name = name
        self._config = self._resolve_config(model, config, engine)
        self._engine = engine
        self._owns_engine = engine is None
        self._adapters: dict[str, int] = {str(k): int(v) for k, v in (adapters or {}).items()}
        self._local_files_only = local_files_only
        self._engine_kwargs: dict[str, Any] = dict(engine_kwargs or {})
        self._served = list(served_models) if served_models else [self._config.model]
        self._lock = asyncio.Lock()
        self._closed = False

    @staticmethod
    def _resolve_config(
        model: str | None,
        config: EngineConfig | Mapping[str, Any] | None,
        engine: AsyncLLMEngine | None,
    ) -> EngineConfig:
        """Fold the three ways of naming a model into one :class:`EngineConfig`."""
        if isinstance(config, EngineConfig):
            resolved = config if model is None else config.model_copy(update={"model": model})
        elif config is not None:
            data = dict(config)
            if model is not None:
                data["model"] = model
            resolved = EngineConfig.model_validate(data)
        elif engine is not None:
            resolved = engine.engine.config
        elif model is not None:
            resolved = EngineConfig(model=model)
        else:
            raise ValueError(
                "LocalEngineBackend needs a model: pass model=..., config=..., or an engine"
            )
        return resolved

    # -- properties -----------------------------------------------------------------------

    @property
    def supports_lora(self) -> bool:
        """Whether this backend can route a request to an adapter.

        ``True`` only when adapters were configured. Reporting ``True`` with an empty
        mapping would make the router offer the backend for adapter traffic it would then
        reject.
        """
        return bool(self._adapters)

    @property
    def config(self) -> EngineConfig:
        """The engine configuration this backend will build (or did build) from."""
        return self._config

    @property
    def is_loaded(self) -> bool:
        """Whether the engine has been built."""
        return self._engine is not None

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._closed

    def adapter_slot(self, adapter: str | None) -> int:
        """Map an adapter name to an engine LoRA slot.

        Raises:
            AdapterNotFoundError: the tenant named an adapter this backend does not serve.
                Non-retryable on purpose -- another replica of the same pool has the same
                configuration, so retrying would only burn a second request.
        """
        if adapter is None:
            return 0
        slot = self._adapters.get(adapter)
        if slot is None:
            known = ", ".join(sorted(self._adapters)) or "none"
            raise AdapterNotFoundError(
                f"adapter {adapter!r} is not loaded; available: {known}",
                backend=self.name,
                status_code=403,
            )
        return slot

    # -- engine lifecycle ------------------------------------------------------------------

    async def _ensure_engine(self) -> AsyncLLMEngine:
        """Build and start the engine once, under a lock, off the event loop.

        Two callers arriving together must not build two engines -- that would put two
        copies of the weights on the device -- so the lock is held across the whole load,
        and the load itself runs in a worker thread because it is seconds of blocking I/O
        and GPU allocation.
        """
        if self._closed:
            raise BackendUnavailableError("backend is closed", backend=self.name)
        engine = self._engine
        if engine is not None:
            await engine.start()
            return engine
        async with self._lock:
            if self._engine is not None:
                await self._engine.start()
                return self._engine
            from turboserve.engine.runtime.async_engine import AsyncLLMEngine

            logger.info("loading %s for backend %r", self._config.model, self.name)
            try:
                built = await asyncio.to_thread(
                    AsyncLLMEngine.from_config,
                    self._config,
                    local_files_only=self._local_files_only,
                    **self._engine_kwargs,
                )
            except Exception as exc:
                raise BackendUnavailableError(
                    f"could not load {self._config.model!r}: {exc}", backend=self.name
                ) from exc
            await built.start()
            self._engine = built
            return built

    # -- Backend protocol -------------------------------------------------------------------

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Stream one request's tokens as :class:`TokenEvent` deltas plus a terminator.

        Declared as an async generator, per the ``Backend`` protocol: callers do
        ``async for ev in backend.generate(req)`` with no ``await`` on the call itself.
        """
        if req.model not in self._served:
            raise ModelNotFoundError(
                f"model {req.model!r} is not served by backend {self.name!r}",
                backend=self.name,
                status_code=404,
            )
        lora_id = self.adapter_slot(req.lora)
        engine = await self._ensure_engine()
        token_ids = req.prompt_token_ids
        prompt: str | list[int] = token_ids if token_ids is not None else (req.prompt_text or "")
        stream = engine.generate(
            req.request_id,
            prompt,
            req.sampling,
            tenant_id=req.tenant_id,
            priority=req.priority,
            lora_id=lora_id,
        )
        emitted = False
        try:
            async for output in stream:
                if output.finished:
                    yield TokenEvent.final(
                        req.request_id,
                        output.finish_reason or FinishReason.STOP,
                        token_ids=output.new_token_ids,
                        text=output.text_delta,
                        usage=output.usage(),
                    )
                    emitted = True
                    return
                if output.new_token_ids or output.text_delta:
                    yield TokenEvent.delta(
                        req.request_id, list(output.new_token_ids), output.text_delta
                    )
                    emitted = True
        except Exception as exc:
            if not emitted:
                raise
            # The body is already on the wire: report in-band so the gateway records a
            # failed request instead of trying to turn this into an HTTP status.
            logger.warning("stream for %s failed mid-flight: %s", req.request_id, exc)
            yield TokenEvent.failure(req.request_id, str(exc), finish_reason=FinishReason.ABORT)
            return
        finally:
            await stream.aclose()
        if not emitted:
            # The engine closed the stream without a terminator, which only happens if the
            # request was aborted between admission and its first step.
            yield TokenEvent.final(req.request_id, FinishReason.ABORT, usage={"prompt_tokens": 0})

    async def health(self) -> bool:
        """Whether the engine is usable. Never raises, per the protocol.

        An unloaded engine is reported healthy: the backend is configured and will load on
        the first request, and failing readiness until someone sends traffic would make a
        cold gateway permanently unready.
        """
        if self._closed:
            return False
        engine = self._engine
        return True if engine is None else engine.is_healthy

    async def models(self) -> list[str]:
        """Model names this backend answers to."""
        return list(self._served)

    async def close(self) -> None:
        """Stop the engine and release the KV pool. Idempotent."""
        if self._closed:
            return
        self._closed = True
        engine, self._engine = self._engine, None
        if engine is not None:
            await engine.close()

    def stats(self) -> dict[str, int | float | str]:
        """Engine statistics for the gateway's ``/metrics``; a cold-start summary when unloaded."""
        engine = self._engine
        if engine is None:
            return {"backend": self.name, "model": self._config.model, "loaded": 0}
        out = engine.stats()
        out["backend"] = self.name
        out["loaded"] = 1
        return out

    def __repr__(self) -> str:
        return (
            f"LocalEngineBackend(name={self.name!r}, model={self._config.model!r}, "
            f"loaded={self.is_loaded}, adapters={len(self._adapters)})"
        )
