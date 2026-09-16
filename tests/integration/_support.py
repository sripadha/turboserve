"""Shared machinery for the gateway-to-engine end-to-end tests.

Both integration files run the *same* assertions through the *same* app builder; they differ
only in which checkpoint is behind it — the cached tiny-random Qwen2 for the default suite,
and a real Qwen2.5-0.5B for the ``slow`` one. Keeping the builder here rather than in either
file is what makes that claim true: `tests/` has no ``__init__.py``, so a helper duplicated
into two test modules is a helper that can quietly diverge.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx

from turboserve.config import Settings
from turboserve.engine.core.types import EngineConfig, SchedulerConfig
from turboserve.gateway.app import GatewayOptions, create_app
from turboserve.gateway.auth import hash_api_key
from turboserve.gateway.backends.local_engine import LocalEngineBackend
from turboserve.gateway.metrics import GatewayMetrics
from turboserve.gateway.router import Router
from turboserve.gateway.tenants import Tenant, TenantRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from fastapi import FastAPI

#: The one tenant these tests authenticate as.
KEY = "sk-integration-acme"
AUTH = {"Authorization": f"Bearer {KEY}"}

#: Real checkpoint for the ``slow`` tier, resolved from the local cache only.
SLOW_MODEL_REPOS: tuple[str, ...] = ("Qwen/Qwen2.5-0.5B-Instruct",)


def engine_config(model_path: Path) -> EngineConfig:
    """A pool small enough to build instantly and large enough not to preempt this load."""
    return EngineConfig(
        model=str(model_path),
        device="cpu",
        dtype="float32",
        scheduler=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=256,
            block_size=16,
            num_blocks=64,
        ),
    )


def build_app(model_path: Path) -> tuple[FastAPI, LocalEngineBackend, GatewayMetrics]:
    """A one-tenant gateway whose only pool is an in-process reference engine."""
    model = str(model_path)
    backend = LocalEngineBackend(
        name="reference",
        config=engine_config(model_path),
        served_models=[model],
        local_files_only=True,
    )
    router = Router(health_ttl_s=0.0)
    router.add_backend(model, backend)
    tenants = TenantRegistry(
        [Tenant(tenant_id="acme", keys_sha256=[hash_api_key(KEY)], max_concurrency=4)]
    )
    metrics = GatewayMetrics()
    app = create_app(
        Settings(model=model),
        router,
        tenants=tenants,
        metrics=metrics,
        options=GatewayOptions(require_auth=True, served_model_names=[model]),
    )
    return app, backend, metrics


async def running_gateway(model_path: Path) -> AsyncIterator[dict[str, Any]]:
    """Client, model name and metrics registry for one gateway instance.

    The engine is built lazily by the backend on the first request and shut down here, in
    the same event loop that started it — an ``AsyncLLMEngine``'s step-loop task belongs to
    the loop it was started in, so the teardown cannot be moved out of the async world.
    """
    app, backend, metrics = build_app(model_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        try:
            yield {"client": client, "model": str(model_path), "metrics": metrics}
        finally:
            await backend.close()


def sse_payloads(body: str) -> list[dict[str, Any]]:
    """Parse an OpenAI SSE body strictly, assert the ``[DONE]`` sentinel and drop it."""
    assert body.endswith("\n\n"), "the stream must end with a blank line"
    payloads: list[str] = []
    for block in body.split("\n\n"):
        if not block:
            continue
        assert "\n" not in block, f"unexpected multi-line SSE event: {block!r}"
        assert block.startswith("data: "), f"not an OpenAI data event: {block!r}"
        payloads.append(block[len("data: ") :])
    assert payloads[-1] == "[DONE]", "an OpenAI stream must terminate with [DONE]"
    return [json.loads(payload) for payload in payloads[:-1]]
