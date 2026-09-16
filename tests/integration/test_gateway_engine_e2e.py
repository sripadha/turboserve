"""Gateway to reference engine, over HTTP, on a real checkpoint.

Every other suite tests one layer with the ones underneath it faked: the gateway against a
``MockBackend``, the backend against a fake engine, the engine against tensors. This file
is the one place where an OpenAI-shaped HTTP request travels the whole way down --
authentication, the tenant's quota, the router, ``LocalEngineBackend``, ``AsyncLLMEngine``,
the scheduler, paged attention, the sampler and the incremental detokenizer -- and comes
back as an SSE stream. It is what catches a contract that each side tested against its own
idea of the other.

The checkpoint is the cached tiny-random Qwen2 (a few megabytes, fp32, CPU), so the whole
file runs in seconds and needs neither the network nor a GPU; it is therefore an ordinary
test rather than a ``slow`` one. The tokens it produces are noise -- the model is randomly
initialised -- so nothing here asserts on their content, only on counts, order, framing
and accounting, which are the things the integration actually owns. The ``slow`` twin in
``test_gateway_engine_real_model.py`` runs the same path on a real 0.5B checkpoint, where
the text is worth reading.

The app builder, the SSE parser and the ``gateway`` fixture live in ``conftest.py`` because
both files use them.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
from _support import AUTH, sse_payloads

if TYPE_CHECKING:
    from turboserve.gateway.metrics import GatewayMetrics


async def test_chat_completion_streams_the_engines_tokens(gateway: dict[str, Any]) -> None:
    """A streamed chat completion arrives as OpenAI chunks with usage at the end."""
    client: httpx.AsyncClient = gateway["client"]
    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "model": gateway["model"],
            "messages": [{"role": "user", "content": "Hello world"}],
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")

    chunks = sse_payloads(response.text)
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    # The first chunk announces the role; the rest carry content; exactly one ends the choice.
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    finishes = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert len(finishes) == 1
    assert finishes[0]["choices"][0]["finish_reason"] == "length"

    usage = next(chunk["usage"] for chunk in chunks if chunk.get("usage"))
    assert usage["completion_tokens"] == 8
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


async def test_buffered_completion_matches_the_streamed_one(gateway: dict[str, Any]) -> None:
    """The same greedy request produces the same tokens buffered and streamed.

    This is the assertion that ties the two response paths together: the buffered body is
    assembled by the gateway from the same ``TokenEvent`` sequence the SSE writer frames,
    so a divergence means one of them is inventing or dropping a token.
    """
    client: httpx.AsyncClient = gateway["client"]
    body = {
        "model": gateway["model"],
        "prompt": "Hello world",
        "max_tokens": 6,
        "temperature": 0,
        "ignore_eos": True,
    }
    buffered = await client.post("/v1/completions", headers=AUTH, json=body)
    assert buffered.status_code == 200, buffered.text
    payload = buffered.json()
    assert payload["object"] == "text_completion"
    assert payload["usage"]["completion_tokens"] == 6
    assert payload["choices"][0]["finish_reason"] == "length"

    streamed = await client.post("/v1/completions", headers=AUTH, json={**body, "stream": True})
    text = "".join(chunk["choices"][0]["text"] for chunk in sse_payloads(streamed.text))
    assert text == payload["choices"][0]["text"]


async def test_concurrent_requests_share_one_engine(gateway: dict[str, Any]) -> None:
    """Four requests in flight at once are batched by the scheduler and all complete.

    Continuous batching is the engine's reason to exist, and the gateway's per-tenant
    concurrency gate sits directly in front of it; this is the smallest test that runs
    both at the same time on a real model.
    """
    client: httpx.AsyncClient = gateway["client"]

    async def one(index: int) -> httpx.Response:
        return await client.post(
            "/v1/completions",
            headers=AUTH,
            json={
                "model": gateway["model"],
                "prompt": f"prompt number {index}",
                "max_tokens": 5,
                "temperature": 0,
                "ignore_eos": True,
            },
        )

    responses = await asyncio.gather(*(one(index) for index in range(4)))
    assert [r.status_code for r in responses] == [200] * 4
    assert all(r.json()["usage"]["completion_tokens"] == 5 for r in responses)
    # Distinct prompts through a shared engine must not bleed into one another.
    assert len({r.json()["id"] for r in responses}) == 4


async def test_unknown_key_never_reaches_the_engine(gateway: dict[str, Any]) -> None:
    """Authentication runs before routing, so a bad key costs no engine work."""
    client: httpx.AsyncClient = gateway["client"]
    response = await client.post(
        "/v1/completions",
        headers={"Authorization": "Bearer sk-not-a-key"},
        json={"model": gateway["model"], "prompt": "hi", "max_tokens": 1},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


async def test_models_health_and_metrics_describe_the_engine(gateway: dict[str, Any]) -> None:
    """``/v1/models``, ``/readyz`` and ``/metrics`` all agree about the served model."""
    client: httpx.AsyncClient = gateway["client"]
    model = gateway["model"]

    listed = await client.get("/v1/models", headers=AUTH)
    assert listed.status_code == 200
    assert [card["id"] for card in listed.json()["data"]] == [model]

    assert (await client.get("/healthz")).status_code == 200
    ready = await client.get("/readyz")
    assert ready.status_code == 200, ready.text
    assert ready.json()["backends"][model]["reference"] is True

    # One request, then the counters: the label set is the contract the Grafana dashboard
    # and the Prometheus alert rules are written against.
    completed = await client.post(
        "/v1/completions",
        headers=AUTH,
        json={"model": model, "prompt": "hi", "max_tokens": 3, "temperature": 0},
    )
    assert completed.status_code == 200, completed.text

    metrics: GatewayMetrics = gateway["metrics"]
    labels = {"tenant": "acme", "model": model, "backend": "reference", "lane": "stable"}
    requests_total = metrics.sample_value(
        "turboserve_gateway_requests_total", status="ok", **labels
    )
    completions = metrics.sample_value(
        "turboserve_gateway_tokens_total", kind="completion", **labels
    )
    assert requests_total == 1.0
    assert completions == 3.0

    exposition = await client.get("/metrics")
    assert exposition.status_code == 200
    assert "turboserve_gateway_requests_total" in exposition.text
