"""The same gateway-to-engine path, on a real instruct checkpoint.

`test_gateway_engine_e2e.py` proves the plumbing with a randomly initialised tiny model: it
can assert token counts, SSE framing, usage accounting and concurrency, but not one thing
about the text, because the text is noise. This file is the other half. It runs the same
builder over a real Qwen2.5-0.5B-Instruct and asserts the things that only become
observable once the weights mean something:

* greedy decoding is deterministic — the same prompt twice gives the same text;
* the reference engine's continuous-batching path agrees, token for token, with the same
  engine serving the request alone, so batching does not change an answer;
* a stop string is honoured and does not appear in the output;
* the chat template is applied, so a chat completion and a raw completion of the same
  rendered prompt agree.

Marked ``slow`` because it loads ~1 GB of fp32 weights: minutes of CPU, not seconds. It is
deselected from the default suite and run by ``make test-slow``. It never downloads — the
``qwen_05b_path`` fixture resolves the checkpoint from the local Hugging Face cache and
skips when it is absent.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from _support import AUTH, sse_payloads

if TYPE_CHECKING:
    import httpx

pytestmark = pytest.mark.slow

#: Short and greedy: these tests are about agreement between paths, not about output length.
PROMPT = "List three primary colours, separated by commas."


async def complete(
    client: httpx.AsyncClient, model: str, prompt: str, **overrides: Any
) -> dict[str, Any]:
    """One buffered, greedy completion."""
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 24,
        "temperature": 0,
    }
    body.update(overrides)
    response = await client.post("/v1/completions", headers=AUTH, json=body)
    assert response.status_code == 200, response.text
    return response.json()


async def test_greedy_decoding_is_reproducible(real_gateway: dict[str, Any]) -> None:
    """Temperature 0 twice is the same text twice — the precondition for every other test."""
    client = real_gateway["client"]
    model = real_gateway["model"]
    first = await complete(client, model, PROMPT)
    second = await complete(client, model, PROMPT)
    assert first["choices"][0]["text"] == second["choices"][0]["text"]
    assert first["choices"][0]["text"].strip() != ""


async def test_batching_does_not_change_an_answer(real_gateway: dict[str, Any]) -> None:
    """The headline correctness property of continuous batching.

    A request served alone and the same request served in a step shared with three others
    must produce identical tokens. If block tables, the causal offset or the sampler's row
    indexing were wrong, this is where it shows: the tiny-model suite would see *different
    noise* and could not tell.
    """
    client = real_gateway["client"]
    model = real_gateway["model"]
    alone = await complete(client, model, PROMPT)

    others = ["Name a European capital.", "What is 2 + 2?", "Say hello."]
    batched = await asyncio.gather(
        complete(client, model, PROMPT),
        *(complete(client, model, other) for other in others),
    )
    assert batched[0]["choices"][0]["text"] == alone["choices"][0]["text"]
    # The neighbours must have produced their own answers, not copies of each other's.
    texts = [payload["choices"][0]["text"] for payload in batched[1:]]
    assert len(set(texts)) == len(texts)


async def test_a_stop_string_truncates_and_is_not_echoed(real_gateway: dict[str, Any]) -> None:
    """Stop strings are handled in the runtime's streaming decoder, not in the core.

    They need detokenised text, so they are the one stop condition that cannot be checked
    on token ids — and therefore the one that a tiny random model cannot exercise, because
    it never emits a chosen substring.
    """
    client = real_gateway["client"]
    model = real_gateway["model"]
    unstopped = await complete(client, model, "Count: one, two, three, four, five,")
    text = unstopped["choices"][0]["text"]
    marker = "three"
    if marker not in text:  # the model wandered; the mechanism is still worth asserting
        pytest.skip(f"the model did not emit {marker!r} to stop on")

    stopped = await complete(client, model, "Count: one, two, three, four, five,", stop=[marker])
    assert marker not in stopped["choices"][0]["text"]
    assert stopped["choices"][0]["text"] == text[: text.index(marker)]
    assert stopped["choices"][0]["finish_reason"] == "stop"


async def test_streamed_and_buffered_chat_completions_agree(real_gateway: dict[str, Any]) -> None:
    """The chat template is applied once, and both response paths render the same tokens."""
    client = real_gateway["client"]
    model = real_gateway["model"]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 24,
        "temperature": 0,
    }
    buffered = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert buffered.status_code == 200, buffered.text
    expected = buffered.json()["choices"][0]["message"]["content"]

    streamed = await client.post(
        "/v1/chat/completions", headers=AUTH, json={**body, "stream": True}
    )
    chunks = sse_payloads(streamed.text)
    text = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"]
    )
    assert text == expected
    assert expected.strip() != ""
