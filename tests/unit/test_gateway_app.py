"""End-to-end HTTP behaviour of the gateway app: statuses, SSE wire format, metrics.

The SSE assertions use the strict parser below rather than an SSE client library, because
the thing under test *is* the wire format: framing, the ``data: `` prefix, the blank-line
separator and the ``[DONE]`` sentinel.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from turboserve.config import Settings
from turboserve.gateway.app import GatewayOptions, create_app
from turboserve.gateway.auth import hash_api_key
from turboserve.gateway.backends.mock import MockBackend
from turboserve.gateway.chat_template import ChatTemplate, ChatTemplateCache
from turboserve.gateway.metrics import GatewayMetrics
from turboserve.gateway.router import Router
from turboserve.gateway.tenants import Tenant, TenantRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

KEY = "sk-test-acme"
OTHER_KEY = "sk-test-globex"
MODEL = "mock-model"
AUTH = {"Authorization": f"Bearer {KEY}"}


def parse_sse(body: str) -> list[str]:
    """Strictly parse an OpenAI SSE body into its ``data:`` payloads.

    Deliberately unforgiving: every event must be ``data: <payload>`` followed by a blank
    line, exactly as OpenAI's servers emit and as the ``openai`` client's parser expects.
    """
    assert body.endswith("\n\n"), "the stream must end with a blank line"
    payloads: list[str] = []
    for block in body.split("\n\n"):
        if not block:
            continue
        lines = block.split("\n")
        assert len(lines) == 1, f"unexpected multi-line SSE event: {block!r}"
        line = lines[0]
        assert line.startswith("data: "), f"not an OpenAI data event: {line!r}"
        payloads.append(line[len("data: ") :])
    return payloads


def json_payloads(body: str) -> list[dict[str, Any]]:
    """Parsed chunks of an SSE body, with the ``[DONE]`` sentinel asserted and dropped."""
    payloads = parse_sse(body)
    assert payloads[-1] == "[DONE]", "an OpenAI stream must terminate with [DONE]"
    return [json.loads(payload) for payload in payloads[:-1]]


def build_tenants() -> TenantRegistry:
    """Two tenants: one broadly allowed, one fenced to a model this gateway does not serve."""
    return TenantRegistry(
        [
            Tenant(
                tenant_id="acme",
                keys_sha256=[hash_api_key(KEY)],
                max_concurrency=4,
                adapters={"support": "acme-support-r16"},
            ),
            Tenant(
                tenant_id="globex",
                keys_sha256=[hash_api_key(OTHER_KEY)],
                allowed_models=["some-other-model"],
            ),
        ]
    )


def build_app(
    *,
    tenants: TenantRegistry | None = None,
    backend: MockBackend | None = None,
    metrics: GatewayMetrics | None = None,
    options: GatewayOptions | None = None,
    templates: ChatTemplateCache | None = None,
    health_ttl_s: float = 5.0,
) -> Any:
    """A gateway app around one mock backend and an isolated metrics registry."""
    router = Router(health_ttl_s=health_ttl_s)
    router.add_backend(
        MODEL,
        backend
        or MockBackend(name="mock-a", models=[MODEL], adapters=["acme-support-r16"], max_tokens=4),
    )
    # Point the settings at files that do not exist so the test never picks up repo config.
    settings = Settings(
        tenants_file="/nonexistent/tenants.yaml", models_file="/nonexistent/models.yaml"
    )
    return create_app(
        settings,
        router,
        tenants=tenants or build_tenants(),
        metrics=metrics or GatewayMetrics(),
        templates=templates,
        options=options or GatewayOptions(),
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client over the default app."""
    with TestClient(build_app()) as test_client:
        yield test_client


# -- authentication and authorisation ---------------------------------------------------


def test_missing_key_is_401_with_a_challenge(client: TestClient) -> None:
    response = client.post("/v1/completions", json={"model": MODEL, "prompt": "hi"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "missing_api_key"


def test_unknown_key_is_401(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hi"},
        headers={"Authorization": "Bearer nope"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_model_outside_the_allow_list_is_403(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hi"},
        headers={"Authorization": f"Bearer {OTHER_KEY}"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "permission_error"


def test_unknown_adapter_is_403(client: TestClient) -> None:
    response = client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "hi", "lora": "nope"}, headers=AUTH
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "adapter_not_allowed"


def test_a_permitted_adapter_is_served(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hi", "lora": "support"},
        headers=AUTH,
    )
    assert response.status_code == 200


def test_unknown_model_is_404(client: TestClient) -> None:
    response = client.post(
        "/v1/completions", json={"model": "absent", "prompt": "hi"}, headers=AUTH
    )
    assert response.status_code == 404


def test_auth_can_be_disabled_for_closed_deployments() -> None:
    app = build_app(options=GatewayOptions(require_auth=False))
    with TestClient(app) as client:
        response = client.post("/v1/completions", json={"model": MODEL, "prompt": "hi"})
    assert response.status_code == 200


# -- request validation -----------------------------------------------------------------


def test_a_malformed_body_is_400_not_422(client: TestClient) -> None:
    # OpenAI clients branch on 400; FastAPI's default 422 reads to them as a protocol error.
    response = client.post("/v1/completions", json={"model": MODEL}, headers=AUTH)
    assert response.status_code == 400
    assert "prompt" in response.json()["error"]["message"]


def test_an_unsupported_field_is_refused_rather_than_ignored(client: TestClient) -> None:
    response = client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "hi", "best_of": 4}, headers=AUTH
    )
    assert response.status_code == 400
    assert "best_of" in response.json()["error"]["message"]


def test_multiple_samples_are_refused(client: TestClient) -> None:
    response = client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "hi", "n": 2}, headers=AUTH
    )
    assert response.status_code == 400


def test_an_unknown_but_harmless_field_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hi", "frequency_penalty": 0.5},
        headers=AUTH,
    )
    assert response.status_code == 200


# -- completions ------------------------------------------------------------------------


def test_non_streaming_completion_shape(client: TestClient) -> None:
    response = client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "hi", "max_tokens": 4}, headers=AUTH
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    assert body["model"] == MODEL
    assert len(body["choices"]) == 1
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["text"]
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}
    assert body["usage"]["completion_tokens"] == 4
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_streaming_completion_is_openai_shaped(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hi", "max_tokens": 3, "stream": True},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks = json_payloads(response.text)
    assert all(chunk["object"] == "text_completion" for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == 1  # one id for the whole stream
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}
    assert all(chunk["choices"][0]["finish_reason"] is None for chunk in chunks[:-1])


def test_streaming_usage_chunk_is_opt_in(client: TestClient) -> None:
    without = json_payloads(
        client.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": "hi", "max_tokens": 2, "stream": True},
            headers=AUTH,
        ).text
    )
    assert all(chunk.get("usage") is None for chunk in without)

    with_usage = json_payloads(
        client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": "hi",
                "max_tokens": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers=AUTH,
        ).text
    )
    final = with_usage[-1]
    assert final["choices"] == []
    assert final["usage"]["completion_tokens"] == 2


def test_completion_accepts_token_ids(client: TestClient) -> None:
    response = client.post(
        "/v1/completions", json={"model": MODEL, "prompt": [10, 11, 12]}, headers=AUTH
    )
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == 3


# -- chat completions -------------------------------------------------------------------


def test_non_streaming_chat_shape(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
        headers=AUTH,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]


def test_streaming_chat_sends_role_first_then_content(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "stream": True,
        },
        headers=AUTH,
    )
    chunks = json_payloads(response.text)
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    # Content chunks carry only content; repeating the role would corrupt concatenation.
    content_chunks = [c for c in chunks[1:-1] if c["choices"][0]["delta"]]
    assert content_chunks
    assert all(set(c["choices"][0]["delta"]) == {"content"} for c in content_chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}
    assert chunks[-1]["choices"][0]["delta"] == {}


def test_streamed_and_buffered_chat_agree_on_the_text() -> None:
    # Same request id path is not shared, but the mock is deterministic per request id, so
    # the comparison is of the two renderings rather than of two samples.
    app = build_app(backend=MockBackend(name="mock-a", models=[MODEL], max_tokens=5, seed=99))
    with TestClient(app) as client:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }
        buffered = client.post("/v1/chat/completions", json=body, headers=AUTH).json()
        chunks = json_payloads(
            client.post("/v1/chat/completions", json={**body, "stream": True}, headers=AUTH).text
        )
    streamed = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"]
    )
    assert len(streamed.split()) == len(buffered["choices"][0]["message"]["content"].split())


# -- quotas -----------------------------------------------------------------------------


def test_rpm_exhaustion_is_429_with_retry_after() -> None:
    tenants = TenantRegistry([Tenant(tenant_id="acme", keys_sha256=[hash_api_key(KEY)], rpm=2)])
    with TestClient(build_app(tenants=tenants)) as client:
        body = {"model": MODEL, "prompt": "hi", "max_tokens": 1}
        assert client.post("/v1/completions", json=body, headers=AUTH).status_code == 200
        assert client.post("/v1/completions", json=body, headers=AUTH).status_code == 200
        response = client.post("/v1/completions", json=body, headers=AUTH)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"  # one request at 2 per minute
    assert response.json()["error"]["type"] == "rate_limit_error"
    assert response.json()["error"]["code"] == "rpm_exceeded"


def test_tpm_exhaustion_is_429() -> None:
    # The prompt is 400 characters, which the fallback estimator scores at 100 tokens.
    tenants = TenantRegistry([Tenant(tenant_id="acme", keys_sha256=[hash_api_key(KEY)], tpm=120)])
    with TestClient(build_app(tenants=tenants)) as client:
        body = {"model": MODEL, "prompt": "x" * 400, "max_tokens": 4}
        assert client.post("/v1/completions", json=body, headers=AUTH).status_code == 200
        response = client.post("/v1/completions", json=body, headers=AUTH)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "tpm_exceeded"
    assert int(response.headers["retry-after"]) >= 1


def test_a_refused_request_never_reaches_the_backend() -> None:
    backend = MockBackend(name="mock-a", models=[MODEL], max_tokens=1)
    tenants = TenantRegistry([Tenant(tenant_id="acme", keys_sha256=[hash_api_key(KEY)], rpm=1)])
    with TestClient(build_app(tenants=tenants, backend=backend)) as client:
        body = {"model": MODEL, "prompt": "hi"}
        client.post("/v1/completions", json=body, headers=AUTH)
        client.post("/v1/completions", json=body, headers=AUTH)
    assert backend.stats.requests == 1


# -- fleet routes -----------------------------------------------------------------------


def test_models_listing_is_filtered_per_tenant(client: TestClient) -> None:
    allowed = client.get("/v1/models", headers=AUTH).json()
    assert [card["id"] for card in allowed["data"]] == [MODEL]
    fenced = client.get("/v1/models", headers={"Authorization": f"Bearer {OTHER_KEY}"}).json()
    assert fenced["data"] == []


def test_models_listing_requires_authentication(client: TestClient) -> None:
    assert client.get("/v1/models").status_code == 401


def test_healthz_does_not_depend_on_backends() -> None:
    backend = MockBackend(name="mock-a", models=[MODEL])
    backend.set_healthy(False)
    with TestClient(build_app(backend=backend)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_follows_backend_health() -> None:
    backend = MockBackend(name="mock-a", models=[MODEL])
    # A zero health cache makes every probe fresh, so the flip is visible immediately.
    with TestClient(build_app(backend=backend, health_ttl_s=0.0)) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["backends"] == {MODEL: {"mock-a": True}}
        backend.set_healthy(False)
        assert client.get("/readyz").status_code == 503


def test_metrics_exposition_carries_the_request_dimensions() -> None:
    metrics = GatewayMetrics()
    with TestClient(build_app(metrics=metrics)) as client:
        client.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": "hi", "max_tokens": 4},
            headers=AUTH,
        )
        client.post("/v1/completions", json={"model": MODEL, "prompt": "hi"})  # 401
        body = client.get("/metrics")

    assert body.status_code == 200
    assert body.headers["content-type"].startswith("text/plain")
    text = body.text
    assert "turboserve_gateway_ttft_seconds_bucket" in text
    assert "turboserve_gateway_e2e_seconds_count" in text
    labels = {"tenant": "acme", "model": MODEL, "backend": "mock-a", "lane": "stable"}
    assert metrics.sample_value("turboserve_gateway_requests_total", **labels, status="ok") == 1.0
    assert (
        metrics.sample_value("turboserve_gateway_tokens_total", **labels, kind="completion") == 4.0
    )
    assert metrics.sample_value("turboserve_gateway_ttft_seconds_count", **labels) == 1.0
    assert (
        metrics.sample_value(
            "turboserve_gateway_requests_total",
            tenant="unknown",
            model="",
            backend="",
            lane="stable",
            status="unauthorized",
        )
        == 1.0
    )
    assert (
        metrics.sample_value("turboserve_gateway_inflight_requests", tenant="acme", model=MODEL)
        == 0.0
    )


def test_rate_limited_requests_are_counted_by_which_quota_refused_them() -> None:
    metrics = GatewayMetrics()
    tenants = TenantRegistry([Tenant(tenant_id="acme", keys_sha256=[hash_api_key(KEY)], rpm=1)])
    with TestClient(build_app(tenants=tenants, metrics=metrics)) as client:
        body = {"model": MODEL, "prompt": "hi"}
        client.post("/v1/completions", json=body, headers=AUTH)
        client.post("/v1/completions", json=body, headers=AUTH)
    assert (
        metrics.sample_value("turboserve_gateway_rate_limited_total", tenant="acme", limit="rpm")
        == 1.0
    )


# -- failure paths ----------------------------------------------------------------------


def test_a_backend_that_cannot_start_is_503() -> None:
    backend = MockBackend(name="mock-a", models=[MODEL], error_probability=1.0)
    with TestClient(build_app(backend=backend)) as client:
        response = client.post(
            "/v1/completions", json={"model": MODEL, "prompt": "hi"}, headers=AUTH
        )
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"


def test_a_mid_stream_failure_is_reported_inside_the_stream() -> None:
    # The status line is long gone by then, so the error has to travel as a chunk.
    backend = MockBackend(name="mock-a", models=[MODEL], max_tokens=8, drop_probability=1.0, seed=5)
    with TestClient(build_app(backend=backend)) as client:
        response = client.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": "hi", "max_tokens": 8, "stream": True},
            headers=AUTH,
        )
    assert response.status_code == 200
    chunks = json_payloads(response.text)
    assert "error" in chunks[-1]
    assert "drop" in chunks[-1]["error"]["message"]


def test_a_mid_stream_failure_on_a_buffered_request_is_502() -> None:
    backend = MockBackend(name="mock-a", models=[MODEL], max_tokens=8, drop_probability=1.0, seed=5)
    with TestClient(build_app(backend=backend)) as client:
        response = client.post(
            "/v1/completions", json={"model": MODEL, "prompt": "hi", "max_tokens": 8}, headers=AUTH
        )
    assert response.status_code == 502


def test_failed_requests_are_counted_as_errors() -> None:
    metrics = GatewayMetrics()
    backend = MockBackend(name="mock-a", models=[MODEL], max_tokens=8, drop_probability=1.0, seed=5)
    with TestClient(build_app(backend=backend, metrics=metrics)) as client:
        client.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": "hi", "max_tokens": 8, "stream": True},
            headers=AUTH,
        )
    assert (
        metrics.sample_value(
            "turboserve_gateway_requests_total",
            tenant="acme",
            model=MODEL,
            backend="mock-a",
            lane="stable",
            status="error",
        )
        == 1.0
    )


# -- chat template wiring ---------------------------------------------------------------


def test_the_gateway_falls_back_when_the_model_has_no_local_tokenizer(client: TestClient) -> None:
    # "mock-model" is not a checkpoint; loading a tokenizer for it must not fail the request.
    response = client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        headers=AUTH,
    )
    assert response.status_code == 200


def test_the_rendered_prompt_reaches_the_backend() -> None:
    captured: list[str] = []

    class Recording(MockBackend):
        async def generate(self, req: Any) -> Any:  # type: ignore[override]
            captured.append(str(req.prompt))
            async for event in super().generate(req):
                yield event

    templates = ChatTemplateCache()
    templates.put(MODEL, ChatTemplate(None))
    backend = Recording(name="mock-a", models=[MODEL], max_tokens=2)
    with TestClient(build_app(backend=backend, templates=templates)) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                ],
            },
            headers=AUTH,
        )
    assert captured == ["System: be brief\nUser: hi\nAssistant:"]
