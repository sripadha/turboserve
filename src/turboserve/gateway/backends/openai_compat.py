"""A backend that streams from any OpenAI-compatible HTTP server (vLLM, SGLang, TGI, ...).

This is the production path: the same gateway that fronts the in-process reference engine
fronts a fleet of vLLM or SGLang pods, so all of them can be measured behind identical auth,
routing, quota and accounting code. Without that, a comparison between the reference engine
and a production one would also be a comparison between two different serving stacks.

The second production engine needed no new code on the request path, which is the point of
coding against a wire protocol rather than against a server: vLLM and SGLang take the same
request body (``ignore_eos`` and the other extra members included), frame their SSE the same
way, honour ``stream_options.include_usage`` the same way and address a LoRA adapter through
the same ``model`` field. What does differ is what each will *say about itself*, and
:meth:`OpenAICompatBackend.server_info` is the whole of the difference.

Three details of the upstream protocol drive the code below:

* **Extra sampling knobs travel as top-level JSON keys.** What the ``openai`` Python client
  calls ``extra_body`` is, on the wire, just additional members of the request object; that
  is how vLLM receives ``top_k``, ``repetition_penalty`` and ``ignore_eos``. They are sent
  only when they differ from their neutral value, so a stricter upstream that rejects
  unknown fields still works for ordinary requests.
* **A LoRA adapter is addressed through the ``model`` field.** vLLM serves each loaded
  adapter under its adapter name, so routing to an adapter is choosing a different ``model``
  string, not a separate parameter.
* **Failure is two different things either side of the first token.** Before it, nothing has
  been promised and a retryable error lets the router try another replica. After it, the
  client already holds part of a completion, so the failure is delivered *in band* as a
  terminating error event -- never raised, never retried.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit, urlunsplit

import httpx

from turboserve.engine.core.types import FinishReason
from turboserve.gateway.backends import register_backend
from turboserve.gateway.backends.protocol import (
    BackendOverloadedError,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
    TokenEvent,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

__all__ = ["OpenAICompatBackend"]

_DATA_PREFIX: Final = "data:"
_DONE: Final = "[DONE]"

#: Native (non-OpenAI) endpoints asked for by :meth:`OpenAICompatBackend.server_info`.
#: ``/version`` is served by both vLLM and SGLang; ``/get_server_info`` is SGLang's, and a
#: server that answers it is therefore an SGLang server.
_VERSION_PATH: Final = "/version"
_SERVER_INFO_PATH: Final = "/get_server_info"

#: Fields of a ``/get_server_info`` response worth recording. A whitelist rather than the
#: whole document: the server returns its complete launch configuration, most of which is
#: defaults, and a result file should carry the settings that change what was measured --
#: the model, the numeric type, the context window, the scheduler's limits, and whether
#: prefix caching, speculation or multi-LoRA were on -- not a copy of an argument parser.
_SERVER_INFO_FIELDS: Final[tuple[str, ...]] = (
    "model_path",
    "served_model_name",
    "tokenizer_path",
    "dtype",
    "context_length",
    "max_running_requests",
    "max_total_tokens",
    "chunked_prefill_size",
    "mem_fraction_static",
    "disable_radix_cache",
    "speculative_algorithm",
    "speculative_num_steps",
    "speculative_num_draft_tokens",
    "max_loras_per_batch",
    "attention_backend",
    "schedule_policy",
    "tp_size",
    "dp_size",
)

#: Upstream ``finish_reason`` strings mapped onto the engine's enum. Anything unrecognised
#: becomes ``STOP``: the completion did end, and inventing a new reason would break the
#: response models that validate this field against a closed set.
_FINISH_REASONS: Final[dict[str, FinishReason]] = {
    "stop": FinishReason.STOP,
    "eos_token": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "abort": FinishReason.ABORT,
    "cancelled": FinishReason.ABORT,
}


def _origin(base_url: str) -> str:
    """Scheme and host of a base URL, without its path.

    ``/health`` and ``/metrics`` live at the server root while completions live under
    ``/v1``, so the health probe cannot simply append to ``base_url``.
    """
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _server_settings(payload: Any) -> dict[str, Any]:
    """The recorded subset of a ``/get_server_info`` response.

    The document nests its launch arguments under ``server_args`` in some releases and
    flattens them in others, so both shapes are read and the flat one wins -- a field a
    running server reports about itself is more current than the argument it was started
    with.
    """
    if not isinstance(payload, Mapping):
        return {}
    nested = payload.get("server_args")
    sources = [nested, payload] if isinstance(nested, Mapping) else [payload]
    settings: dict[str, Any] = {}
    for source in sources:
        for field in _SERVER_INFO_FIELDS:
            value = source.get(field)
            if isinstance(value, str | int | float | bool):
                settings[field] = value
    return settings


@register_backend("openai")
class OpenAICompatBackend:
    """Streams completions from an OpenAI-compatible server over HTTP."""

    supports_lora = True

    def __init__(
        self,
        base_url: str,
        *,
        name: str | None = None,
        api_key: str | None = None,
        model_map: Mapping[str, str] | None = None,
        adapter_models: Mapping[str, str] | None = None,
        timeout_s: float = 300.0,
        connect_timeout_s: float = 5.0,
        headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
        health_path: str = "/health",
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required, e.g. http://127.0.0.1:8000/v1")
        self.base_url = base_url.rstrip("/")
        self.name = name or urlsplit(self.base_url).netloc or "openai"
        self._model_map = dict(model_map or {})
        self._adapter_models = dict(adapter_models or {})
        self._extra_body = dict(extra_body or {})
        self._health_url = _origin(self.base_url) + health_path
        self._owns_client = client is None
        request_headers = {"accept": "text/event-stream", **dict(headers or {})}
        if api_key:
            request_headers["authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            headers=request_headers,
        )
        if client is not None:
            # A caller-supplied client (tests, connection-pool sharing) keeps its own
            # headers; ours are merged so authentication still travels.
            self._client.headers.update(request_headers)
        self._closed = False

    # -- request construction -----------------------------------------------------------

    def upstream_model(self, req: GenerateRequest) -> str:
        """The ``model`` string to send upstream, resolving adapters and aliases."""
        if req.lora is not None:
            return self._adapter_models.get(req.lora, req.lora)
        return self._model_map.get(req.model, req.model)

    def build_body(self, req: GenerateRequest) -> dict[str, Any]:
        """Render a :class:`GenerateRequest` as an upstream completions request body.

        Only non-neutral options are included. ``stream_options.include_usage`` asks the
        server for a final chunk carrying token counts, which is the only way to get the
        upstream's own tokenisation of the prompt rather than a guess at it.
        """
        sampling = req.sampling
        body: dict[str, Any] = {
            "model": self.upstream_model(req),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
        }
        ids = req.prompt_token_ids
        body["prompt"] = ids if ids is not None else req.prompt_text
        if sampling.seed is not None:
            body["seed"] = sampling.seed
        if sampling.stop:
            body["stop"] = list(sampling.stop)
        if sampling.stop_token_ids:
            body["stop_token_ids"] = list(sampling.stop_token_ids)
        if sampling.top_k:
            body["top_k"] = sampling.top_k
        if sampling.repetition_penalty != 1.0:
            body["repetition_penalty"] = sampling.repetition_penalty
        if sampling.ignore_eos:
            body["ignore_eos"] = True
        if req.priority:
            body["priority"] = req.priority
        body.update(self._extra_body)
        return body

    # -- generation ---------------------------------------------------------------------

    async def generate(self, req: GenerateRequest) -> AsyncIterator[TokenEvent]:
        """Stream the upstream completion, translating its SSE into token events."""
        if self._closed:
            raise BackendUnavailableError("backend is closed", backend=self.name)
        body = self.build_body(req)
        started = False
        finish_reason: FinishReason | None = None
        usage: dict[str, Any] | None = None
        try:
            async with self._client.stream("POST", "/completions", json=body) as response:
                if response.status_code != httpx.codes.OK:
                    raise await self._status_error(response, req)
                async for line in response.aiter_lines():
                    payload = self._payload(line)
                    if payload is None:
                        continue
                    if payload == _DONE:
                        break
                    chunk = self._decode(payload)
                    if chunk is None:
                        continue
                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage = chunk_usage
                    text, reason = self._choice(chunk)
                    if reason is not None:
                        finish_reason = reason
                    if text:
                        started = True
                        yield TokenEvent.delta(req.request_id, [], text)
        except httpx.HTTPError as exc:
            error = self._transport_error(exc)
            if not started:
                raise error from exc
            logger.warning("stream from %s interrupted after first token: %s", self.name, exc)
            yield TokenEvent.failure(req.request_id, str(error), finish_reason=FinishReason.ABORT)
            return
        yield TokenEvent.final(
            req.request_id,
            finish_reason or FinishReason.STOP,
            usage=self._normalise_usage(usage),
        )

    @staticmethod
    def _payload(line: str) -> str | None:
        """Extract the payload of one SSE line, or ``None`` for blanks and comments."""
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            return None
        if not stripped.startswith(_DATA_PREFIX):
            return None
        return stripped[len(_DATA_PREFIX) :].strip()

    @staticmethod
    def _decode(payload: str) -> dict[str, Any] | None:
        """Parse one SSE payload, tolerating a malformed chunk rather than dying on it."""
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("discarding unparseable SSE payload: %.120s", payload)
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _choice(chunk: Mapping[str, Any]) -> tuple[str, FinishReason | None]:
        """Pull text and finish reason out of a chunk, in either completions or chat shape."""
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return "", None
        choice = choices[0]
        if not isinstance(choice, dict):
            return "", None
        text = choice.get("text")
        if not isinstance(text, str):
            delta = choice.get("delta")
            text = delta.get("content") if isinstance(delta, dict) else None
        reason = choice.get("finish_reason")
        mapped = _FINISH_REASONS.get(reason, FinishReason.STOP) if isinstance(reason, str) else None
        return (text if isinstance(text, str) else ""), mapped

    @staticmethod
    def _normalise_usage(usage: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Keep the integer fields of an upstream usage block, dropping the rest."""
        if not usage:
            return None
        out: dict[str, Any] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                out[key] = value
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
            out["cached_prompt_tokens"] = details["cached_tokens"]
        return out or None

    # -- error mapping ------------------------------------------------------------------

    async def _status_error(self, response: httpx.Response, req: GenerateRequest) -> Exception:
        """Turn a non-200 response into the error class the router knows how to act on."""
        try:
            detail = (await response.aread()).decode("utf-8", "replace")[:500]
        except httpx.HTTPError:  # pragma: no cover - body already gone
            detail = ""
        status = response.status_code
        message = (
            f"{self.name} returned HTTP {status}: {detail}"
            if detail
            else (f"{self.name} returned HTTP {status}")
        )
        if status in (httpx.codes.TOO_MANY_REQUESTS, httpx.codes.SERVICE_UNAVAILABLE):
            return BackendOverloadedError(message, backend=self.name, status_code=status)
        if status == httpx.codes.NOT_FOUND:
            return ModelNotFoundError(
                f"{self.name} does not serve model {self.upstream_model(req)!r}",
                backend=self.name,
                status_code=status,
            )
        if status in (httpx.codes.REQUEST_TIMEOUT, httpx.codes.GATEWAY_TIMEOUT):
            return BackendTimeoutError(message, backend=self.name, status_code=status)
        if httpx.codes.BAD_REQUEST <= status < httpx.codes.INTERNAL_SERVER_ERROR:
            return BackendRequestError(message, backend=self.name, status_code=status)
        return BackendUnavailableError(message, backend=self.name, status_code=status)

    def _transport_error(self, exc: httpx.HTTPError) -> Exception:
        """Classify a transport-level failure.

        A timeout and a refused connection are both retryable but they are different
        operational signals -- one says the replica is saturated, the other says it is gone --
        so they keep separate classes and separate log lines.
        """
        if isinstance(exc, httpx.TimeoutException):
            return BackendTimeoutError(f"{self.name} timed out: {exc}", backend=self.name)
        return BackendUnavailableError(f"{self.name} is unreachable: {exc}", backend=self.name)

    # -- protocol remainder -------------------------------------------------------------

    async def health(self) -> bool:
        """Probe the server's health endpoint, falling back to the model list.

        Never raises. Not every OpenAI-compatible server exposes ``/health`` (TGI does, some
        proxies do not), so a 404 there is answered by asking for the model list instead --
        which is a stricter check anyway, since it exercises the API surface we use.
        """
        if self._closed:
            return False
        try:
            response = await self._client.get(self._health_url)
            if response.status_code == httpx.codes.OK:
                return True
            if response.status_code != httpx.codes.NOT_FOUND:
                return False
        except httpx.HTTPError as exc:
            logger.debug("health probe of %s failed: %s", self.name, exc)
            return False
        try:
            await self.models()
        except Exception as exc:  # noqa: BLE001 - health must never raise
            logger.debug("model listing of %s failed: %s", self.name, exc)
            return False
        return True

    async def server_info(self) -> dict[str, Any]:
        """What the upstream server reports about itself, or ``{}`` when it says nothing.

        Two native endpoints, neither of them part of the OpenAI API:

        * ``GET /version`` -> ``{"version": ...}``. vLLM and SGLang both serve it.
        * ``GET /get_server_info`` -> the launch configuration (model path, dtype, context
          length, scheduler settings). This one is SGLang's, so an answer here is how a
          server identifies itself as SGLang rather than vLLM.

        The result belongs in a benchmark's result file, where it is the difference between
        "an OpenAI-compatible server at this URL" and a run somebody else can reproduce: the
        flags a client cannot otherwise see -- whether the radix cache was disabled, what the
        context window was, how many adapters a batch could mix -- are exactly the flags that
        decide what the numbers mean.

        Never raises and never blocks a request path. A server that exposes neither endpoint
        (a proxy, TGI, an older build) contributes an empty block rather than an error, which
        is the same policy :meth:`health` follows for the same reason.
        """
        if self._closed:
            return {}
        info: dict[str, Any] = {}
        version = await self._get_root_json(_VERSION_PATH)
        if isinstance(version, Mapping) and isinstance(version.get("version"), str):
            info["version"] = version["version"]
        settings = _server_settings(await self._get_root_json(_SERVER_INFO_PATH))
        if settings:
            info["settings"] = settings
        return info

    async def _get_root_json(self, path: str) -> Any:
        """GET ``path`` at the server root and decode it, or ``None``.

        The root, not ``base_url``: completions live under ``/v1`` while the native
        endpoints sit beside it, exactly as ``/health`` does.
        """
        try:
            response = await self._client.get(_origin(self.base_url) + path)
        except httpx.HTTPError as exc:
            logger.debug("%s of %s failed: %s", path, self.name, exc)
            return None
        if response.status_code != httpx.codes.OK:
            return None
        try:
            return response.json()
        except ValueError:
            logger.debug("%s of %s returned a non-JSON body", path, self.name)
            return None

    async def models(self) -> list[str]:
        """Model ids the upstream server advertises."""
        if self._closed:
            raise BackendUnavailableError("backend is closed", backend=self.name)
        try:
            response = await self._client.get("/models")
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        if response.status_code != httpx.codes.OK:
            raise BackendUnavailableError(
                f"{self.name} returned HTTP {response.status_code} for /models",
                backend=self.name,
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendUnavailableError(
                f"{self.name} returned a non-JSON model list", backend=self.name
            ) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        return [entry["id"] for entry in data if isinstance(entry, dict) and "id" in entry]

    async def close(self) -> None:
        """Close the HTTP client if this backend created it. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"OpenAICompatBackend(name={self.name!r}, base_url={self.base_url!r})"
