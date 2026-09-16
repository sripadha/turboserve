"""The OpenAI HTTP surface: request bodies, response bodies and streaming chunks.

The point of speaking OpenAI's dialect is that nothing has to be written twice -- the
benchmark harness, the ``openai`` Python client, vLLM's own client tooling and a Grafana
Loki query all already understand it. So the shapes here follow the published wire format
exactly, including the field names that look redundant (``object``, ``created``,
``index``), because clients validate them.

Two policies are worth stating up front:

* **Unknown fields are accepted, unsupported ones are refused.** OpenAI clients send fields
  this gateway has no engine for (``tools``, ``logit_bias``, ``best_of``). Silently ignoring
  them would produce a plausible but wrong answer -- a client that asked for a tool call and
  got prose has been lied to. So a short list of fields with *semantic* consequences is
  rejected with a 400, and everything else unknown is ignored.
* **Sampling knobs beyond OpenAI's set are first-class.** ``top_k``, ``repetition_penalty``
  and ``ignore_eos`` are how the benchmark scenarios pin down decode behaviour, and they are
  what vLLM accepts as ``extra_body``. They are declared here rather than smuggled through,
  so they are validated like everything else.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from turboserve.engine.core.types import FinishReason, SamplingParams

__all__ = [
    "SSE_DONE",
    "UNSUPPORTED_FIELDS",
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionDelta",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "ChatMessageContentPart",
    "CompletionChoice",
    "CompletionRequest",
    "CompletionResponse",
    "ErrorInfo",
    "ErrorResponse",
    "ModelCard",
    "ModelList",
    "StreamOptions",
    "UsageInfo",
    "created_timestamp",
    "new_chat_completion_id",
    "new_completion_id",
    "openai_finish_reason",
]

#: The sentinel payload that terminates an OpenAI SSE stream. The gateway emits it as
#: ``data: [DONE]\n\n``; every OpenAI client stops on it rather than on connection close.
SSE_DONE = "[DONE]"

#: Request fields that change what the answer *means* and that this gateway does not
#: implement. Accepting them silently would be the dishonest option, so they are a 400.
UNSUPPORTED_FIELDS: tuple[str, ...] = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "logit_bias",
    "best_of",
    "echo",
    "suffix",
)


def created_timestamp() -> int:
    """Unix seconds for the ``created`` field of a response."""
    return int(time.time())


def new_completion_id() -> str:
    """Identifier for a ``/v1/completions`` response, in OpenAI's ``cmpl-`` form."""
    return f"cmpl-{uuid.uuid4().hex}"


def new_chat_completion_id() -> str:
    """Identifier for a ``/v1/chat/completions`` response, in OpenAI's ``chatcmpl-`` form."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def openai_finish_reason(reason: FinishReason | str | None) -> str | None:
    """Map an engine finish reason onto the vocabulary OpenAI clients accept.

    ``abort`` has no OpenAI equivalent and is reported as ``stop``: clients validate this
    field against a closed set, and an unknown value makes strict ones raise rather than
    surface the partial completion they already hold. A request that aborted is separately
    visible as a failure in ``requests_total`` and in the usage record.
    """
    if reason is None:
        return None
    value = str(reason)
    if value == str(FinishReason.LENGTH):
        return "length"
    return "stop"


# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------


class ChatMessageContentPart(BaseModel):
    """One element of the multi-part ``content`` array."""

    model_config = ConfigDict(extra="allow")

    type: str = "text"
    text: str | None = None


class ChatMessage(BaseModel):
    """One turn of a conversation."""

    model_config = ConfigDict(extra="allow")

    role: str = Field(min_length=1)
    content: str | list[ChatMessageContentPart] | None = None
    name: str | None = None


class StreamOptions(BaseModel):
    """OpenAI's ``stream_options``; only ``include_usage`` is meaningful here."""

    model_config = ConfigDict(extra="ignore")

    include_usage: bool = False


class _GenerationRequest(BaseModel):
    """Fields shared by the completions and chat-completions bodies.

    ``extra="allow"`` keeps real clients working, and the validator below turns the subset
    of extras that would change the answer into a 400.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    user: str | None = None
    logprobs: bool | None = None

    # turboserve extensions, matching what vLLM accepts as extra_body
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    ignore_eos: bool = False
    stop_token_ids: list[int] = Field(default_factory=list)
    lora: str | None = None
    """Adapter name as the tenant knows it; resolved against the tenant's adapter map."""

    priority: int | None = None
    """Preemption priority passed through to the engine.

    It does *not* reorder admission: the engine admits by arrival order (or by tenant fair
    share). When the KV pool is full the lowest-priority running sequence is the one
    preempted, so a higher number means a request is preempted last.
    """

    @model_validator(mode="after")
    def _reject_unsupported(self) -> _GenerationRequest:
        extras = self.model_extra or {}
        present = [
            field for field in UNSUPPORTED_FIELDS if extras.get(field) not in (None, [], {}, False)
        ]
        if present:
            raise ValueError(f"unsupported field(s) for this gateway: {', '.join(sorted(present))}")
        if self.n != 1:
            raise ValueError("only n=1 is supported; issue separate requests for more samples")
        return self

    @property
    def include_usage(self) -> bool:
        """Whether the client asked for a final usage-carrying chunk."""
        return bool(self.stream_options and self.stream_options.include_usage)

    def stop_sequences(self) -> list[str]:
        """``stop`` normalised to a list."""
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)

    def sampling_params(self, *, default_max_tokens: int = 128) -> SamplingParams:
        """Build the engine's :class:`SamplingParams` from this body.

        ``max_tokens`` is optional in OpenAI's schema but mandatory for a server that must
        bound its own work, so an unset value takes the gateway's configured default rather
        than being unbounded.
        """
        return SamplingParams(
            max_tokens=self.max_tokens if self.max_tokens is not None else default_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            stop=self.stop_sequences(),
            stop_token_ids=list(self.stop_token_ids),
            repetition_penalty=self.repetition_penalty,
            ignore_eos=self.ignore_eos,
            logprobs=bool(self.logprobs),
        )


class CompletionRequest(_GenerationRequest):
    """Body of ``POST /v1/completions``."""

    prompt: str | list[str] | list[int] | list[list[int]]

    @model_validator(mode="after")
    def _prompt_is_a_single_non_empty_prompt(self) -> CompletionRequest:
        """Reject batched and empty prompts.

        OpenAI's ``prompt`` may be a batch. This gateway serves one completion per request
        so that a 429, a retry and a usage record all describe one thing; a client wanting a
        batch issues a batch of requests, which is also what gets it real concurrency.
        """
        prompt = self.prompt
        if isinstance(prompt, str):
            if not prompt:
                raise ValueError("prompt must not be empty")
            return self
        if len(prompt) == 0:
            raise ValueError("prompt must not be empty")
        if len(prompt) > 1 and not all(isinstance(item, int) for item in prompt):
            raise ValueError("batched prompts are not supported; send one request per prompt")
        return self

    def single_prompt(self) -> str | list[int]:
        """The one prompt this request carries, as text or as token ids."""
        prompt = self.prompt
        if isinstance(prompt, str):
            return prompt
        first = prompt[0]
        if isinstance(first, str):
            return first
        if isinstance(first, int):
            return [token for token in prompt if isinstance(token, int)]
        return list(first)


class ChatCompletionRequest(_GenerationRequest):
    """Body of ``POST /v1/chat/completions``."""

    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    """OpenAI's newer name for ``max_tokens``; it wins when both are present."""

    add_generation_prompt: bool = True
    """Whether the chat template appends the assistant header. Off is useful for prefill
    experiments that want the raw conversation."""

    def sampling_params(self, *, default_max_tokens: int = 128) -> SamplingParams:
        """As the base method, honouring ``max_completion_tokens`` first."""
        params = super().sampling_params(default_max_tokens=default_max_tokens)
        if self.max_completion_tokens is not None:
            params.max_tokens = self.max_completion_tokens
        return params


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------


class UsageInfo(BaseModel):
    """Token accounting attached to a response or a final streaming chunk."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: dict[str, Any] | None = None
    """``{"cached_tokens": n}`` when the engine's prefix cache served part of the prompt."""


class CompletionChoice(BaseModel):
    """One completion in a ``/v1/completions`` response."""

    index: int = 0
    text: str = ""
    finish_reason: str | None = None
    logprobs: None = None


class CompletionResponse(BaseModel):
    """Body of a non-streaming ``/v1/completions`` response, and of each stream chunk."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=created_timestamp)
    model: str
    choices: list[CompletionChoice] = Field(default_factory=list)
    usage: UsageInfo | None = None


class ChatCompletionMessage(BaseModel):
    """The assistant turn a chat completion produced."""

    role: str = "assistant"
    content: str = ""


class ChatCompletionChoice(BaseModel):
    """One choice in a non-streaming chat response."""

    index: int = 0
    message: ChatCompletionMessage = Field(default_factory=ChatCompletionMessage)
    finish_reason: str | None = None
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    """Body of a non-streaming ``/v1/chat/completions`` response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=created_timestamp)
    model: str
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: UsageInfo | None = None


class ChatCompletionDelta(BaseModel):
    """The incremental part of a streaming chat chunk.

    ``role`` appears only on the first chunk and ``content`` only on chunks that carry text,
    which is what OpenAI does and what clients assume when they concatenate.

    The custom serialiser drops unset members of *this* object only. A blanket
    ``exclude_none`` on the enclosing chunk would also delete ``finish_reason: null``, which
    OpenAI sends on every non-final chunk and which strict clients read as a required key.
    """

    role: str | None = None
    content: str | None = None

    @model_serializer
    def _only_what_is_set(self) -> dict[str, str]:
        """Serialise to the members that carry a value, in OpenAI's order."""
        data: dict[str, str] = {}
        if self.role is not None:
            data["role"] = self.role
        if self.content is not None:
            data["content"] = self.content
        return data


class ChatCompletionChunkChoice(BaseModel):
    """One choice inside a streaming chat chunk."""

    index: int = 0
    delta: ChatCompletionDelta = Field(default_factory=ChatCompletionDelta)
    finish_reason: str | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    """One ``data:`` payload of a streaming chat completion."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=created_timestamp)
    model: str
    choices: list[ChatCompletionChunkChoice] = Field(default_factory=list)
    usage: UsageInfo | None = None
    """Present only on the extra final chunk requested via ``stream_options.include_usage``."""


class ModelCard(BaseModel):
    """One entry of ``GET /v1/models``."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=created_timestamp)
    owned_by: str = "turboserve"
    root: str | None = None


class ModelList(BaseModel):
    """Body of ``GET /v1/models``."""

    object: Literal["list"] = "list"
    data: list[ModelCard] = Field(default_factory=list)


class ErrorInfo(BaseModel):
    """The inner object of an OpenAI error body."""

    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """OpenAI's error envelope, which clients unwrap to build their exceptions."""

    error: ErrorInfo

    @classmethod
    def of(
        cls,
        message: str,
        *,
        type_: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ) -> ErrorResponse:
        """Build an error body in one call."""
        return cls(error=ErrorInfo(message=message, type=type_, code=code, param=param))
