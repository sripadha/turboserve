"""Turning a list of chat messages into the string a base model actually sees.

``POST /v1/chat/completions`` speaks messages; a causal language model consumes one token
sequence. The mapping between them is *model-specific* -- Qwen2 wants ChatML
(``<|im_start|>role\\n...<|im_end|>``), Llama wants its own header tokens -- and getting it
wrong does not raise: it silently produces a model that rambles, never stops, or ignores
the system prompt. So the tokenizer's own ``chat_template`` is always preferred, because it
ships with the checkpoint and is by definition the one the model was tuned on.

When there is no tokenizer to ask -- the gateway fronts a remote server, or the checkpoint
has no template -- a neutral role-labelled fallback is used. The fallback is honest rather
than clever: it does not pretend to be any model's real template, and
:attr:`ChatTemplate.uses_tokenizer` tells callers (and the tests) which path ran.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "FALLBACK_ROLE_LABELS",
    "ChatTemplate",
    "ChatTemplateCache",
    "ChatTemplateError",
    "content_to_text",
    "fallback_chat_prompt",
    "normalise_messages",
]

#: Prefixes the fallback template puts in front of each turn. Plain English words rather
#: than special tokens: a special token that is not in the model's vocabulary would be
#: split into pieces and confuse it, while ``User:``/``Assistant:`` is understood by every
#: instruction-tuned checkpoint to some degree.
FALLBACK_ROLE_LABELS: Final[dict[str, str]] = {
    "system": "System",
    "developer": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
    "function": "Tool",
}

_FALLBACK_GENERATION_PREFIX: Final = "Assistant:"


class ChatTemplateError(ValueError):
    """The messages cannot be rendered (empty list, unknown role, unusable content)."""


def content_to_text(content: object) -> str:
    """Flatten OpenAI message content into plain text.

    Content may be a string, ``None`` (an assistant turn that only carried tool calls), or
    the multi-part array form ``[{"type": "text", "text": ...}, ...]``. Non-text parts are
    dropped rather than rendered as placeholders: this gateway serves text models, and a
    fake ``[image]`` marker in the prompt would be a silent lie to the model.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for part in content:
            text = _part_text(part)
            if text:
                parts.append(text)
        return "".join(parts)
    raise ChatTemplateError(f"unsupported message content of type {type(content).__name__}")


def _part_text(part: object) -> str:
    """Text of one content part, or ``""`` when the part carries none."""
    if isinstance(part, str):
        return part
    getter = getattr(part, "get", None)
    if callable(getter):
        value = getter("text")
        return value if isinstance(value, str) else ""
    value = getattr(part, "text", None)
    return value if isinstance(value, str) else ""


def normalise_messages(messages: Iterable[Any]) -> list[dict[str, str]]:
    """Coerce pydantic message models or mappings into ``{"role", "content"}`` dicts.

    Both shapes reach this module: the route hands over validated
    :class:`~turboserve.gateway.openai_types.ChatMessage` objects, while the chat template
    of a tokenizer wants plain dictionaries. Doing the conversion once, here, is what keeps
    :mod:`turboserve.gateway.openai_types` free of any tokenizer knowledge.
    """
    out: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, Mapping):
            role = message.get("role")
            content = message.get("content")
            name = message.get("name")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            name = getattr(message, "name", None)
        if not isinstance(role, str) or not role:
            raise ChatTemplateError("every message needs a non-empty 'role'")
        entry = {"role": role, "content": content_to_text(content)}
        if isinstance(name, str) and name:
            entry["name"] = name
        out.append(entry)
    if not out:
        raise ChatTemplateError("at least one message is required")
    return out


def fallback_chat_prompt(
    messages: Iterable[Any],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render messages with the neutral role-labelled template.

    Used when the tokenizer has no ``chat_template``, or when the gateway has no tokenizer
    at all. An unknown role is labelled by its capitalised name rather than rejected, so a
    client using a role this gateway has not heard of still gets a coherent prompt.
    """
    normalised = normalise_messages(messages)
    lines: list[str] = []
    for message in normalised:
        label = FALLBACK_ROLE_LABELS.get(message["role"], message["role"].capitalize())
        name = message.get("name")
        header = f"{label} ({name}):" if name else f"{label}:"
        lines.append(f"{header} {message['content']}".rstrip())
    rendered = "\n".join(lines)
    if add_generation_prompt:
        rendered = f"{rendered}\n{_FALLBACK_GENERATION_PREFIX}"
    return rendered


class ChatTemplate:
    """Renders chat messages for one model, via its tokenizer when there is one.

    Immutable and cheap to copy; the expensive part (loading a tokenizer) happens once in
    :meth:`load` and is shared through :class:`ChatTemplateCache`.
    """

    __slots__ = ("_source", "_tokenizer")

    def __init__(self, tokenizer: Any | None = None, *, source: str = "fallback") -> None:
        self._tokenizer = tokenizer
        self._source = source if tokenizer is None else source

    @property
    def uses_tokenizer(self) -> bool:
        """Whether rendering goes through a tokenizer's own ``chat_template``."""
        return self._tokenizer is not None

    @property
    def source(self) -> str:
        """Where the template came from: ``"fallback"`` or the model id that supplied it."""
        return self._source

    @property
    def tokenizer(self) -> Any | None:
        """The underlying tokenizer, or ``None``."""
        return self._tokenizer

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, *, source: str = "tokenizer") -> ChatTemplate:
        """Wrap a tokenizer, falling back when it carries no chat template.

        A tokenizer without ``chat_template`` is not an error -- base (non-instruct)
        checkpoints have none -- but calling ``apply_chat_template`` on one raises or, worse
        in some versions, silently applies a default template. Checking up front makes the
        choice explicit and observable.
        """
        if getattr(tokenizer, "chat_template", None):
            return cls(tokenizer, source=source)
        logger.debug("tokenizer %s has no chat_template; using the fallback", source)
        return cls(None, source="fallback")

    @classmethod
    def load(
        cls,
        model: str,
        *,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
    ) -> ChatTemplate:
        """Load a tokenizer for ``model`` and wrap it, or return the fallback.

        Never raises. A gateway fronting a remote vLLM server usually has no local copy of
        the checkpoint, and refusing to start for that reason would be wrong; the fallback
        template keeps ``/v1/chat/completions`` working and the debug log says which path
        was taken. ``local_files_only`` defaults to ``True`` so a server start never turns
        into a multi-gigabyte download.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:  # pragma: no cover - transformers is a hard dependency
            logger.debug("transformers is not installed; using the fallback chat template")
            return cls(None)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError, ImportError) as exc:
            logger.debug("no local tokenizer for %s (%s); using the fallback", model, exc)
            return cls(None)
        return cls.from_tokenizer(tokenizer, source=model)

    def render(self, messages: Iterable[Any], *, add_generation_prompt: bool = True) -> str:
        """Render messages to the prompt string the model should complete.

        A tokenizer whose template raises at render time (a template that requires a field
        this gateway does not send) degrades to the fallback rather than failing the
        request: a slightly off prompt beats a 500.
        """
        normalised = normalise_messages(messages)
        if self._tokenizer is None:
            return fallback_chat_prompt(normalised, add_generation_prompt=add_generation_prompt)
        try:
            rendered = self._tokenizer.apply_chat_template(
                normalised,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning(
                "chat template of %s failed to render (%s); falling back", self._source, exc
            )
            return fallback_chat_prompt(normalised, add_generation_prompt=add_generation_prompt)
        if not isinstance(rendered, str):  # tokenize=False must give a string
            return fallback_chat_prompt(normalised, add_generation_prompt=add_generation_prompt)
        return rendered

    def count_tokens(self, text: str) -> int | None:
        """Exact token count for ``text``, or ``None`` when there is no tokenizer.

        ``None`` rather than an estimate, so the caller decides whether to estimate and can
        mark the resulting usage record as estimated.
        """
        if self._tokenizer is None:
            return None
        try:
            ids = self._tokenizer.encode(text, add_special_tokens=False)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            logger.debug("tokenizer %s failed to encode (%s)", self._source, exc)
            return None
        return len(ids)

    def __repr__(self) -> str:
        return f"ChatTemplate(source={self._source!r}, uses_tokenizer={self.uses_tokenizer})"


class ChatTemplateCache:
    """One :class:`ChatTemplate` per model name, loaded on first use.

    The gateway may serve several models, and loading a tokenizer costs hundreds of
    milliseconds -- acceptable once per model at first request, not once per request. Misses
    are cached too (as the fallback), so a model with no local tokenizer is not retried on
    every call.
    """

    __slots__ = ("_local_files_only", "_templates", "_trust_remote_code")

    def __init__(
        self,
        *,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        self._templates: dict[str, ChatTemplate] = {}
        self._local_files_only = local_files_only
        self._trust_remote_code = trust_remote_code

    def get(self, model: str) -> ChatTemplate:
        """Template for ``model``, loading it if this is the first request for it."""
        template = self._templates.get(model)
        if template is None:
            template = ChatTemplate.load(
                model,
                local_files_only=self._local_files_only,
                trust_remote_code=self._trust_remote_code,
            )
            self._templates[model] = template
            logger.info(
                "chat template for %s: %s",
                model,
                "tokenizer" if template.uses_tokenizer else "fallback",
            )
        return template

    def put(self, model: str, template: ChatTemplate) -> None:
        """Install a template explicitly; used by tests and by the in-process engine path."""
        self._templates[model] = template

    def __contains__(self, model: object) -> bool:
        return model in self._templates

    def __len__(self) -> int:
        return len(self._templates)
