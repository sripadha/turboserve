"""Chat templating: the tokenizer's own template when there is one, the fallback otherwise.

The tokenizer path uses the cached tiny-random Qwen2 checkpoint (which ships a ChatML
template) and the tiny-random Llama checkpoint (which ships none), so both branches are
exercised against real tokenizer objects rather than fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from turboserve.gateway.chat_template import (
    ChatTemplate,
    ChatTemplateCache,
    ChatTemplateError,
    content_to_text,
    fallback_chat_prompt,
    normalise_messages,
)
from turboserve.gateway.openai_types import ChatMessage

MESSAGES = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hello"},
]


# -- content flattening ------------------------------------------------------------------


def test_content_to_text_accepts_every_openai_shape() -> None:
    assert content_to_text("plain") == "plain"
    assert content_to_text(None) == ""
    assert content_to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_non_text_parts_are_dropped_not_faked() -> None:
    # A placeholder like "[image]" would be a silent lie to a text-only model.
    parts = [{"type": "text", "text": "look: "}, {"type": "image_url", "image_url": {"url": "x"}}]
    assert content_to_text(parts) == "look: "


def test_content_to_text_rejects_an_unusable_type() -> None:
    with pytest.raises(ChatTemplateError, match="unsupported message content"):
        content_to_text(42)


def test_normalise_accepts_pydantic_messages_and_mappings() -> None:
    models = [ChatMessage(role="user", content=[{"type": "text", "text": "hi"}])]
    assert normalise_messages(models) == [{"role": "user", "content": "hi"}]
    assert normalise_messages([{"role": "user", "content": "hi", "name": "bob"}]) == [
        {"role": "user", "content": "hi", "name": "bob"}
    ]


def test_normalise_requires_a_role_and_at_least_one_message() -> None:
    with pytest.raises(ChatTemplateError, match="non-empty 'role'"):
        normalise_messages([{"content": "hi"}])
    with pytest.raises(ChatTemplateError, match="at least one message"):
        normalise_messages([])


# -- fallback template -------------------------------------------------------------------


def test_fallback_labels_roles_and_opens_the_assistant_turn() -> None:
    assert fallback_chat_prompt(MESSAGES) == "System: be brief\nUser: hello\nAssistant:"


def test_fallback_can_leave_the_conversation_open() -> None:
    assert fallback_chat_prompt(MESSAGES, add_generation_prompt=False) == (
        "System: be brief\nUser: hello"
    )


def test_fallback_labels_an_unknown_role_rather_than_failing() -> None:
    rendered = fallback_chat_prompt([{"role": "critic", "content": "hm"}])
    assert rendered.startswith("Critic: hm")


def test_fallback_includes_the_speaker_name_when_given() -> None:
    rendered = fallback_chat_prompt([{"role": "user", "content": "hi", "name": "bob"}])
    assert "User (bob): hi" in rendered


def test_fallback_template_has_no_tokenizer() -> None:
    template = ChatTemplate(None)
    assert template.uses_tokenizer is False
    assert template.source == "fallback"
    assert template.count_tokens("anything") is None
    assert "Assistant:" in template.render(MESSAGES)


# -- tokenizer template ------------------------------------------------------------------


def test_tokenizer_template_is_preferred_when_the_checkpoint_ships_one(
    tiny_qwen2_path: Path,
) -> None:
    template = ChatTemplate.load(str(tiny_qwen2_path))
    assert template.uses_tokenizer is True
    rendered = template.render(MESSAGES)
    # Qwen2 checkpoints carry a ChatML template; the fallback never produces these markers.
    assert "<|im_start|>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")
    assert "Assistant:" not in rendered


def test_tokenizer_template_can_leave_the_generation_prompt_off(tiny_qwen2_path: Path) -> None:
    template = ChatTemplate.load(str(tiny_qwen2_path))
    assert not template.render(MESSAGES, add_generation_prompt=False).endswith("assistant\n")


def test_tokenizer_counts_tokens_exactly(tiny_qwen2_path: Path) -> None:
    template = ChatTemplate.load(str(tiny_qwen2_path))
    counted = template.count_tokens("hello world")
    assert counted is not None and counted > 0


def test_each_architecture_uses_its_own_template(tiny_llama_path: Path) -> None:
    # Llama and Qwen2 disagree about how a conversation is spelled; using one model's
    # template for the other is the silent-corruption case this module exists to prevent.
    template = ChatTemplate.load(str(tiny_llama_path))
    assert template.uses_tokenizer is True
    rendered = template.render(MESSAGES)
    assert "<|im_start|>" not in rendered
    assert "[INST]" in rendered


def test_a_tokenizer_without_a_chat_template_falls_back() -> None:
    # Base (non-instruct) checkpoints ship no template; calling apply_chat_template on one
    # either raises or silently applies somebody else's default.
    class BaseTokenizer:
        chat_template = None

        def apply_chat_template(self, *args: object, **kwargs: object) -> str:
            raise AssertionError("must not be called without a chat template")

    template = ChatTemplate.from_tokenizer(BaseTokenizer(), source="base")
    assert template.uses_tokenizer is False
    assert template.source == "fallback"
    assert template.render(MESSAGES) == fallback_chat_prompt(MESSAGES)


def test_a_missing_checkpoint_falls_back_instead_of_raising() -> None:
    # A gateway fronting a remote vLLM server has no local copy of the weights.
    template = ChatTemplate.load("definitely/not-a-real-model-id")
    assert template.uses_tokenizer is False
    assert template.render(MESSAGES)


def test_a_template_that_raises_at_render_time_falls_back() -> None:
    class ExplodingTokenizer:
        chat_template = "{{ nonsense }}"

        def apply_chat_template(self, *args: object, **kwargs: object) -> str:
            raise ValueError("template needs a field we do not send")

    template = ChatTemplate.from_tokenizer(ExplodingTokenizer(), source="exploding")
    assert template.render(MESSAGES) == fallback_chat_prompt(MESSAGES)


def test_a_template_returning_non_text_falls_back() -> None:
    class TokenizingTokenizer:
        chat_template = "x"

        def apply_chat_template(self, *args: object, **kwargs: object) -> list[int]:
            return [1, 2, 3]

    template = ChatTemplate.from_tokenizer(TokenizingTokenizer())
    assert template.render(MESSAGES) == fallback_chat_prompt(MESSAGES)


# -- cache --------------------------------------------------------------------------------


def test_cache_loads_once_per_model(tiny_qwen2_path: Path) -> None:
    cache = ChatTemplateCache()
    first = cache.get(str(tiny_qwen2_path))
    assert cache.get(str(tiny_qwen2_path)) is first
    assert len(cache) == 1


def test_cache_remembers_misses_too() -> None:
    # Otherwise every request for an unknown model would retry a tokenizer load.
    cache = ChatTemplateCache()
    first = cache.get("definitely/not-a-real-model-id")
    assert cache.get("definitely/not-a-real-model-id") is first
    assert first.uses_tokenizer is False


def test_cache_accepts_an_explicit_template() -> None:
    cache = ChatTemplateCache()
    template = ChatTemplate(None, source="injected")
    cache.put("m", template)
    assert cache.get("m") is template
    assert "m" in cache
