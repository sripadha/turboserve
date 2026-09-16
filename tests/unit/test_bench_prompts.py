"""Prompt source tests: exact token counts, shared prefixes, determinism and loaders.

Most of these run against a toy tokenizer whose decode/encode round trip is exact, which
isolates the builder's arithmetic from any one tokenizer's quirks; the last test repeats
the length guarantee against a real cached Qwen2 tokenizer.

The ShareGPT and prompt-file fixtures below are **synthetic** -- written by this test, not
captured from any service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from turboserve.bench.prompts import (
    BenchPrompt,
    PromptError,
    PromptSpec,
    SyntheticPromptBuilder,
    build_prompts,
    load_file_prompts,
    load_sharegpt_prompts,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence


class FakeTokenizer:
    """A whitespace tokenizer whose round trip is exact: id ``n`` renders as ``wn``."""

    vocab_size = 1000
    all_special_ids = (0, 1)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(word[1:]) for word in text.split() if word.startswith("w")]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"w{token}" for token in token_ids)


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


# -- synthetic prompts --------------------------------------------------------------------


def test_prompts_have_exactly_the_requested_token_count(tokenizer: FakeTokenizer) -> None:
    builder = SyntheticPromptBuilder(tokenizer, seed=1)
    for length in (1, 7, 64, 257):
        prompt = builder.build_one(f"p{length}", prompt_tokens=length, max_tokens=4)
        assert prompt.num_prompt_tokens == length
        assert prompt.round_trip_exact
        assert tokenizer.encode(prompt.text) == prompt.token_ids


def test_batch_lengths_fall_inside_the_requested_ranges(tokenizer: FakeTokenizer) -> None:
    builder = SyntheticPromptBuilder(tokenizer, seed=2)
    prompts = builder.build_batch(40, input_tokens=(16, 48), output_tokens=(4, 9))
    assert len(prompts) == 40
    assert all(16 <= prompt.num_prompt_tokens <= 48 for prompt in prompts)
    assert all(4 <= prompt.max_tokens <= 9 for prompt in prompts)
    assert len({prompt.num_prompt_tokens for prompt in prompts}) > 1
    assert len({prompt.prompt_id for prompt in prompts}) == 40


def test_every_prompt_opens_with_the_identical_shared_prefix(tokenizer: FakeTokenizer) -> None:
    builder = SyntheticPromptBuilder(tokenizer, seed=3)
    prompts = builder.build_batch(
        8, input_tokens=(64, 96), output_tokens=(4, 4), shared_prefix_tokens=32
    )
    prefix = prompts[0].token_ids[:32]
    assert all(prompt.token_ids[:32] == prefix for prompt in prompts)
    assert all(prompt.shared_prefix_tokens == 32 for prompt in prompts)
    # The suffixes must differ, or the cache hit rate would be trivially one.
    assert len({tuple(prompt.token_ids[32:]) for prompt in prompts}) == 8
    ids, text = builder.shared_prefix(32)
    assert ids == prefix
    assert tokenizer.encode(text) == prefix


def test_the_same_seed_reproduces_the_workload(tokenizer: FakeTokenizer) -> None:
    kwargs = {"input_tokens": (16, 32), "output_tokens": (2, 6), "shared_prefix_tokens": 8}
    first = SyntheticPromptBuilder(tokenizer, seed=5).build_batch(10, **kwargs)  # type: ignore[arg-type]
    second = SyntheticPromptBuilder(tokenizer, seed=5).build_batch(10, **kwargs)  # type: ignore[arg-type]
    third = SyntheticPromptBuilder(tokenizer, seed=6).build_batch(10, **kwargs)  # type: ignore[arg-type]
    assert [p.token_ids for p in first] == [p.token_ids for p in second]
    assert [p.token_ids for p in first] != [p.token_ids for p in third]

    builder = SyntheticPromptBuilder(tokenizer, seed=5)
    again = builder.build_batch(10, **kwargs)  # type: ignore[arg-type]
    builder.reset()
    assert [p.token_ids for p in builder.build_batch(10, **kwargs)] == [  # type: ignore[arg-type]
        p.token_ids for p in again
    ]


def test_tenants_are_assigned_round_robin(tokenizer: FakeTokenizer) -> None:
    builder = SyntheticPromptBuilder(tokenizer, seed=4)
    prompts = builder.build_batch(
        9, input_tokens=(8, 8), output_tokens=(2, 2), tenants=("a", "b", "c")
    )
    assert [prompt.tenant for prompt in prompts[:4]] == ["a", "b", "c", "a"]
    assert sum(1 for prompt in prompts if prompt.tenant == "a") == 3


def test_builder_exposes_its_seed_and_prompt_summary(tokenizer: FakeTokenizer) -> None:
    builder = SyntheticPromptBuilder(tokenizer, seed=42)
    assert builder.seed == 42
    prompt = builder.build_one("p", prompt_tokens=10, max_tokens=3, shared_prefix_tokens=4)
    assert prompt.to_dict() == {
        "prompt_id": "p",
        "prompt_tokens": 10,
        "max_tokens": 3,
        "tenant": "bench",
        "shared_prefix_tokens": 4,
        "source": "synthetic",
        "round_trip_exact": True,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt_tokens": 0, "max_tokens": 1}, "prompt_tokens"),
        ({"prompt_tokens": 4, "max_tokens": 0}, "max_tokens"),
        ({"prompt_tokens": 4, "max_tokens": 1, "shared_prefix_tokens": 4}, "shared_prefix_tokens"),
    ],
)
def test_build_one_rejects_impossible_requests(
    tokenizer: FakeTokenizer, kwargs: dict[str, int], message: str
) -> None:
    builder = SyntheticPromptBuilder(tokenizer, seed=0)
    with pytest.raises(PromptError, match=message):
        builder.build_one("p", **kwargs)  # type: ignore[arg-type]


def test_build_batch_rejects_impossible_ranges(tokenizer: FakeTokenizer) -> None:
    builder = SyntheticPromptBuilder(tokenizer, seed=0)
    with pytest.raises(PromptError, match="count"):
        builder.build_batch(0, input_tokens=(4, 4), output_tokens=(1, 1))
    with pytest.raises(PromptError, match="input_tokens"):
        builder.build_batch(1, input_tokens=(8, 4), output_tokens=(1, 1))
    with pytest.raises(PromptError, match="output_tokens"):
        builder.build_batch(1, input_tokens=(4, 4), output_tokens=(0, 1))
    with pytest.raises(PromptError, match="tenant"):
        builder.build_batch(1, input_tokens=(4, 4), output_tokens=(1, 1), tenants=())


def test_a_tokenizer_without_a_vocabulary_needs_explicit_ids() -> None:
    class Bare:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1]

        def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
            return "x"

    with pytest.raises(PromptError, match="vocab_size"):
        SyntheticPromptBuilder(Bare())
    builder = SyntheticPromptBuilder(Bare(), vocab_ids=[3, 4, 5])
    assert builder.seed == 0
    with pytest.raises(PromptError, match="empty"):
        SyntheticPromptBuilder(Bare(), vocab_ids=[])


# -- PromptSpec ---------------------------------------------------------------------------


def test_prompt_spec_validates_the_combination(tmp_path: Path) -> None:
    assert PromptSpec(count=4).source == "synthetic"
    with pytest.raises(ValueError, match="takes no path"):
        PromptSpec(count=1, path=tmp_path / "x.txt")
    with pytest.raises(ValueError, match="needs a path"):
        PromptSpec(count=1, source="file")
    with pytest.raises(ValueError, match="shared_prefix_tokens"):
        PromptSpec(count=1, input_tokens=(8, 16), shared_prefix_tokens=8)
    with pytest.raises(ValueError, match="input_tokens"):
        PromptSpec(count=1, input_tokens=(16, 8))
    with pytest.raises(ValueError, match="extra_key"):
        PromptSpec(count=1, extra_key=1)  # type: ignore[call-arg]


def test_build_prompts_dispatches_to_the_synthetic_builder(tokenizer: FakeTokenizer) -> None:
    spec = PromptSpec(count=5, input_tokens=(8, 12), output_tokens=(2, 3), seed=9)
    prompts = build_prompts(spec, tokenizer)
    assert len(prompts) == 5
    assert all(prompt.source == "synthetic" for prompt in prompts)


# -- ShareGPT -----------------------------------------------------------------------------


def write_sharegpt(path: Path, sizes: list[tuple[int, int]]) -> None:
    """Write a synthetic ShareGPT-shaped file with the given (prompt, reply) token counts."""
    entries = [
        {
            "id": f"c{index}",
            "conversations": [
                {"from": "human", "value": " ".join(f"w{100 + n}" for n in range(prompt_len))},
                {"from": "gpt", "value": " ".join(f"w{200 + n}" for n in range(reply_len))},
            ],
        }
        for index, (prompt_len, reply_len) in enumerate(sizes)
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_sharegpt_uses_the_reply_length_as_the_output_budget(
    tmp_path: Path, tokenizer: FakeTokenizer
) -> None:
    path = tmp_path / "sharegpt.json"
    write_sharegpt(path, [(10, 5), (20, 7), (30, 9)])
    prompts = load_sharegpt_prompts(path, tokenizer, count=3)
    assert len(prompts) == 3
    assert {prompt.num_prompt_tokens for prompt in prompts} == {10, 20, 30}
    assert {prompt.max_tokens for prompt in prompts} == {5, 7, 9}
    assert all(prompt.source == "sharegpt" for prompt in prompts)


def test_sharegpt_filters_by_length_and_reports_a_shortfall(
    tmp_path: Path, tokenizer: FakeTokenizer
) -> None:
    path = tmp_path / "sharegpt.json"
    write_sharegpt(path, [(10, 5), (200, 7), (30, 900)])
    prompts = load_sharegpt_prompts(
        path, tokenizer, count=1, input_tokens=(5, 50), output_tokens=(1, 20)
    )
    assert prompts[0].num_prompt_tokens == 10
    with pytest.raises(PromptError, match="usable conversations"):
        load_sharegpt_prompts(path, tokenizer, count=3, input_tokens=(5, 50), output_tokens=(1, 20))


def test_sharegpt_selection_is_seeded(tmp_path: Path, tokenizer: FakeTokenizer) -> None:
    path = tmp_path / "sharegpt.json"
    write_sharegpt(path, [(n, 4) for n in range(5, 45)])
    first = load_sharegpt_prompts(path, tokenizer, count=6, seed=3)
    second = load_sharegpt_prompts(path, tokenizer, count=6, seed=3)
    third = load_sharegpt_prompts(path, tokenizer, count=6, seed=4)
    assert [p.token_ids for p in first] == [p.token_ids for p in second]
    assert [p.token_ids for p in first] != [p.token_ids for p in third]


def test_sharegpt_skips_malformed_entries(tmp_path: Path, tokenizer: FakeTokenizer) -> None:
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": "w1"}]},  # no reply
                "not an object",
                {"conversations": [{"from": "gpt", "value": "w2"}, {"from": "gpt", "value": "w3"}]},
                {
                    "conversations": [
                        {"from": "human", "value": "w1 w2"},
                        {"from": "gpt", "value": "w3"},
                    ]
                },
            ]
        ),
        encoding="utf-8",
    )
    prompts = load_sharegpt_prompts(path, tokenizer, count=1)
    assert prompts[0].num_prompt_tokens == 2
    not_an_array = tmp_path / "object.json"
    not_an_array.write_text(json.dumps({"conversations": []}), encoding="utf-8")
    with pytest.raises(PromptError, match="JSON array"):
        load_sharegpt_prompts(not_an_array, tokenizer, count=1)
    with pytest.raises(PromptError, match="does not exist"):
        load_sharegpt_prompts(tmp_path / "bad.json", tokenizer, count=1)


# -- prompt files -------------------------------------------------------------------------


def test_text_file_prompts_one_per_line(tmp_path: Path, tokenizer: FakeTokenizer) -> None:
    path = tmp_path / "prompts.txt"
    path.write_text("w1 w2\n\nw3 w4 w5\n", encoding="utf-8")
    prompts = load_file_prompts(path, tokenizer, default_max_tokens=17)
    assert [prompt.num_prompt_tokens for prompt in prompts] == [2, 3]
    assert all(prompt.max_tokens == 17 for prompt in prompts)
    assert all(prompt.source == "file" for prompt in prompts)


def test_jsonl_and_json_objects_carry_their_own_settings(
    tmp_path: Path, tokenizer: FakeTokenizer
) -> None:
    jsonl = tmp_path / "prompts.jsonl"
    jsonl.write_text(
        '{"prompt": "w1 w2", "max_tokens": 5, "tenant": "t9", "prompt_id": "given", "x": 1}\n'
        '{"prompt": "w3"}\n',
        encoding="utf-8",
    )
    prompts = load_file_prompts(jsonl, tokenizer, default_max_tokens=3)
    assert prompts[0].prompt_id == "given"
    assert prompts[0].tenant == "t9"
    assert prompts[0].max_tokens == 5
    assert prompts[1].max_tokens == 3

    as_json = tmp_path / "prompts.json"
    as_json.write_text(json.dumps(["w1", {"prompt": "w2 w3"}]), encoding="utf-8")
    assert len(load_file_prompts(as_json, tokenizer)) == 2


def test_file_prompt_errors(tmp_path: Path, tokenizer: FakeTokenizer) -> None:
    with pytest.raises(PromptError, match="does not exist"):
        load_file_prompts(tmp_path / "missing.txt", tokenizer)
    bad_suffix = tmp_path / "prompts.csv"
    bad_suffix.write_text("w1", encoding="utf-8")
    with pytest.raises(PromptError, match="unsupported prompt file suffix"):
        load_file_prompts(bad_suffix, tokenizer)
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(PromptError, match="no prompts"):
        load_file_prompts(empty, tokenizer)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{nope}\n", encoding="utf-8")
    with pytest.raises(PromptError, match="not valid JSON"):
        load_file_prompts(broken, tokenizer)
    no_prompt = tmp_path / "no_prompt.jsonl"
    no_prompt.write_text('{"max_tokens": 2}\n', encoding="utf-8")
    with pytest.raises(PromptError, match="no non-empty 'prompt'"):
        load_file_prompts(no_prompt, tokenizer)
    too_few = tmp_path / "few.txt"
    too_few.write_text("w1\n", encoding="utf-8")
    with pytest.raises(PromptError, match="needed 5"):
        load_file_prompts(too_few, tokenizer, count=5)


def test_build_prompts_reads_a_file_source(tmp_path: Path, tokenizer: FakeTokenizer) -> None:
    path = tmp_path / "prompts.txt"
    path.write_text("w1 w2\nw3 w4\n", encoding="utf-8")
    spec = PromptSpec(source="file", count=2, output_tokens=(1, 6), path=path)
    prompts = build_prompts(spec, tokenizer)
    assert [prompt.max_tokens for prompt in prompts] == [6, 6]


# -- against a real tokenizer -------------------------------------------------------------


def test_exact_lengths_hold_for_a_real_qwen2_tokenizer(tiny_qwen2_path: Path) -> None:
    """The length guarantee is about token ids, which is what the engine consumes."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_qwen2_path))
    builder = SyntheticPromptBuilder(tokenizer, seed=13)
    prompts = builder.build_batch(
        6, input_tokens=(48, 64), output_tokens=(4, 8), shared_prefix_tokens=16
    )
    assert all(48 <= prompt.num_prompt_tokens <= 64 for prompt in prompts)
    prefix = prompts[0].token_ids[:16]
    assert all(prompt.token_ids[:16] == prefix for prompt in prompts)
    assert all(prompt.round_trip_exact for prompt in prompts)
    assert isinstance(prompts[0], BenchPrompt)
