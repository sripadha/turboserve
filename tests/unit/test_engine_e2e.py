"""End-to-end tests for the synchronous engine, on CPU with a cached tiny-random model.

The contract being pinned here is the one everything else depends on: whatever the
scheduler, the paged KV cache and the prefix cache do, the tokens must be the tokens
``transformers`` would have produced. Every correctness test therefore compares against HF
greedy decoding on the same checkpoint rather than against a stored expectation, so a change
in the tiny model cannot silently invalidate the test.

The memory-sizing and detokenizer units are exercised here too, because they are part of the
same runtime package and neither needs a model of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from turboserve.engine.core.types import EngineConfig, SamplingParams, SchedulerConfig
from turboserve.engine.model.model_config import ModelConfig
from turboserve.engine.runtime.engine import LLMEngine
from turboserve.engine.runtime.memory import (
    KVCacheSizing,
    MemoryProbe,
    MemoryProfileError,
    activation_headroom_bytes,
    probe_memory,
    size_kv_cache,
)
from turboserve.engine.runtime.streaming import (
    IncrementalDetokenizer,
    StopStringMatcher,
    StreamingDecoder,
)

PROMPTS: tuple[str, ...] = (
    "Hello world",
    "The quick brown fox jumps over",
    "A B C D E F G H I J K L",
    "In a distant galaxy there was",
    "def add(a, b):",
    "Once upon a time",
    "Paris is the capital of",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14",
)


def _engine_config(model_path: Path, **scheduler: Any) -> EngineConfig:
    """An all-CPU fp32 config with an explicit block count (no memory profiling in tests)."""
    options: dict[str, Any] = {
        "max_num_seqs": 8,
        "max_num_batched_tokens": 256,
        "block_size": 16,
        "num_blocks": 64,
    }
    options.update(scheduler)
    return EngineConfig(
        model=str(model_path),
        device="cpu",
        dtype="float32",
        scheduler=SchedulerConfig(**options),
    )


@pytest.fixture(scope="module")
def hf_reference(tiny_qwen2_path: Path) -> Any:
    """Greedy continuations from ``transformers`` for :data:`PROMPTS`, decoded one by one.

    One request per ``generate`` call: batching in ``transformers`` needs padding, and a
    padded batch is a different computation from an unpadded one in the last bits of fp32.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_qwen2_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(tiny_qwen2_path), dtype=torch.float32, local_files_only=True
    ).eval()
    eos = tokenizer.eos_token_id
    reference: dict[str, list[int]] = {}
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            generated = model.generate(
                torch.tensor([ids], dtype=torch.long),
                max_new_tokens=16,
                do_sample=False,
                use_cache=True,
                pad_token_id=eos,
                eos_token_id=eos,
            )
            reference[prompt] = generated[0, len(ids) :].tolist()
    del model
    return reference


# ----------------------------------------------------------------------------------------
# greedy parity
# ----------------------------------------------------------------------------------------


def test_batch_of_eight_matches_hf_greedy(tiny_qwen2_path: Path, hf_reference: Any) -> None:
    """Eight mixed-length prompts in one continuous batch equal HF greedy, one by one.

    This is the whole point of the engine: prefill chunks and decode steps for unrelated
    sequences share a forward pass, and the paged attention still gives each sequence exactly
    the context it would have had alone.
    """
    with LLMEngine(_engine_config(tiny_qwen2_path), local_files_only=True) as engine:
        outputs = engine.generate(list(PROMPTS), SamplingParams(max_tokens=16, temperature=0.0))
        assert len(outputs) == len(PROMPTS)
        for prompt, output in zip(PROMPTS, outputs, strict=True):
            assert output.new_token_ids == hf_reference[prompt], prompt
            assert output.finished
            assert output.output_tokens == len(output.new_token_ids)


def test_chunked_prefill_does_not_change_the_tokens(
    tiny_qwen2_path: Path, hf_reference: Any
) -> None:
    """A token budget far below the prompt length splits prefill and changes nothing."""
    config = _engine_config(tiny_qwen2_path, max_num_batched_tokens=16, max_num_seqs=4)
    with LLMEngine(config, local_files_only=True) as engine:
        outputs = engine.generate(list(PROMPTS), SamplingParams(max_tokens=16, temperature=0.0))
        for prompt, output in zip(PROMPTS, outputs, strict=True):
            assert output.new_token_ids == hf_reference[prompt], prompt


def test_stats_report_the_pool_and_the_work(tiny_qwen2_path: Path) -> None:
    """``stats()`` is flat, JSON-safe and reports both the KV pool and the token counters."""
    with LLMEngine(_engine_config(tiny_qwen2_path), local_files_only=True) as engine:
        engine.generate(list(PROMPTS[:2]), SamplingParams(max_tokens=8, temperature=0.0))
        stats = engine.stats()
    assert stats["num_kv_blocks"] == 64
    assert stats["block_size"] == 16
    assert stats["num_finished"] == 2
    assert stats["num_generated_tokens"] == 16
    assert stats["device"] == "cpu"
    assert stats["dtype"] == "float32"
    assert all(isinstance(value, int | float | str) for value in stats.values())


# ----------------------------------------------------------------------------------------
# preemption
# ----------------------------------------------------------------------------------------


def test_preemption_under_a_tiny_pool_preserves_outputs(
    tiny_qwen2_path: Path, hf_reference: Any
) -> None:
    """A KV pool too small for the working set preempts, recomputes and still agrees with HF.

    Recompute preemption is only safe if a resumed sequence re-derives byte-identical KV for
    the tokens it lost. With eight sequences and eight blocks the scheduler is forced to
    preempt repeatedly, so this exercises the path many times in one run.
    """
    config = _engine_config(tiny_qwen2_path, num_blocks=8, block_size=16, max_num_seqs=8)
    with LLMEngine(config, local_files_only=True) as engine:
        outputs = engine.generate(list(PROMPTS), SamplingParams(max_tokens=16, temperature=0.0))
        stats = engine.stats()
        engine.scheduler.check_invariants()
    assert stats["num_preemptions"] > 0, "the pool was not small enough to force preemption"
    for prompt, output in zip(PROMPTS, outputs, strict=True):
        assert output.new_token_ids == hf_reference[prompt], prompt


def test_prompt_larger_than_the_whole_pool_is_rejected(tiny_qwen2_path: Path) -> None:
    """A prompt that can never fit is refused at admission, not left to stall the queue."""
    config = _engine_config(tiny_qwen2_path, num_blocks=2, block_size=16)
    with (
        LLMEngine(config, local_files_only=True) as engine,
        pytest.raises(ValueError, match="KV blocks"),
    ):
        engine.add_request("too-big", list(range(100)))


# ----------------------------------------------------------------------------------------
# abort
# ----------------------------------------------------------------------------------------


def test_abort_mid_generation_frees_the_blocks(tiny_qwen2_path: Path) -> None:
    """Aborting a half-decoded request stops it and returns every block it held."""
    with LLMEngine(_engine_config(tiny_qwen2_path), local_files_only=True) as engine:
        engine.add_request("keep", PROMPTS[0], SamplingParams(max_tokens=32, temperature=0.0))
        engine.add_request("drop", PROMPTS[1], SamplingParams(max_tokens=32, temperature=0.0))
        for _ in range(4):
            engine.step()
        blocks_before = engine.scheduler.block_manager.num_free_blocks
        assert engine.abort("drop") is True
        assert engine.abort("drop") is False
        assert engine.scheduler.block_manager.num_free_blocks >= blocks_before
        assert engine.scheduler.get_sequence("drop") is None

        seen = 0
        while engine.has_unfinished():
            for output in engine.step():
                assert output.request_id == "keep"
                seen += output.output_tokens - seen
        assert seen == 32
        engine.scheduler.check_invariants()


# ----------------------------------------------------------------------------------------
# prefix caching
# ----------------------------------------------------------------------------------------


def _shared_prefix_prompts(count: int = 4) -> list[str]:
    """Prompts sharing a long system-prompt-like prefix, then diverging."""
    prefix = (
        "You are a careful assistant. Answer briefly and never invent facts. "
        "Use the context provided and nothing else. "
    )
    return [f"{prefix}Question {index}: what is {index} plus one?" for index in range(count)]


def test_prefix_caching_matches_uncached_and_records_hits(tiny_qwen2_path: Path) -> None:
    """Prefix caching changes the work, never the tokens, and reports a non-zero hit rate.

    The first request is run alone so that its blocks are published before the others are
    admitted. Requests admitted in the *same* step as the one that computes a prefix cannot
    hit on it -- a block is only indexed once its tokens are known to be computed -- which is
    a documented property of the cache, not a flaky ordering.
    """
    prompts = _shared_prefix_prompts()
    sampling = SamplingParams(max_tokens=12, temperature=0.0)

    def run(*, enable_prefix_caching: bool) -> tuple[list[list[int]], dict[str, Any]]:
        config = _engine_config(tiny_qwen2_path, enable_prefix_caching=enable_prefix_caching)
        with LLMEngine(config, local_files_only=True) as engine:
            first = engine.generate(prompts[:1], sampling, request_id_prefix="warm")
            rest = engine.generate(prompts[1:], sampling, request_id_prefix="rest")
            return [out.new_token_ids for out in (*first, *rest)], dict(engine.stats())

    uncached, cold_stats = run(enable_prefix_caching=False)
    cached, warm_stats = run(enable_prefix_caching=True)

    assert cached == uncached
    assert cold_stats.get("prefix_hits", 0) == 0
    assert warm_stats["prefix_hits"] > 0
    assert warm_stats["prefix_hit_rate"] > 0.0
    assert warm_stats["num_cached_prompt_tokens"] > 0


def test_prefix_cache_reports_cached_prompt_tokens_per_request(tiny_qwen2_path: Path) -> None:
    """A request served partly from the cache says so in its own usage accounting."""
    prompts = _shared_prefix_prompts(2)
    sampling = SamplingParams(max_tokens=4, temperature=0.0)
    with LLMEngine(_engine_config(tiny_qwen2_path), local_files_only=True) as engine:
        # Sequentially, so the second request meets a populated cache rather than racing it.
        first = engine.generate(prompts[:1], sampling, request_id_prefix="a")[0]
        second = engine.generate(prompts[1:], sampling, request_id_prefix="b")[0]
    assert first.cached_prompt_tokens == 0
    assert second.cached_prompt_tokens > 0
    assert second.prompt_tokens > second.cached_prompt_tokens


# ----------------------------------------------------------------------------------------
# scheduling policy
# ----------------------------------------------------------------------------------------


def test_tenant_fair_policy_serves_every_tenant(tiny_qwen2_path: Path) -> None:
    """Weighted fair queueing admits both tenants and produces the same tokens as FCFS."""
    sampling = SamplingParams(max_tokens=8, temperature=0.0)
    prompts = list(PROMPTS[:4])
    tenants = ["alpha", "beta", "alpha", "beta"]

    def run(policy: str) -> dict[str, list[int]]:
        config = _engine_config(
            tiny_qwen2_path,
            policy=policy,
            max_num_seqs=2,
            tenant_weights={"alpha": 3.0, "beta": 1.0} if policy == "tenant_fair" else {},
        )
        results: dict[str, list[int]] = {}
        with LLMEngine(config, local_files_only=True) as engine:
            for index, (prompt, tenant) in enumerate(zip(prompts, tenants, strict=True)):
                engine.add_request(f"r{index}", prompt, sampling, tenant_id=tenant)
            while engine.has_unfinished():
                for output in engine.step():
                    results.setdefault(output.request_id, []).extend(output.new_token_ids)
            assert engine.stats()["num_finished"] == len(prompts)
        return results

    assert run("tenant_fair") == run("fcfs")


# ----------------------------------------------------------------------------------------
# stop conditions
# ----------------------------------------------------------------------------------------


def test_stop_token_ids_and_max_tokens_set_the_finish_reason(tiny_qwen2_path: Path) -> None:
    """``LENGTH`` when the budget runs out, ``STOP`` when a configured token id appears."""
    with LLMEngine(_engine_config(tiny_qwen2_path), local_files_only=True) as engine:
        length = engine.generate(
            [PROMPTS[0]], SamplingParams(max_tokens=6, temperature=0.0, ignore_eos=True)
        )[0]
        assert length.output_tokens == 6
        assert str(length.finish_reason) == "length"

        stop_id = length.new_token_ids[2]
        stopped = engine.generate(
            [PROMPTS[0]],
            SamplingParams(max_tokens=16, temperature=0.0, stop_token_ids=[stop_id]),
            request_id_prefix="stop",
        )[0]
        assert stopped.new_token_ids[-1] == stop_id
        assert str(stopped.finish_reason) == "stop"
        assert stopped.output_tokens == 3


def test_stop_string_truncates_the_text(tiny_qwen2_path: Path) -> None:
    """A stop string ends the request and never reaches the client."""
    with LLMEngine(_engine_config(tiny_qwen2_path), local_files_only=True) as engine:
        baseline = engine.generate(
            [PROMPTS[0]], SamplingParams(max_tokens=12, temperature=0.0, ignore_eos=True)
        )[0]
        needle = baseline.text_delta[3:7]
        assert needle, "the tiny model produced no text to cut on"
        stopped = engine.generate(
            [PROMPTS[0]],
            SamplingParams(max_tokens=12, temperature=0.0, ignore_eos=True, stop=[needle]),
            request_id_prefix="stopstr",
        )[0]
    assert needle not in stopped.text_delta
    assert baseline.text_delta.startswith(stopped.text_delta)
    assert str(stopped.finish_reason) == "stop"


# ----------------------------------------------------------------------------------------
# memory sizing
# ----------------------------------------------------------------------------------------


def test_probe_memory_reports_a_plausible_host(tmp_path: Path) -> None:
    """The CPU probe returns a positive capacity and free <= total."""
    probe = probe_memory("cpu")
    assert probe.device == "cpu"
    assert probe.total_bytes > 0
    assert 0 <= probe.free_bytes <= probe.total_bytes
    assert probe.used_bytes == probe.total_bytes - probe.free_bytes
    assert probe.source in {"proc-meminfo", "fallback"}
    assert probe.to_dict()["device"] == "cpu"


def test_activation_headroom_scales_with_the_step_budget(tiny_qwen2_path: Path) -> None:
    """Doubling the token budget increases the reserve; the logits term is batch-driven."""
    config = ModelConfig.from_hf(tiny_qwen2_path, local_files_only=True)
    small = activation_headroom_bytes(
        config, max_num_batched_tokens=128, max_num_seqs=4, dtype=torch.float32
    )
    wide = activation_headroom_bytes(
        config, max_num_batched_tokens=256, max_num_seqs=4, dtype=torch.float32
    )
    many = activation_headroom_bytes(
        config, max_num_batched_tokens=128, max_num_seqs=8, dtype=torch.float32
    )
    assert small < wide
    assert small < many
    with pytest.raises(ValueError, match="positive"):
        activation_headroom_bytes(
            config, max_num_batched_tokens=0, max_num_seqs=1, dtype=torch.float32
        )


def test_size_kv_cache_honours_an_explicit_block_count(tiny_qwen2_path: Path) -> None:
    """An explicit ``num_blocks`` wins over the memory budget, which is what tests rely on."""
    config = ModelConfig.from_hf(tiny_qwen2_path, local_files_only=True)
    sizing = size_kv_cache(
        config,
        SchedulerConfig(block_size=16, num_blocks=13),
        device="cpu",
        dtype=torch.float32,
        num_blocks=13,
    )
    assert isinstance(sizing, KVCacheSizing)
    assert sizing.num_blocks == 13
    assert sizing.reason == "explicit"
    assert sizing.num_slots == 13 * 16
    assert sizing.kv_bytes == 13 * sizing.bytes_per_block
    assert sizing.to_dict()["reason"] == "explicit"


def test_size_kv_cache_uses_the_memory_budget(tiny_qwen2_path: Path) -> None:
    """With no explicit count the pool is whatever the budget pays for, and no more."""
    config = ModelConfig.from_hf(tiny_qwen2_path, local_files_only=True)
    probe = MemoryProbe(
        device="cpu", total_bytes=512 * 1024 * 1024, free_bytes=512 * 1024 * 1024, source="fallback"
    )
    sizing = size_kv_cache(
        config,
        SchedulerConfig(block_size=16, max_num_batched_tokens=256, max_num_seqs=4),
        device="cpu",
        dtype=torch.float32,
        gpu_memory_utilization=0.5,
        probe=probe,
    )
    assert sizing.reason == "memory"
    assert sizing.num_blocks >= 1
    assert sizing.kv_bytes <= sizing.budget_bytes
    assert (sizing.num_blocks + 1) * sizing.bytes_per_block > sizing.budget_bytes


def test_size_kv_cache_caps_at_max_model_len(tiny_qwen2_path: Path) -> None:
    """A tiny context length does not reserve a pool the engine can never use."""
    config = ModelConfig.from_hf(tiny_qwen2_path, local_files_only=True)
    probe = MemoryProbe(
        device="cpu", total_bytes=8 * 1024**3, free_bytes=8 * 1024**3, source="fallback"
    )
    sizing = size_kv_cache(
        config,
        SchedulerConfig(block_size=16, max_num_seqs=2),
        device="cpu",
        dtype=torch.float32,
        max_model_len=32,
        probe=probe,
    )
    assert sizing.reason == "max_model_len"
    assert sizing.num_blocks == 4  # ceil(32/16) blocks for each of 2 sequences


def test_size_kv_cache_refuses_an_impossible_budget(tiny_qwen2_path: Path) -> None:
    """No room for one block is an error with every term in the message, not a zero pool."""
    config = ModelConfig.from_hf(tiny_qwen2_path, local_files_only=True)
    probe = MemoryProbe(device="cpu", total_bytes=1024, free_bytes=1024, source="fallback")
    with pytest.raises(MemoryProfileError, match="no room for a KV block"):
        size_kv_cache(
            config,
            SchedulerConfig(block_size=16),
            device="cpu",
            dtype=torch.float32,
            probe=probe,
        )


# ----------------------------------------------------------------------------------------
# incremental detokenization
# ----------------------------------------------------------------------------------------


class _CharTokenizer:
    """A prefix-stable tokenizer over code points, so text assertions can be exact."""

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def test_detokenizer_emits_each_token_once() -> None:
    """Every character appears exactly once, in order, across the deltas."""
    detokenizer = IncrementalDetokenizer(_CharTokenizer())
    deltas = [detokenizer.append_token(ord(char)) for char in "hello world"]
    assert "".join(deltas) == "hello world"
    assert detokenizer.text == "hello world"
    assert detokenizer.num_tokens == 11
    assert detokenizer.append(()) == ""


def test_detokenizer_holds_back_incomplete_utf8(tiny_qwen2_path: Path) -> None:
    """Multi-token characters are emitted whole, never as replacement characters.

    A byte-level BPE splits an emoji across several tokens; decoding them one at a time
    yields ``U+FFFD`` for all but the last. The detokenizer must withhold until the character
    is complete, and the concatenated deltas must equal a plain full decode.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_qwen2_path), local_files_only=True)
    text = "hello 🙂 世界 café"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    detokenizer = IncrementalDetokenizer(tokenizer)
    deltas = [detokenizer.append_token(token_id) for token_id in token_ids]
    assert "�" not in "".join(deltas)
    assert "".join(deltas) + detokenizer.flush() == tokenizer.decode(token_ids)


def test_stop_matcher_catches_a_stop_string_split_across_chunks() -> None:
    """A stop string straddling two deltas is still caught, and is never emitted."""
    matcher = StopStringMatcher(["</s>"])
    first = matcher.feed("all done </")
    assert first.text == "all done "
    assert first.stop is None
    assert matcher.buffered == "</"
    second = matcher.feed("s> and more")
    assert second.text == ""
    assert second.stop == "</s>"
    assert matcher.buffered == ""


def test_stop_matcher_releases_text_that_cannot_match() -> None:
    """A partial match that turns out not to be one is released, not lost."""
    matcher = StopStringMatcher(["STOP"])
    assert matcher.feed("abcST").text == "abc"
    assert matcher.feed("X").text == "STX"
    assert matcher.flush() == ""


def test_stop_matcher_is_transparent_without_stop_strings() -> None:
    """With no stop strings nothing is buffered: the common case pays nothing."""
    matcher = StopStringMatcher()
    assert matcher.is_active is False
    assert matcher.feed("streamed straight through").text == "streamed straight through"
    assert matcher.buffered == ""


def test_streaming_decoder_combines_both_stages() -> None:
    """The per-request decoder detokenizes and applies stop strings in one call."""
    decoder = StreamingDecoder.create(_CharTokenizer(), stop=["!!"])
    assert decoder.feed([ord("h"), ord("i")]).text == "hi"
    delta = decoder.feed([ord("!"), ord("!"), ord("x")])
    assert delta.stopped
    assert delta.stop == "!!"
    assert decoder.text == "hi"
