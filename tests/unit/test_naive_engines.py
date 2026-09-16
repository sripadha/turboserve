"""Tests for the ``transformers`` baselines that the reference engine is measured against.

A baseline is only useful if it is honest, so the assertions here are about *fidelity*, not
speed: the tokens must be the ones ``transformers.generate`` produces, the request interface
must be the one the benchmark drives :class:`LLMEngine` through, and the batching behaviour
must be the batching behaviour being described (one at a time; fixed-size groups).

No timing is asserted anywhere. These run on CPU on a tiny random checkpoint, where relative
speed means nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from turboserve.engine.core.types import EngineConfig, SamplingParams, SchedulerConfig
from turboserve.engine.runtime.engine import LLMEngine
from turboserve.engine.runtime.naive import (
    BaselineEngine,
    NaiveHFEngine,
    StaticBatchHFEngine,
)

PROMPTS: tuple[str, ...] = (
    "Hello world",
    "The quick brown fox jumps over",
    "A B C D E F G H I J K L",
    "Paris is the capital of",
)


def _config(model_path: Path, **scheduler: Any) -> EngineConfig:
    options: dict[str, Any] = {
        "max_num_seqs": 4,
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
def hf_greedy(tiny_qwen2_path: Path) -> dict[str, list[int]]:
    """Greedy continuations straight from ``transformers``, one request per call."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_qwen2_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(tiny_qwen2_path), dtype=torch.float32, local_files_only=True
    ).eval()
    reference: dict[str, list[int]] = {}
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            generated = model.generate(
                torch.tensor([ids], dtype=torch.long),
                max_new_tokens=12,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            reference[prompt] = generated[0, len(ids) :].tolist()
    del model
    return reference


def _drain(engine: BaselineEngine, sampling: SamplingParams) -> dict[str, list[int]]:
    """Queue every prompt, step until the queue is empty, collect the completions."""
    for index, prompt in enumerate(PROMPTS):
        engine.add_request(f"r{index}", prompt, sampling)
    results: dict[str, list[int]] = {}
    while engine.has_unfinished():
        for output in engine.step():
            assert output.finished
            results[output.request_id] = list(output.new_token_ids)
    return results


# ----------------------------------------------------------------------------------------
# fidelity
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [NaiveHFEngine, StaticBatchHFEngine])
def test_baselines_match_hf_greedy(
    cls: type[BaselineEngine], tiny_qwen2_path: Path, hf_greedy: dict[str, list[int]]
) -> None:
    """Both baselines reproduce ``transformers.generate`` exactly, padding notwithstanding.

    For :class:`StaticBatchHFEngine` this is the load-bearing assertion: left padding plus an
    attention mask must make a padded row compute the same thing it would have alone.
    """
    sampling = SamplingParams(max_tokens=12, temperature=0.0)
    with cls(_config(tiny_qwen2_path), local_files_only=True) as engine:
        results = _drain(engine, sampling)
    for index, prompt in enumerate(PROMPTS):
        assert results[f"r{index}"] == hf_greedy[prompt], prompt


def test_baselines_agree_with_the_reference_engine(
    tiny_qwen2_path: Path, hf_greedy: dict[str, list[int]]
) -> None:
    """The reference engine and the baselines are comparable because they agree token by token.

    A throughput comparison between two engines that decode differently would measure
    nothing, so this equality is a precondition of the benchmark, not a nicety.
    """
    sampling = SamplingParams(max_tokens=12, temperature=0.0)
    with LLMEngine(_config(tiny_qwen2_path), local_files_only=True) as engine:
        outputs = engine.generate(list(PROMPTS), sampling, request_id_prefix="r")
        reference = {f"r{index}": out.new_token_ids for index, out in enumerate(outputs)}
    with NaiveHFEngine(_config(tiny_qwen2_path), local_files_only=True) as naive:
        assert _drain(naive, sampling) == reference


def test_output_carries_timing_and_usage(tiny_qwen2_path: Path) -> None:
    """A baseline result fills in the same timing fields the reference engine does."""
    sampling = SamplingParams(max_tokens=6, temperature=0.0, ignore_eos=True)
    with NaiveHFEngine(_config(tiny_qwen2_path), local_files_only=True) as engine:
        engine.add_request("r0", PROMPTS[0], sampling)
        output = engine.step()[0]
    timing = output.timing
    assert timing.t_arrival is not None
    assert timing.t_first_token is not None
    assert timing.t_finish is not None
    assert timing.ttft() is not None and timing.ttft() > 0
    assert timing.e2e() is not None and timing.e2e() >= timing.ttft()
    usage = output.usage()
    assert usage["completion_tokens"] == 6
    assert usage["total_tokens"] == usage["prompt_tokens"] + 6
    assert output.text_delta != ""


# ----------------------------------------------------------------------------------------
# batching behaviour
# ----------------------------------------------------------------------------------------


def test_naive_engine_serves_exactly_one_request_per_step(tiny_qwen2_path: Path) -> None:
    """The naive baseline never puts two requests in one forward pass."""
    sampling = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
    with NaiveHFEngine(_config(tiny_qwen2_path), local_files_only=True) as engine:
        for index, prompt in enumerate(PROMPTS):
            engine.add_request(f"r{index}", prompt, sampling)
        assert len(engine) == 4
        batches = []
        while engine.has_unfinished():
            batches.append(len(engine.step()))
        stats = engine.stats()
    assert batches == [1, 1, 1, 1]
    assert stats["num_batches"] == 4
    assert stats["backend"] == "naive_hf"
    assert stats["num_finished"] == 4


def test_static_batch_groups_requests_and_waits_for_the_batch(tiny_qwen2_path: Path) -> None:
    """A fixed batch size groups requests; a partial tail is flushed when configured to."""
    sampling = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
    with StaticBatchHFEngine(
        _config(tiny_qwen2_path), batch_size=3, local_files_only=True
    ) as engine:
        for index, prompt in enumerate(PROMPTS):
            engine.add_request(f"r{index}", prompt, sampling)
        sizes = []
        while engine.has_unfinished():
            sizes.append(len(engine.step()))
        stats = engine.stats()
    assert sizes == [3, 1]
    assert stats["backend"] == "static_batch"
    assert stats["num_batches"] == 2


def test_static_batch_can_refuse_to_flush_a_partial_batch(tiny_qwen2_path: Path) -> None:
    """With ``flush_partial=False`` an incomplete batch waits, as a real server would."""
    sampling = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
    with StaticBatchHFEngine(
        _config(tiny_qwen2_path), batch_size=3, flush_partial=False, local_files_only=True
    ) as engine:
        engine.add_request("r0", PROMPTS[0], sampling)
        engine.add_request("r1", PROMPTS[1], sampling)
        assert engine.step() == []
        assert engine.has_unfinished()
        engine.add_request("r2", PROMPTS[2], sampling)
        assert len(engine.step()) == 3
        assert not engine.has_unfinished()


# ----------------------------------------------------------------------------------------
# interface parity and error handling
# ----------------------------------------------------------------------------------------


def test_interface_matches_the_reference_engine(tiny_qwen2_path: Path) -> None:
    """The benchmark drives all three engines through the same names, so they must exist."""
    for name in ("add_request", "step", "abort", "has_unfinished", "stats", "close"):
        assert callable(getattr(NaiveHFEngine, name))
        assert callable(getattr(StaticBatchHFEngine, name))
        assert callable(getattr(LLMEngine, name))


def test_abort_removes_a_queued_request(tiny_qwen2_path: Path) -> None:
    """A request that has not started yet can be cancelled; ids are unique while queued."""
    sampling = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
    with NaiveHFEngine(_config(tiny_qwen2_path), local_files_only=True) as engine:
        engine.add_request("r0", PROMPTS[0], sampling)
        engine.add_request("r1", PROMPTS[1], sampling)
        with pytest.raises(ValueError, match="already in flight"):
            engine.add_request("r1", PROMPTS[2], sampling)
        assert engine.abort("r1") is True
        assert engine.abort("r1") is False
        assert len(engine) == 1
        assert len(engine.step()) == 1
        assert not engine.has_unfinished()


def test_token_id_prompts_are_accepted(tiny_qwen2_path: Path) -> None:
    """Pre-tokenised prompts skip the tokenizer, as they do in the reference engine."""
    sampling = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
    with NaiveHFEngine(_config(tiny_qwen2_path), local_files_only=True) as engine:
        token_ids = engine.encode(PROMPTS[0])
        engine.add_request("text", PROMPTS[0], sampling)
        engine.add_request("ids", token_ids, sampling)
        from_text = engine.step()[0]
        from_ids = engine.step()[0]
    assert from_text.new_token_ids == from_ids.new_token_ids


def test_a_heterogeneous_batch_is_refused_rather_than_silently_wrong(
    tiny_qwen2_path: Path,
) -> None:
    """``generate`` has no per-row sampling, so mixed parameters in one batch must fail loudly."""
    with StaticBatchHFEngine(
        _config(tiny_qwen2_path), batch_size=2, local_files_only=True
    ) as engine:
        engine.add_request("a", PROMPTS[0], SamplingParams(max_tokens=4, temperature=0.0))
        engine.add_request("b", PROMPTS[1], SamplingParams(max_tokens=4, temperature=0.7))
        with pytest.raises(ValueError, match="one sampling configuration"):
            engine.step()


def test_empty_prompt_and_closed_engine_are_errors(tiny_qwen2_path: Path) -> None:
    """Two failure modes that must not become silent no-ops."""
    engine = NaiveHFEngine(_config(tiny_qwen2_path), local_files_only=True)
    with pytest.raises(ValueError, match="empty prompt"):
        engine.add_request("empty", [])
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.add_request("late", PROMPTS[0])
