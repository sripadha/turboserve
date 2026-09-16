"""Smoke tests for the cached tiny checkpoints the engine tests are written against.

They assert two things the rest of the suite depends on: the tiny Qwen2 and Llama
checkpoints resolve from the local Hugging Face cache, and the installed transformers
version can load them and run a forward pass on CPU. When the engine's own Qwen2/Llama
implementation lands, its reference comparison uses exactly these fixtures, so a failure
here localises the problem to the environment rather than to the engine.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT_LEN = 6


def _forward(path: Path, expected_arch: str) -> None:
    config = AutoConfig.from_pretrained(path)
    assert config.architectures == [expected_arch]

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.eval()

    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(1, PROMPT_LEN), generator=generator
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    assert logits.shape == (1, PROMPT_LEN, config.vocab_size)
    assert logits.dtype is torch.float32
    assert torch.isfinite(logits).all()


def test_tiny_qwen2_forward(tiny_qwen2_path: Path) -> None:
    _forward(tiny_qwen2_path, "Qwen2ForCausalLM")


def test_tiny_llama_forward(tiny_llama_path: Path) -> None:
    _forward(tiny_llama_path, "LlamaForCausalLM")


def test_tiny_qwen2_tokenizer_round_trips(tiny_qwen2_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_qwen2_path)
    ids = tokenizer("hello turboserve", add_special_tokens=False)["input_ids"]
    assert len(ids) > 0
    assert tokenizer.decode(ids) == "hello turboserve"


def test_tiny_models_are_small_enough_for_unit_tests(
    tiny_qwen2_path: Path, tiny_llama_path: Path
) -> None:
    """Guard against a fixture silently pointing at a real (multi-GB) checkpoint."""
    for path in (tiny_qwen2_path, tiny_llama_path):
        weight_bytes = sum(f.stat().st_size for f in path.glob("*.safetensors"))
        assert 0 < weight_bytes < 512 * 1024 * 1024, path
