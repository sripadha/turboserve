"""``CausalLM`` reproduces ``transformers`` exactly, and is invariant to how KV got cached.

Four properties are asserted, on both supported architectures, in fp32 on CPU:

1. **Prefill logits equal HF logits.** The whole point of reimplementing the decoder is to
   change *where the KV lives*, not what the model computes; any deviation here is a bug in
   the rewrite (a wrong activation, an unrotated key, a bias on the wrong projection).
2. **Twenty greedy decode steps equal ``generate``.** Prefill parity alone would not catch
   a decode-path error such as a position off by one or a stale block table.
3. **Chunked prefill equals whole prefill.** The scheduler splits long prompts across steps
   to bound the batch's token budget; the split must be invisible.
4. **Prefix-cached prefill equals uncached prefill.** A sequence whose first tokens are
   already in the cache -- in *different, scrambled blocks* -- must produce the same logits
   as one that computed them itself. This is the correctness guarantee the whole prefix
   cache rests on.

Everything runs on tiny random checkpoints from the shared Hugging Face cache, in seconds,
with no network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from turboserve.engine.core.kv_cache import KVCache
from turboserve.engine.core.types import AttnMetadata, build_query_start_loc, pad_block_tables
from turboserve.engine.model import CausalLM, ModelConfig

BLOCK_SIZE = 8
PROMPT_LEN = 19
DECODE_STEPS = 20

#: Fixture names for the two architectures the engine implements.
MODEL_FIXTURES = ("tiny_qwen2_path", "tiny_llama_path")


def _blocks_for(num_tokens: int, *, scramble: bool, pool: int) -> list[int]:
    """Block ids for ``num_tokens`` tokens, optionally in a deliberately jumbled order."""
    needed = -(-num_tokens // BLOCK_SIZE)
    if needed > pool:
        raise AssertionError(f"test pool of {pool} blocks is too small for {num_tokens} tokens")
    ids = list(range(pool))
    if scramble:
        ids = ids[::-1]
    return ids[:needed]


def _step_metadata(blocks: list[int], num_computed: int, query_len: int) -> AttnMetadata:
    """Metadata for one single-sequence step with ``num_computed`` tokens already cached."""
    slots = [
        blocks[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
        for pos in range(num_computed, num_computed + query_len)
    ]
    context = num_computed + query_len
    return AttnMetadata(
        slot_mapping=torch.tensor(slots, dtype=torch.long),
        block_tables=pad_block_tables([blocks]),
        context_lens=torch.tensor([context], dtype=torch.long),
        query_start_loc=build_query_start_loc([query_len]),
        max_query_len=query_len,
        max_context_len=context,
        num_prefill_seqs=1 if query_len > 1 else 0,
        num_decode_seqs=0 if query_len > 1 else 1,
    )


def _new_cache(config: ModelConfig, *, num_blocks: int) -> KVCache:
    return KVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device="cpu",
    )


def _run_chunk(
    model: CausalLM,
    cache: KVCache,
    tokens: torch.Tensor,
    *,
    blocks: list[int],
    num_computed: int,
) -> torch.Tensor:
    """Run one chunk of ``tokens`` and return its hidden states."""
    query_len = int(tokens.shape[0])
    meta = _step_metadata(blocks, num_computed, query_len)
    meta.validate(block_size=BLOCK_SIZE)
    positions = torch.arange(num_computed, num_computed + query_len, dtype=torch.long)
    with torch.no_grad():
        return model(tokens, positions, cache, meta)


def _prefill_logits(
    model: CausalLM, prompt: torch.Tensor, *, chunk: int | None = None, scramble: bool = False
) -> torch.Tensor:
    """Prefill ``prompt`` (whole, or in ``chunk``-sized pieces) and return all logits."""
    pool = 8
    blocks = _blocks_for(int(prompt.shape[0]), scramble=scramble, pool=pool)
    cache = _new_cache(model.config, num_blocks=pool)
    size = chunk or int(prompt.shape[0])
    hidden_parts: list[torch.Tensor] = []
    for start in range(0, int(prompt.shape[0]), size):
        piece = prompt[start : start + size]
        hidden_parts.append(_run_chunk(model, cache, piece, blocks=blocks, num_computed=start))
    hidden = torch.cat(hidden_parts, dim=0)
    with torch.no_grad():
        return model.compute_logits(hidden)


def _greedy_decode(model: CausalLM, prompt: torch.Tensor, steps: int) -> list[int]:
    """Prefill then take ``steps`` greedy decode steps, returning the generated token ids."""
    pool = 16
    blocks = _blocks_for(int(prompt.shape[0]) + steps, scramble=False, pool=pool)
    cache = _new_cache(model.config, num_blocks=pool)
    hidden = _run_chunk(model, cache, prompt, blocks=blocks, num_computed=0)
    with torch.no_grad():
        logits = model.compute_logits(hidden, [int(prompt.shape[0]) - 1])
    generated: list[int] = [int(logits[0].argmax())]

    for step in range(steps - 1):
        num_computed = int(prompt.shape[0]) + step
        token = torch.tensor([generated[-1]], dtype=torch.long)
        hidden = _run_chunk(model, cache, token, blocks=blocks, num_computed=num_computed)
        with torch.no_grad():
            logits = model.compute_logits(hidden, [0])
        generated.append(int(logits[0].argmax()))
    return generated


def _load_pair(path: Path) -> tuple[CausalLM, object]:
    """Load our model and the reference ``transformers`` model from the same checkpoint."""
    from transformers import AutoModelForCausalLM

    ours = CausalLM.from_pretrained(path, dtype=torch.float32, device="cpu", local_files_only=True)
    reference = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
    reference.eval()
    return ours, reference


def _prompt_for(config: ModelConfig, *, length: int = PROMPT_LEN) -> torch.Tensor:
    """A deterministic prompt of in-vocabulary ids (low ids are real tokens in both models)."""
    generator = torch.Generator().manual_seed(1234)
    high = min(config.vocab_size, 1000)
    return torch.randint(2, high, (length,), generator=generator, dtype=torch.long)


@pytest.fixture(params=MODEL_FIXTURES)
def model_path(request: pytest.FixtureRequest) -> Path:
    """Parametrised over the tiny Qwen2 and tiny Llama checkpoints."""
    path: Path = request.getfixturevalue(request.param)
    return path


def test_config_matches_the_checkpoint(model_path: Path) -> None:
    """``ModelConfig.from_hf`` reads the architecture the checkpoint declares."""
    config = ModelConfig.from_hf(model_path, local_files_only=True)
    assert config.architecture in {"Qwen2ForCausalLM", "LlamaForCausalLM"}
    assert config.head_dim * config.num_attention_heads >= config.hidden_size
    assert config.num_attention_heads % config.num_key_value_heads == 0
    # Qwen2 always carries a q/k/v bias; Llama checkpoints released so far never do.
    assert config.qkv_bias == (config.architecture == "Qwen2ForCausalLM")
    assert config.o_proj_bias is False


def test_prefill_logits_match_transformers(model_path: Path) -> None:
    """Whole-prompt prefill through the paged cache equals HF's dense forward."""
    ours, reference = _load_pair(model_path)
    prompt = _prompt_for(ours.config)

    logits = _prefill_logits(ours, prompt)
    with torch.no_grad():
        expected = reference(input_ids=prompt.unsqueeze(0)).logits[0].float()

    assert logits.shape == expected.shape
    torch.testing.assert_close(logits, expected, rtol=0, atol=1e-4)


def test_greedy_decode_matches_hf_generate(model_path: Path) -> None:
    """Twenty greedy steps reproduce ``generate`` token for token.

    ``min_new_tokens`` keeps ``generate`` from stopping at an EOS that a random-weight tiny
    model can emit at any time; the comparison is then over a fixed-length window rather
    than over whatever prefix happened to be produced.
    """
    ours, reference = _load_pair(model_path)
    prompt = _prompt_for(ours.config)
    eos = ours.config.eos_token_ids[0] if ours.config.eos_token_ids else 0

    with torch.no_grad():
        generated = reference.generate(
            input_ids=prompt.unsqueeze(0),
            attention_mask=torch.ones_like(prompt).unsqueeze(0),
            max_new_tokens=DECODE_STEPS,
            min_new_tokens=DECODE_STEPS,
            do_sample=False,
            pad_token_id=eos,
        )
    expected = generated[0, prompt.shape[0] :].tolist()
    assert len(expected) == DECODE_STEPS

    # Our greedy loop must suppress EOS for exactly as long as `generate` does, otherwise
    # the two are answering different questions once an EOS becomes the argmax.
    ours_tokens = _greedy_decode_without_eos(ours, prompt, DECODE_STEPS, ours.config.eos_token_ids)
    assert ours_tokens == expected


def _greedy_decode_without_eos(
    model: CausalLM, prompt: torch.Tensor, steps: int, eos_ids: tuple[int, ...]
) -> list[int]:
    """Greedy decode with the EOS ids masked out, mirroring ``min_new_tokens``."""
    if not eos_ids:
        return _greedy_decode(model, prompt, steps)
    original = model.compute_logits

    def masked(hidden: torch.Tensor, indices: object = None) -> torch.Tensor:
        logits = original(hidden, indices)  # type: ignore[arg-type]
        logits[:, list(eos_ids)] = float("-inf")
        return logits

    model.compute_logits = masked  # type: ignore[method-assign]
    try:
        return _greedy_decode(model, prompt, steps)
    finally:
        model.compute_logits = original  # type: ignore[method-assign]


@pytest.mark.parametrize("chunk", [3, 8])
def test_chunked_prefill_equals_whole_prefill(model_path: Path, chunk: int) -> None:
    """Splitting a prompt across scheduler steps does not change a single logit.

    Chunk 8 is exactly one block and chunk 3 straddles block boundaries, which is where a
    slot-mapping or position off-by-one would show up.
    """
    ours, _ = _load_pair(model_path)
    prompt = _prompt_for(ours.config)

    whole = _prefill_logits(ours, prompt)
    chunked = _prefill_logits(ours, prompt, chunk=chunk)

    torch.testing.assert_close(chunked, whole, rtol=0, atol=1e-5)


def test_prefix_cached_prefill_equals_uncached(model_path: Path) -> None:
    """A prompt whose prefix is already cached in other blocks yields identical logits.

    The cached prefix is computed by a *different sequence* and lives in scrambled blocks;
    the suffix sequence then attends to it through its own block table. If block
    indirection or the causal offset were wrong, this is where it would surface.
    """
    ours, _ = _load_pair(model_path)
    prompt = _prompt_for(ours.config)
    prefix_len = 2 * BLOCK_SIZE  # a whole number of blocks, as the prefix cache requires

    uncached = _prefill_logits(ours, prompt)

    pool = 8
    blocks = _blocks_for(int(prompt.shape[0]), scramble=True, pool=pool)
    cache = _new_cache(ours.config, num_blocks=pool)
    # Step 1: some earlier request populated the prefix blocks.
    _run_chunk(ours, cache, prompt[:prefix_len], blocks=blocks, num_computed=0)
    # Step 2: the new request hits that prefix and only computes the suffix.
    suffix_hidden = _run_chunk(
        ours, cache, prompt[prefix_len:], blocks=blocks, num_computed=prefix_len
    )
    with torch.no_grad():
        cached_logits = ours.compute_logits(suffix_hidden)

    torch.testing.assert_close(cached_logits, uncached[prefix_len:], rtol=0, atol=1e-5)


def test_tied_embeddings_share_storage_when_configured() -> None:
    """A tied model reuses the embedding tensor for ``lm_head`` instead of copying it."""
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "vocab_size": 64,
        "max_position_embeddings": 64,
    }
    tied = CausalLM(
        ModelConfig.from_dict({**raw, "tie_word_embeddings": True}), dtype=torch.float32
    )
    assert tied.lm_head.weight is tied.model.embed_tokens.weight
    assert tied.tied_parameter_names() == ("lm_head.weight",)

    untied = CausalLM(
        ModelConfig.from_dict({**raw, "tie_word_embeddings": False}), dtype=torch.float32
    )
    assert untied.lm_head.weight is not untied.model.embed_tokens.weight
    assert untied.tied_parameter_names() == ()
    # The tied model saves exactly one vocabulary projection (64 x 16 values).
    assert untied.num_parameters() - tied.num_parameters() == 64 * 16


def test_init_weights_is_deterministic_and_finite() -> None:
    """Seeded initialisation gives reproducible, usable weights for model-free tests."""
    config = ModelConfig.from_dict(
        {
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "vocab_size": 32,
            "max_position_embeddings": 64,
        }
    )
    first = CausalLM(config, dtype=torch.float32)
    first.init_weights(seed=7)
    second = CausalLM(config, dtype=torch.float32)
    second.init_weights(seed=7)

    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        assert torch.isfinite(a).all(), name
        torch.testing.assert_close(a, b)
    # Norm gains initialise to one and biases to zero, as in a real checkpoint.
    assert bool((first.model.norm.weight == 1.0).all())
    assert bool((first.model.layers[0].self_attn.q_proj.bias == 0.0).all())


# -- the hook the multi-LoRA module builds on ------------------------------------------


def _tiny_config() -> ModelConfig:
    """A two-layer Qwen2-shaped config used by the structural tests below."""
    return ModelConfig.from_dict(
        {
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "vocab_size": 32,
            "max_position_embeddings": 64,
        }
    )


def test_named_linears_exposes_every_lora_target() -> None:
    """The LoRA registry finds its targets through this helper, not by walking attributes."""
    from turboserve.engine.model import LORA_TARGET_MODULES, named_linears

    model = CausalLM(_tiny_config(), dtype=torch.float32)
    names = {name for name, _ in named_linears(model)}
    for layer in range(2):
        for target in LORA_TARGET_MODULES:
            group = (
                "self_attn"
                if target.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
                else ("mlp")
            )
            assert f"model.layers.{layer}.{group}.{target}" in names
    assert "lm_head" in names


def test_linear_factory_substitutes_every_projection_and_threads_the_context() -> None:
    """``use_linear_factory`` is how multi-LoRA installs itself without editing the model.

    The substituted class records the ``lora_ctx`` it was handed, which proves the decoder
    threads the context all the way down to the projections -- the property the batched
    SGMV-style adapter application depends on.
    """
    from turboserve.engine.core.types import LoRAContext
    from turboserve.engine.model import LinearBase, use_linear_factory

    seen: list[str] = []
    contexts: list[LoRAContext | None] = []

    class Recording(LinearBase):
        def forward(self, x: torch.Tensor, lora_ctx: LoRAContext | None = None) -> torch.Tensor:
            contexts.append(lora_ctx)
            return super().forward(x, None)

    def factory(*args: object, **kwargs: object) -> LinearBase:
        seen.append(str(kwargs.get("name", "")))
        return Recording(*args, **kwargs)  # type: ignore[arg-type]

    config = _tiny_config()
    with use_linear_factory(factory):
        model = CausalLM(config, dtype=torch.float32)
    assert LinearBase.linear_factory is None, "the hook must not outlive the with-block"
    assert "model.layers.0.self_attn.q_proj" in seen
    assert all(isinstance(linear, Recording) for _, linear in _all_linears(model))

    model.init_weights(seed=0)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    cache = _new_cache(config, num_blocks=2)
    ctx = LoRAContext.from_slots([2, 2, 2])
    meta = _step_metadata([0, 1], 0, 3)
    with torch.no_grad():
        model(prompt, torch.arange(3), cache, meta, ctx)

    assert contexts, "no projection was called"
    assert all(received is ctx for received in contexts)


def _all_linears(model: CausalLM) -> list[tuple[str, object]]:
    from turboserve.engine.model import named_linears

    return list(named_linears(model))
