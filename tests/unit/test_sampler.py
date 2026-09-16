"""Unit tests for :mod:`turboserve.engine.core.sampler`.

The sampler's hard requirement is that a batch of differently configured requests gets the
same tokens it would have got alone: continuous batching must not be observable from a
single request's output. So the tests assert per-stage properties (greedy is the argmax,
top-k restricts the support, top-p keeps exactly the minimal mass) and then assert that
those properties still hold when the rows are mixed into one batch, and that a seeded row is
reproducible regardless of its neighbours.
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from turboserve.engine.core.sampler import (
    Sampler,
    SamplerOutput,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
)
from turboserve.engine.core.sequence import Sequence
from turboserve.engine.core.types import SamplingParams

VOCAB = 8


def logits_from(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def support(row: torch.Tensor) -> set[int]:
    """Token ids that survived filtering."""
    return {int(index) for index in torch.nonzero(torch.isfinite(row)).flatten()}


# -- stages -----------------------------------------------------------------------------------


def test_apply_temperature_leaves_greedy_rows_finite() -> None:
    logits = logits_from([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    out = apply_temperature(logits, [0.0, 2.0])
    assert torch.equal(out[0], logits[0])
    assert torch.allclose(out[1], logits[1] / 2.0)
    assert torch.isfinite(out).all()
    assert apply_temperature(logits, [1.0, 1.0]) is logits


def test_apply_top_k_keeps_exactly_k_tokens_per_row() -> None:
    logits = logits_from([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]] * 3)
    out = apply_top_k(logits, [2, 0, 100])
    assert support(out[0]) == {6, 7}
    assert support(out[1]) == set(range(VOCAB))
    assert support(out[2]) == set(range(VOCAB))
    assert apply_top_k(logits, [0, 0, 0]) is logits


def test_apply_top_p_keeps_the_minimal_set_reaching_the_mass() -> None:
    # Cumulative mass ahead of each token: 0, 0.5, 0.75, 0.9. A token survives while that
    # mass has not passed p, so the thresholds below sit clear of the boundaries where a
    # tie would make either answer defensible.
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.10]], dtype=torch.float32))
    assert support(apply_top_p(logits, [0.4])[0]) == {0}
    assert support(apply_top_p(logits, [0.55])[0]) == {0, 1}
    assert support(apply_top_p(logits, [0.8])[0]) == {0, 1, 2}
    assert support(apply_top_p(logits, [0.95])[0]) == {0, 1, 2, 3}
    assert support(apply_top_p(logits, [1.0])[0]) == {0, 1, 2, 3}


def test_apply_top_p_property_over_random_distributions() -> None:
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(6, 32, generator=generator)
    probabilities = logits.softmax(dim=-1)
    for p in (0.1, 0.5, 0.9, 0.99):
        kept = apply_top_p(logits, [p] * 6)
        for row in range(6):
            survivors = support(kept[row])
            mass = float(probabilities[row, sorted(survivors)].sum())
            assert mass >= p - 1e-6, "the kept set must reach the requested mass"
            smallest = min(survivors, key=lambda token: float(probabilities[row, token]))
            assert mass - float(probabilities[row, smallest]) < p, "the set must be minimal"


def test_repetition_penalty_pushes_seen_tokens_down_whatever_their_sign() -> None:
    logits = logits_from([[2.0, -2.0, 0.5, 0.0]])
    out = apply_repetition_penalty(logits, [2.0], [[0, 1]])
    assert out[0, 0] == pytest.approx(1.0)  # positive logits are divided
    assert out[0, 1] == pytest.approx(-4.0)  # negative logits are multiplied
    assert out[0, 2] == pytest.approx(0.5)
    assert torch.equal(logits, logits_from([[2.0, -2.0, 0.5, 0.0]])), "input must not change"


def test_repetition_penalty_handles_repeats_and_skips_neutral_rows() -> None:
    logits = logits_from([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    out = apply_repetition_penalty(logits, [2.0, 1.0], [[0, 0, 0], [0, 1]])
    assert out[0, 0] == pytest.approx(1.0), "a repeated token is penalised once, not once per copy"
    assert torch.equal(out[1], logits[1])
    assert apply_repetition_penalty(logits, [1.0, 1.0], [[0], [1]]) is logits
    assert apply_repetition_penalty(logits, [2.0, 2.0], [[], []]) is logits


# -- end to end --------------------------------------------------------------------------------


def test_greedy_is_the_argmax_and_ignores_truncation() -> None:
    sampler = Sampler()
    logits = torch.randn(5, 16, generator=torch.Generator().manual_seed(3))
    params = [SamplingParams(temperature=0.0, top_k=3, top_p=0.5)] * 5
    out = sampler(logits, params)
    assert out.token_ids == logits.argmax(dim=-1).tolist()
    assert len(out) == 5
    assert list(out) == out.token_ids
    assert out.logprobs is None


def test_sampling_respects_top_k_over_many_draws() -> None:
    sampler = Sampler()
    logits = logits_from([[float(index) for index in range(VOCAB)]])
    params = [SamplingParams(temperature=1.0, top_k=2)]
    torch.manual_seed(0)
    drawn = {sampler(logits, params).token_ids[0] for _ in range(200)}
    assert drawn <= {VOCAB - 1, VOCAB - 2}
    assert len(drawn) == 2


def test_a_seeded_row_is_reproducible_and_independent_of_its_neighbours() -> None:
    sampler = Sampler()
    logits = torch.randn(3, 32, generator=torch.Generator().manual_seed(5))
    seeded = SamplingParams(temperature=1.0, seed=99)
    free = SamplingParams(temperature=1.0)

    def draw(order: list[SamplingParams]) -> list[int]:
        generators = [
            torch.Generator().manual_seed(p.seed) if p.seed is not None else None for p in order
        ]
        torch.manual_seed(1234)
        return sampler(logits, order, generators=generators).token_ids

    alone = draw([seeded, free, free])
    with_others = draw([seeded, free, free])
    assert alone[0] == with_others[0]
    torch.manual_seed(999)
    shifted = draw([seeded, free, free])
    assert shifted[0] == alone[0], "a seeded row must not depend on the global RNG"


def test_sample_sequences_uses_each_sequence_configuration() -> None:
    sampler = Sampler()
    greedy = Sequence(
        seq_id=0,
        request_id="greedy",
        prompt_token_ids=[0, 1],
        sampling=SamplingParams(temperature=0.0, logprobs=True),
    )
    penalised = Sequence(
        seq_id=1,
        request_id="penalised",
        prompt_token_ids=[3],
        sampling=SamplingParams(temperature=0.0, repetition_penalty=100.0),
    )
    logits = logits_from([[0.0, 1.0, 2.0, 5.0], [0.0, 1.0, 2.0, 5.0]])
    out = sampler.sample_sequences(logits, [greedy, penalised])
    assert out.token_ids[0] == 3, "no penalty: the largest logit wins"
    assert out.token_ids[1] == 2, "token 3 is in the prompt and is penalised out of the way"
    assert out.logprobs is not None
    expected = math.log(float(torch.tensor([0.0, 1.0, 2.0, 5.0]).softmax(0)[3]))
    assert out.logprobs[0] == pytest.approx(expected)
    assert math.isnan(out.logprobs[1]), "rows that did not ask for logprobs stay aligned as nan"


def test_logprobs_are_taken_before_temperature_and_truncation() -> None:
    sampler = Sampler()
    logits = logits_from([[0.0, 1.0, 2.0, 3.0]])
    sharp = sampler(logits, [SamplingParams(temperature=0.0, logprobs=True)])
    flat = sampler(logits, [SamplingParams(temperature=0.01, top_p=0.5, logprobs=True)])
    assert sharp.logprobs is not None and flat.logprobs is not None
    assert sharp.token_ids == flat.token_ids
    assert sharp.logprobs[0] == pytest.approx(flat.logprobs[0])


def test_the_input_logits_are_never_modified() -> None:
    sampler = Sampler()
    logits = torch.randn(4, 16, generator=torch.Generator().manual_seed(7))
    snapshot = logits.clone()
    sampler(
        logits,
        [SamplingParams(temperature=0.8, top_k=4, top_p=0.9, repetition_penalty=1.3)] * 4,
        token_history=[[1, 2]] * 4,
    )
    assert torch.equal(logits, snapshot)


def test_mixed_batch_matches_row_by_row_sampling() -> None:
    sampler = Sampler()
    logits = torch.randn(3, 24, generator=torch.Generator().manual_seed(17))
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=0.7, top_k=5, seed=42),
        SamplingParams(temperature=0.0, top_p=0.3),
    ]
    generators = [
        torch.Generator().manual_seed(p.seed) if p.seed is not None else None for p in params
    ]
    batched = sampler(logits, params, generators=generators).token_ids
    for index, param in enumerate(params):
        single_generator = (
            torch.Generator().manual_seed(param.seed) if param.seed is not None else None
        )
        alone = sampler(logits[index : index + 1], [param], generators=[single_generator]).token_ids
        assert alone == [batched[index]]


def test_argument_validation() -> None:
    sampler = Sampler()
    logits = torch.zeros(2, 4)
    params = [SamplingParams(), SamplingParams()]
    with pytest.raises(ValueError, match=r"\[batch, vocab\]"):
        sampler(torch.zeros(4), params)
    with pytest.raises(ValueError, match="sampling params for"):
        sampler(logits, params[:1])
    with pytest.raises(ValueError, match="generators for"):
        sampler(logits, params, generators=[None])
    with pytest.raises(ValueError, match="histories for"):
        sampler(logits, params, token_history=[[1]])
    assert sampler(torch.zeros(0, 4), []) == SamplerOutput(token_ids=[])


class _FakeCudaGenerator:
    """Stands in for a generator built on another device, which CPU-only CI cannot make."""

    device = torch.device("cuda", 0)


def test_a_generator_on_the_wrong_device_is_reported_clearly() -> None:
    sampler = Sampler()
    logits = torch.zeros(1, 4)
    generator = cast("torch.Generator", _FakeCudaGenerator())
    with pytest.raises(ValueError, match="ensure_generator"):
        sampler(logits, [SamplingParams(temperature=1.0)], generators=[generator])
