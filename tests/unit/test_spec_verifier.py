"""Verification is where speculative decoding either is or is not exact, so it is tested hard.

Two properties matter and nothing else does:

* greedy verification emits exactly the tokens the target model would have emitted alone;
* rejection-sampling verification emits tokens distributed exactly as the target's own
  next-token distribution, not approximately.

The second is a statement about a distribution, so it is tested as one: many draws through
the real code path, then a chi-square goodness-of-fit test against the target distribution.
The draws use a seeded generator, so the test is a deterministic replay rather than a coin
flip that fails once a month.

Everything here is pure tensor work on a four-token vocabulary: no model, no cache, no GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from turboserve.engine.core.types import SamplingParams
from turboserve.engine.spec.verifier import (
    Verification,
    processed_logits,
    sampling_probs,
    verify_batch,
    verify_greedy,
    verify_rejection_sampling,
    verify_sequence,
)

#: chi-square critical value at alpha = 0.001 for 3 degrees of freedom (a 4-token vocabulary).
CHI2_CRITICAL_DF3 = 16.266

TARGET = torch.tensor([0.50, 0.30, 0.15, 0.05], dtype=torch.float32)
DRAFT = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)


def chi_square(counts: list[int], probs: torch.Tensor) -> float:
    """Pearson's statistic for observed ``counts`` against ``probs``."""
    total = sum(counts)
    expected = [total * float(p) for p in probs]
    return sum((count - exp) ** 2 / exp for count, exp in zip(counts, expected, strict=True))


def rows(distribution: torch.Tensor, count: int) -> torch.Tensor:
    """``count`` identical rows of ``distribution``, the shape the verifier expects."""
    return distribution.unsqueeze(0).expand(count, -1).contiguous()


# -- greedy ---------------------------------------------------------------------------------


def test_greedy_accepts_every_draft_token_the_target_would_have_chosen() -> None:
    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0, 0.0],  # target would say 1
            [0.0, 0.0, 5.0, 0.0],  # target would say 2
            [5.0, 0.0, 0.0, 0.0],  # bonus position: target says 0
        ]
    )
    result = verify_greedy(logits, [1, 2])
    assert result.accepted_tokens == (1, 2)
    assert result.bonus_token == 0
    assert result.tokens == [1, 2, 0]
    assert result.all_accepted
    assert result.num_emitted == 3


def test_greedy_stops_at_the_first_disagreement_and_emits_the_target_token() -> None:
    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0, 0.0],  # agrees with draft token 1
            [5.0, 0.0, 0.0, 0.0],  # target says 0, the draft said 3
            [0.0, 0.0, 5.0, 0.0],
        ]
    )
    result = verify_greedy(logits, [1, 3])
    assert result.accepted_tokens == (1,)
    assert result.bonus_token == 0
    assert not result.all_accepted
    assert result.num_emitted == 2


def test_greedy_with_no_draft_is_an_ordinary_argmax() -> None:
    result = verify_greedy(torch.tensor([[0.0, 0.0, 7.0, 0.0]]), [])
    assert result.accepted_tokens == ()
    assert result.bonus_token == 2
    assert result.num_emitted == 1


def test_greedy_rejects_logits_of_the_wrong_height() -> None:
    with pytest.raises(ValueError, match="target_logits must be"):
        verify_greedy(torch.zeros((2, 4)), [1, 2])


# -- rejection sampling: the distribution invariant --------------------------------------------


def test_emitted_token_is_distributed_exactly_as_the_target() -> None:
    """The headline guarantee: p(emitted == x) == target(x) for every x."""
    generator = torch.Generator().manual_seed(20260916)
    target_rows = rows(TARGET, 2)
    draft_rows = rows(DRAFT, 1)
    counts = [0, 0, 0, 0]
    trials = 12000
    for _ in range(trials):
        drafted = int(torch.multinomial(DRAFT, 1, generator=generator)[0])
        result = verify_rejection_sampling(target_rows, [drafted], draft_rows, generator=generator)
        counts[result.tokens[0]] += 1
    assert sum(counts) == trials
    assert chi_square(counts, TARGET) < CHI2_CRITICAL_DF3


def test_the_token_after_an_accepted_draft_is_target_distributed_too() -> None:
    """Acceptance does not bias what comes next: every emitted position is a draw from p."""
    generator = torch.Generator().manual_seed(4242)
    target_rows = rows(TARGET, 3)
    draft_rows = rows(DRAFT, 2)
    first: list[int] = [0, 0, 0, 0]
    second: list[int] = [0, 0, 0, 0]
    for _ in range(12000):
        draws = torch.multinomial(DRAFT, 2, replacement=True, generator=generator)
        drafted = [int(token) for token in draws]
        result = verify_rejection_sampling(target_rows, drafted, draft_rows, generator=generator)
        first[result.tokens[0]] += 1
        if len(result.tokens) > 1:
            second[result.tokens[1]] += 1
    assert chi_square(first, TARGET) < CHI2_CRITICAL_DF3
    assert sum(second) > 1000, "the draft should have been accepted often enough to test"
    assert chi_square(second, TARGET) < CHI2_CRITICAL_DF3


def test_a_drafter_without_probabilities_is_exact_as_well() -> None:
    """``draft_probs=None`` (the n-gram drafter) is read as a point mass and stays exact."""
    generator = torch.Generator().manual_seed(7)
    target_rows = rows(TARGET, 2)
    counts = [0, 0, 0, 0]
    trials = 12000
    for index in range(trials):
        drafted = index % 4  # a deterministic drafter: every token proposed equally often
        result = verify_rejection_sampling(target_rows, [drafted], None, generator=generator)
        counts[result.tokens[0]] += 1
    assert chi_square(counts, TARGET) < CHI2_CRITICAL_DF3


def test_an_identical_draft_distribution_accepts_the_whole_draft() -> None:
    generator = torch.Generator().manual_seed(11)
    target_rows = rows(TARGET, 4)
    draft_rows = rows(TARGET, 3)
    for _ in range(200):
        draws = torch.multinomial(TARGET, 3, replacement=True, generator=generator)
        drafted = [int(token) for token in draws]
        result = verify_rejection_sampling(target_rows, drafted, draft_rows, generator=generator)
        assert result.all_accepted, "p == q must accept with probability one"
        assert result.accepted_tokens == tuple(drafted)


def test_a_draft_the_target_gives_no_mass_is_always_rejected() -> None:
    """Truncation is respected: a token outside the target's support can never be emitted."""
    target = torch.tensor([0.6, 0.4, 0.0, 0.0])
    draft = torch.tensor([0.0, 0.0, 0.5, 0.5])
    generator = torch.Generator().manual_seed(99)
    counts = [0, 0, 0, 0]
    for _ in range(500):
        result = verify_rejection_sampling(
            rows(target, 2), [2], rows(draft, 1), generator=generator
        )
        assert result.num_accepted == 0
        counts[result.bonus_token] += 1
    assert counts[2] == 0 and counts[3] == 0
    assert counts[0] > 0 and counts[1] > 0


def test_zero_residual_falls_back_to_the_target_distribution() -> None:
    """When p and q agree the residual is empty; the emitted token still comes from p."""
    shared = torch.tensor([0.5, 0.5, 0.0, 0.0])
    generator = torch.Generator().manual_seed(3)
    result = verify_rejection_sampling(rows(shared, 2), [0], rows(shared, 1), generator=generator)
    assert result.bonus_token in (0, 1)


def test_rejection_sampling_validates_its_shapes_and_devices() -> None:
    with pytest.raises(ValueError, match="target_probs must be"):
        verify_rejection_sampling(rows(TARGET, 1), [0], rows(DRAFT, 1))
    with pytest.raises(ValueError, match="draft_probs must be"):
        verify_rejection_sampling(rows(TARGET, 2), [0], rows(DRAFT, 2))


def test_verification_record_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="accepted tokens out of"):
        Verification(accepted_tokens=(1, 2), bonus_token=0, num_drafted=1)
    with pytest.raises(ValueError, match="num_drafted must be"):
        Verification(accepted_tokens=(), bonus_token=0, num_drafted=-1)


# -- the sampler transform chain ----------------------------------------------------------------


def test_processed_logits_applies_temperature_top_k_and_top_p() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    params = SamplingParams(temperature=0.5, top_k=2)
    processed = processed_logits(logits, params)
    assert processed[0, 0] == pytest.approx(4.0)
    assert processed[0, 1] == pytest.approx(2.0)
    assert math.isinf(float(processed[0, 2])) and float(processed[0, 2]) < 0
    probs = sampling_probs(logits, params)
    assert float(probs[0, 2]) == 0.0
    assert float(probs.sum()) == pytest.approx(1.0)


def test_processed_logits_honours_the_repetition_penalty_per_row() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0]])
    params = SamplingParams(repetition_penalty=2.0)
    processed = processed_logits(logits, params, token_history=[[], [0]])
    assert processed[0, 0] == pytest.approx(2.0), "an empty history penalises nothing"
    assert processed[1, 0] == pytest.approx(1.0), "a seen positive logit is divided"


def test_greedy_verification_ignores_transforms_that_cannot_move_an_argmax() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0, 0.0], [4.0, 0.0, 0.0, 0.0]])
    greedy = SamplingParams(temperature=0.0, top_k=1, top_p=0.1)
    assert verify_sequence(logits, [1], greedy).tokens == [1, 0]


# -- dispatch and batching --------------------------------------------------------------------


def test_verify_sequence_picks_rejection_sampling_for_a_sampled_request() -> None:
    logits = torch.log(rows(TARGET, 2))
    params = SamplingParams(temperature=1.0)
    generator = torch.Generator().manual_seed(5)
    emitted = [
        verify_sequence(
            logits, [3], params, draft_probs=rows(DRAFT, 1), generator=generator
        ).tokens[0]
        for _ in range(400)
    ]
    # Token 3 carries a twentieth of the target's mass against a quarter of the draft's, so a
    # draft of 3 survives about a fifth of the time.
    assert 40 < emitted.count(3) < 130
    # Conditioned on always drafting 3, token 2 can never be emitted: the target gives it less
    # mass than the draft does, so it is absent from the residual max(0, p - q). The marginal
    # over a draft actually sampled from q is still exactly p -- that is what the chi-square
    # tests above check.
    assert 2 not in emitted
    assert {0, 1, 3} <= set(emitted)


def test_verify_batch_verifies_each_sequence_with_its_own_parameters() -> None:
    greedy_logits = torch.tensor([[0.0, 5.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
    sampled_logits = torch.log(rows(TARGET, 2))
    results = verify_batch(
        [greedy_logits, sampled_logits],
        [[1], [0]],
        [SamplingParams(temperature=0.0), SamplingParams(temperature=1.0)],
        draft_probs=[None, rows(DRAFT, 1)],
        generators=[None, torch.Generator().manual_seed(1)],
    )
    assert results[0].tokens == [1, 0]
    assert results[1].num_emitted in (1, 2)


def test_verify_batch_checks_that_its_inputs_line_up() -> None:
    with pytest.raises(ValueError, match="parameter sets"):
        verify_batch([torch.zeros((1, 4))], [[], []], [SamplingParams()])
