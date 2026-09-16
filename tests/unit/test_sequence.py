"""Unit tests for :mod:`turboserve.engine.core.sequence`.

The properties worth pinning down are the ones the scheduler leans on: that the
"uncomputed tokens" arithmetic covers prefill, chunked prefill and decode with one formula,
that progress never runs past the end of the sequence, and that a preempted sequence loses
its KV bookkeeping but not its generated text.
"""

from __future__ import annotations

import pytest
import torch

from turboserve.engine.core.sequence import DEFAULT_TENANT, SeqStatus, Sequence
from turboserve.engine.core.types import NO_LORA, FinishReason, SamplingParams


def make_seq(prompt: list[int] | None = None, **kwargs: object) -> Sequence:
    """A sequence with sane defaults, so each test states only what it cares about."""
    return Sequence(
        seq_id=kwargs.pop("seq_id", 0),  # type: ignore[arg-type]
        request_id=kwargs.pop("request_id", "req"),  # type: ignore[arg-type]
        prompt_token_ids=list(prompt if prompt is not None else [1, 2, 3, 4, 5]),
        **kwargs,  # type: ignore[arg-type]
    )


def test_defaults_and_validation() -> None:
    seq = make_seq()
    assert seq.tenant_id == DEFAULT_TENANT
    assert seq.lora_id == NO_LORA
    assert seq.status is SeqStatus.WAITING
    assert seq.generator is None
    assert not seq.is_finished
    with pytest.raises(ValueError, match="empty prompt"):
        make_seq([])
    with pytest.raises(ValueError, match="lora_id"):
        make_seq(lora_id=-1)


def test_token_accounting() -> None:
    seq = make_seq([1, 2, 3])
    assert (seq.num_prompt_tokens, seq.num_output_tokens, seq.num_tokens) == (3, 0, 3)
    seq.append_token(7)
    seq.append_token(8)
    assert seq.num_tokens == 5
    assert seq.token_ids == [1, 2, 3, 7, 8]
    assert seq.tokens_in_range(2, 5) == [3, 7, 8]
    assert seq.tokens_in_range(0, 2) == [1, 2]
    assert seq.tokens_in_range(3, 5) == [7, 8]
    assert seq.token_at(4) == 8
    with pytest.raises(IndexError):
        seq.token_at(5)
    with pytest.raises(ValueError, match="not inside"):
        seq.tokens_in_range(0, 6)


def test_block_tokens_requires_a_full_block() -> None:
    seq = make_seq([1, 2, 3, 4, 5])
    assert seq.block_tokens(0, 4) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="not full"):
        seq.block_tokens(1, 4)
    seq.append_token(6)
    seq.append_token(7)
    seq.append_token(8)
    assert seq.block_tokens(1, 4) == [5, 6, 7, 8]
    assert seq.num_blocks_needed(4) == 2


def test_one_formula_covers_prefill_chunking_and_decode() -> None:
    seq = make_seq(list(range(10)))
    # A fresh prompt: the whole prompt, capped by the step budget.
    assert seq.num_new_tokens_to_compute(4) == 4
    assert seq.num_new_tokens_to_compute(100) == 10
    seq.advance_computed(4)
    # Mid-prefill: only the remainder is left.
    assert seq.num_new_tokens_to_compute(100) == 6
    seq.advance_computed(6)
    # Prefill done, nothing sampled yet.
    assert seq.num_new_tokens_to_compute(100) == 0
    seq.append_token(11)
    # Decoding: exactly the one token just sampled.
    assert seq.num_new_tokens_to_compute(100) == 1
    assert seq.num_new_tokens_to_compute(0) == 0
    assert seq.is_prefill is False


def test_advance_computed_cannot_overrun() -> None:
    seq = make_seq([1, 2, 3])
    with pytest.raises(ValueError, match="only holds"):
        seq.advance_computed(4)
    with pytest.raises(ValueError, match="cannot advance by -1"):
        seq.advance_computed(-1)
    assert seq.num_computed_tokens == 0


def test_reset_for_recompute_keeps_output_but_drops_progress() -> None:
    seq = make_seq([1, 2, 3])
    seq.advance_computed(3)
    seq.append_token(9)
    seq.block_table.extend([4, 5])
    seq.timing.num_cached_prompt_tokens = 2
    seq.reset_for_recompute()
    assert seq.num_computed_tokens == 0
    assert seq.block_table == []
    assert seq.timing.num_cached_prompt_tokens == 0
    assert seq.output_token_ids == [9]
    assert seq.status is SeqStatus.PREEMPTED
    assert seq.num_preemptions == 1
    assert seq.num_uncomputed_tokens == 4


def test_check_stop_conditions() -> None:
    seq = make_seq([1], sampling=SamplingParams(max_tokens=3, stop_token_ids=[42]))
    seq.append_token(5)
    assert seq.check_stop(5, eos_token_id=99) is None
    assert seq.check_stop(42, eos_token_id=99) is FinishReason.STOP
    assert seq.check_stop(99, eos_token_id=99) is FinishReason.STOP
    seq.append_token(5)
    seq.append_token(5)
    assert seq.check_stop(5, eos_token_id=99) is FinishReason.LENGTH


def test_ignore_eos_disables_only_the_eos_stop() -> None:
    seq = make_seq([1], sampling=SamplingParams(max_tokens=5, ignore_eos=True, stop_token_ids=[7]))
    seq.append_token(2)
    assert seq.check_stop(2, eos_token_id=2) is None
    assert seq.check_stop(7, eos_token_id=2) is FinishReason.STOP


def test_finish_is_idempotent_and_stamps_once() -> None:
    seq = make_seq()
    seq.timing.t_arrival = 1.0
    seq.append_token(3, now=2.0)
    seq.finish(FinishReason.STOP, now=3.0, stop_token_id=3)
    assert seq.is_finished and seq.finish_reason is FinishReason.STOP
    assert seq.stop_token_id == 3
    seq.finish(FinishReason.ABORT, now=9.0)
    assert seq.finish_reason is FinishReason.STOP
    assert seq.timing.t_finish == 3.0
    assert seq.timing.ttft() == pytest.approx(1.0)
    assert seq.timing.e2e() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="no further tokens"):
        seq.append_token(4)


def test_first_token_timestamp_is_taken_from_the_first_append_only() -> None:
    seq = make_seq()
    seq.append_token(1, now=5.0)
    seq.append_token(2, now=6.0)
    assert seq.timing.t_first_token == 5.0


def test_seeded_generator_is_per_request_and_reproducible() -> None:
    first = make_seq(sampling=SamplingParams(seed=1234))
    second = make_seq(sampling=SamplingParams(seed=1234))
    assert first.generator is not None
    draw_a = torch.rand(4, generator=first.ensure_generator("cpu"))
    draw_b = torch.rand(4, generator=second.ensure_generator("cpu"))
    assert torch.equal(draw_a, draw_b)
    assert make_seq().ensure_generator("cpu") is None


def test_ensure_generator_is_stable_for_the_same_device() -> None:
    seq = make_seq(sampling=SamplingParams(seed=7))
    generator = seq.ensure_generator("cpu")
    assert seq.ensure_generator("cpu") is generator


def test_make_output_reports_deltas_and_cumulative_counts() -> None:
    seq = make_seq([1, 2, 3])
    seq.timing.num_cached_prompt_tokens = 2
    seq.append_token(10)
    seq.append_token(11)
    out = seq.make_output(new_token_ids=[11], text_delta=" world")
    assert out.request_id == "req"
    assert out.new_token_ids == [11]
    assert out.text_delta == " world"
    assert out.prompt_tokens == 3
    assert out.output_tokens == 2
    assert out.cached_prompt_tokens == 2
    assert out.usage() == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "cached_prompt_tokens": 2,
    }
    assert not out.finished
    seq.finish(FinishReason.LENGTH)
    assert seq.make_output().finished


def test_sequences_are_identity_hashable() -> None:
    first, second = make_seq(), make_seq()
    assert first != second
    assert len({first, second, first}) == 2


def test_status_terminality() -> None:
    assert SeqStatus.FINISHED.is_terminal
    assert not SeqStatus.RUNNING.is_terminal
    assert str(SeqStatus.WAITING) == "waiting"
