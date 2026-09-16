"""Verification: turning ``k`` draft tokens into between 1 and ``k+1`` target tokens.

Speculative decoding is only worth doing if it changes nothing the client can observe. The
whole of that guarantee lives in this module, in two functions:

:func:`verify_greedy`
    For a greedy request, a draft token is kept exactly when it is the token the target
    model would itself have chosen. The emitted continuation is therefore character-for-
    character the target's greedy continuation, and the only thing speculation changed is
    how many forward passes it took to produce it.

:func:`verify_rejection_sampling`
    For a sampled request, modified rejection sampling (Leviathan et al., *Fast Inference
    from Transformers via Speculative Decoding*, ICML 2023, and Chen et al., *Accelerating
    Large Language Model Decoding with Speculative Sampling*, 2023) keeps draft token
    ``x`` drawn from the draft distribution ``q`` with probability ``min(1, p(x)/q(x))``,
    and on a rejection draws the replacement from the normalised residual
    ``max(0, p - q)``. The emitted token is then distributed exactly as ``p``, the target's
    own next-token distribution -- not approximately, and not only in expectation.

The proof of that last claim is short enough to keep next to the code. Writing ``a(x) =
min(1, p(x)/q(x))`` for the acceptance probability of ``x``, the probability that the
scheme emits ``x`` is::

    q(x)a(x)  +  (sum_y q(y)(1 - a(y))) * residual(x) / sum_z residual(z)

The first term is "drafted ``x`` and kept it" and equals ``min(q(x), p(x))``. The rejection
mass ``sum_y q(y)(1 - a(y))`` equals ``sum_y max(0, q(y) - p(y))``, which is the same number
as ``sum_z max(0, p(z) - q(z)) = sum_z residual(z)`` because both are the total variation
distance between ``p`` and ``q``. Those two factors cancel, leaving ``min(q(x), p(x)) +
max(0, p(x) - q(x)) = p(x)``. :func:`turboserve.engine.spec.verifier.verify_rejection_sampling`
is tested against that identity with a chi-square test in
``tests/unit/test_spec_verifier.py``.

A drafter that cannot report a distribution (the n-gram drafter, which proposes tokens
copied from the context rather than sampled from a model) passes ``draft_probs=None``. That
is read as a point mass on the drafted token: acceptance probability ``p(x)``, residual
``p`` with ``x`` removed. The identity above still holds -- a point mass is a distribution
like any other -- so an n-gram-drafted sampled request is exact too, which is why this
module does not restrict n-gram drafting to greedy requests.

Every function here is pure: it takes tensors and returns a small record, touches no
sequence, no cache and no model, and is therefore the one part of speculative decoding that
can be tested without loading anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from turboserve.engine.core.sampler import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence as SequenceABC

    from turboserve.engine.core.types import SamplingParams

__all__ = [
    "Verification",
    "processed_logits",
    "sampling_probs",
    "verify_batch",
    "verify_greedy",
    "verify_rejection_sampling",
    "verify_sequence",
]


@dataclass(frozen=True, slots=True)
class Verification:
    """The outcome of verifying one sequence's draft.

    ``accepted_tokens`` is a prefix of the draft (possibly empty) and ``bonus_token`` is the
    one token that is always emitted: either the target's own continuation after the last
    accepted draft token, or the replacement drawn from the residual distribution after a
    rejection. A step therefore always makes progress -- even a draft in which nothing is
    accepted emits one token, exactly as ordinary decoding would have.
    """

    accepted_tokens: tuple[int, ...]
    bonus_token: int
    num_drafted: int

    def __post_init__(self) -> None:
        if self.num_drafted < 0:
            raise ValueError(f"num_drafted must be non-negative, got {self.num_drafted}")
        if len(self.accepted_tokens) > self.num_drafted:
            raise ValueError(
                f"{len(self.accepted_tokens)} accepted tokens out of {self.num_drafted} drafted"
            )

    @property
    def num_accepted(self) -> int:
        """How many draft tokens survived verification."""
        return len(self.accepted_tokens)

    @property
    def num_emitted(self) -> int:
        """Tokens this verification contributes to the sequence: accepted plus the bonus."""
        return len(self.accepted_tokens) + 1

    @property
    def all_accepted(self) -> bool:
        """Whether the whole draft was kept, so the bonus came from the last target row."""
        return len(self.accepted_tokens) == self.num_drafted

    @property
    def tokens(self) -> list[int]:
        """The tokens to append to the sequence, in order."""
        return [*self.accepted_tokens, self.bonus_token]


def processed_logits(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    token_history: SequenceABC[SequenceABC[int]] | None = None,
) -> torch.Tensor:
    """Apply the sampler's transform chain to ``[rows, vocab]`` logits of one sequence.

    The transforms and their order are exactly
    :meth:`turboserve.engine.core.sampler.Sampler.forward`'s -- repetition penalty, then
    temperature, then top-k, then top-p -- because the target distribution a draft is
    verified against has to be the distribution the engine would have sampled from without
    speculation. Reusing the sampler's own functions rather than re-deriving them here is
    what makes that guarantee hold when either changes.

    ``token_history`` is per row: the ``i``-th entry is everything the sequence contained
    when the ``i``-th query token was fed, which is what a repetition penalty is defined
    against. Pass ``None`` when no row uses a penalty.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [rows, vocab], got shape {tuple(logits.shape)}")
    work = logits.detach().to(dtype=torch.float32, copy=True)
    rows = int(work.shape[0])
    if token_history is not None and params.repetition_penalty != 1.0:
        if len(token_history) != rows:
            raise ValueError(f"{len(token_history)} histories for {rows} rows of logits")
        work = apply_repetition_penalty(work, [params.repetition_penalty] * rows, token_history)
    work = apply_temperature(work, [params.temperature] * rows)
    work = apply_top_k(work, [params.top_k] * rows)
    work = apply_top_p(work, [params.top_p] * rows)
    return work


def sampling_probs(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    token_history: SequenceABC[SequenceABC[int]] | None = None,
) -> torch.Tensor:
    """The normalised distribution the engine would sample each row from.

    Truncated tokens carry a ``-inf`` logit and therefore exactly zero probability, which
    matters for rejection sampling: a draft token outside the target's top-p set is rejected
    with certainty rather than with a merely small probability.
    """
    return processed_logits(logits, params, token_history=token_history).softmax(dim=-1)


def verify_greedy(
    target_logits: torch.Tensor,
    draft_tokens: SequenceABC[int],
) -> Verification:
    """Accept the longest prefix of the draft the target would itself have produced.

    ``target_logits`` has one row per query position: ``k`` rows predicting the ``k`` draft
    tokens and one final row predicting the token after the whole draft. The result is
    identical to running the target alone for ``num_emitted`` steps, which is why the
    speculative engine's greedy output is compared token-for-token against the ordinary
    engine's in ``tests/unit/test_spec_engine.py``.
    """
    num_drafted = len(draft_tokens)
    if target_logits.dim() != 2 or int(target_logits.shape[0]) != num_drafted + 1:
        raise ValueError(
            f"target_logits must be [{num_drafted + 1}, vocab] for {num_drafted} draft tokens, "
            f"got shape {tuple(target_logits.shape)}"
        )
    greedy = [int(token) for token in target_logits.argmax(dim=-1).tolist()]
    accepted: list[int] = []
    for index, drafted in enumerate(draft_tokens):
        if greedy[index] != int(drafted):
            break
        accepted.append(int(drafted))
    return Verification(
        accepted_tokens=tuple(accepted),
        bonus_token=greedy[len(accepted)],
        num_drafted=num_drafted,
    )


def verify_rejection_sampling(
    target_probs: torch.Tensor,
    draft_tokens: SequenceABC[int],
    draft_probs: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> Verification:
    """Modified rejection sampling: emit tokens distributed exactly as ``target_probs``.

    Args:
        target_probs: ``[k+1, vocab]`` normalised target distributions, row ``j`` being the
            distribution for the ``j``-th draft position and row ``k`` the one used for the
            bonus token when the whole draft is accepted.
        draft_tokens: the ``k`` proposed token ids.
        draft_probs: ``[k, vocab]`` draft distributions, or ``None`` for a drafter that
            proposes deterministically (treated as a point mass on the drafted token).
        generator: the request's RNG, so a seeded request stays reproducible and independent
            of its batch neighbours. Must live on the same device as ``target_probs``.

    The acceptance test is written as ``u * q(x) < p(x)`` rather than ``u < p(x)/q(x)`` so
    that a draft token the target has truncated to zero probability, or one the draft
    distribution itself gives zero mass, needs no special case.
    """
    num_drafted = len(draft_tokens)
    if target_probs.dim() != 2 or int(target_probs.shape[0]) != num_drafted + 1:
        raise ValueError(
            f"target_probs must be [{num_drafted + 1}, vocab] for {num_drafted} draft tokens, "
            f"got shape {tuple(target_probs.shape)}"
        )
    device = target_probs.device
    vocab = int(target_probs.shape[1])
    if draft_probs is not None:
        if tuple(draft_probs.shape) != (num_drafted, vocab):
            raise ValueError(
                f"draft_probs must be [{num_drafted}, {vocab}], got {tuple(draft_probs.shape)}"
            )
        if draft_probs.device != device:
            raise ValueError(
                f"draft_probs are on {draft_probs.device} but target_probs are on {device}"
            )
    if generator is not None and generator.device.type != device.type:
        raise ValueError(
            f"generator is on {generator.device} but the probabilities are on {device}; "
            "build it with Sequence.ensure_generator(device)"
        )

    num_accepted = num_drafted
    if num_drafted:
        drafted = torch.tensor([int(t) for t in draft_tokens], dtype=torch.long, device=device)
        column = drafted.unsqueeze(1)
        target_at_draft = target_probs[:num_drafted].gather(1, column).squeeze(1)
        if draft_probs is None:
            draft_at_draft = torch.ones(num_drafted, dtype=torch.float32, device=device)
        else:
            draft_at_draft = draft_probs.gather(1, column).squeeze(1).to(torch.float32)
        uniform = torch.rand(num_drafted, generator=generator, device=device)
        # One host transfer for the whole draft: the loop below must know where the first
        # rejection is, and asking the device that question token by token would put a
        # synchronisation between every pair of draft positions.
        keep = (uniform * draft_at_draft < target_at_draft).tolist()
        for index, accepted in enumerate(keep):
            if not accepted:
                num_accepted = index
                break

    if num_accepted == num_drafted:
        bonus = _draw(target_probs[num_drafted], generator)
        return Verification(
            accepted_tokens=tuple(int(t) for t in draft_tokens),
            bonus_token=bonus,
            num_drafted=num_drafted,
        )

    residual = target_probs[num_accepted].clone()
    if draft_probs is None:
        residual[int(draft_tokens[num_accepted])] = 0.0
    else:
        residual = (residual - draft_probs[num_accepted]).clamp_min_(0.0)
    if float(residual.sum()) <= 0.0:
        # p and q agree everywhere the target has mass, so there is nothing left to prefer;
        # falling back to the target distribution keeps the emitted token distributed as p.
        residual = target_probs[num_accepted]
    return Verification(
        accepted_tokens=tuple(int(t) for t in draft_tokens[:num_accepted]),
        bonus_token=_draw(residual, generator),
        num_drafted=num_drafted,
    )


def verify_sequence(
    target_logits: torch.Tensor,
    draft_tokens: SequenceABC[int],
    params: SamplingParams,
    *,
    draft_probs: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    token_history: SequenceABC[SequenceABC[int]] | None = None,
) -> Verification:
    """Verify one sequence, choosing greedy or rejection sampling from its parameters.

    This is the entry point the engine uses. Greedy requests never build a probability
    tensor: an ``argmax`` over the raw logits answers the question, and the transforms that
    would have been applied (temperature, top-k, top-p) cannot move an argmax.
    """
    if params.is_greedy:
        history = token_history if params.repetition_penalty != 1.0 else None
        return verify_greedy(
            processed_logits(target_logits, params, token_history=history), draft_tokens
        )
    probs = sampling_probs(target_logits, params, token_history=token_history)
    return verify_rejection_sampling(probs, draft_tokens, draft_probs, generator=generator)


def verify_batch(
    target_logits: SequenceABC[torch.Tensor],
    draft_tokens: SequenceABC[SequenceABC[int]],
    params: SequenceABC[SamplingParams],
    *,
    draft_probs: SequenceABC[torch.Tensor | None] | None = None,
    generators: SequenceABC[torch.Generator | None] | None = None,
    token_history: SequenceABC[SequenceABC[SequenceABC[int]]] | None = None,
) -> list[Verification]:
    """Verify a whole step, one sequence at a time.

    The loop is deliberate. Sequences in a speculative step have different draft lengths
    (an n-gram drafter finds no continuation for some of them, and a sequence whose draft
    pool is full gets none), different sampling parameters and their own generators, so a
    padded batched formulation would have to mask all three. The work per sequence is a
    handful of small tensor operations next to two model forward passes, and the loop keeps
    the acceptance rule in the form the papers state it.
    """
    count = len(target_logits)
    if len(draft_tokens) != count or len(params) != count:
        raise ValueError(
            f"{count} logit blocks, {len(draft_tokens)} drafts and {len(params)} parameter sets"
        )
    if draft_probs is not None and len(draft_probs) != count:
        raise ValueError(f"{len(draft_probs)} draft distributions for {count} sequences")
    if generators is not None and len(generators) != count:
        raise ValueError(f"{len(generators)} generators for {count} sequences")
    if token_history is not None and len(token_history) != count:
        raise ValueError(f"{len(token_history)} history blocks for {count} sequences")
    return [
        verify_sequence(
            target_logits[index],
            draft_tokens[index],
            params[index],
            draft_probs=None if draft_probs is None else draft_probs[index],
            generator=None if generators is None else generators[index],
            token_history=None if token_history is None else token_history[index],
        )
        for index in range(count)
    ]


def _draw(weights: torch.Tensor, generator: torch.Generator | None) -> int:
    """Draw one index from non-negative, not necessarily normalised, ``weights``."""
    return int(torch.multinomial(weights, 1, generator=generator)[0])
