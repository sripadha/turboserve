"""Turning a step's logits into tokens, for a whole batch of differently configured requests.

Continuous batching means one forward pass serves requests with unrelated sampling
settings: a greedy evaluation run, a ``temperature=0.9, top_p=0.95`` chat request and a
seeded reproducibility check all land in the same ``[batch, vocab]`` logits tensor. So every
step here is written as a batched tensor operation with a per-row parameter vector rather
than a Python loop over requests; the only per-row work left is drawing from a request's own
generator, which cannot be batched because ``torch.multinomial`` takes a single generator.

Order of operations, and why it is this order:

1. **Repetition penalty** on the raw logits, because it is a property of what the model has
   already said, independent of how sharply the distribution is later shaped.
2. **Log-probabilities** are captured here, after the penalty and *before* temperature and
   truncation. They are then the model's own belief about the token it emitted, comparable
   across requests that used different sampling settings -- which is what an evaluation, a
   drift check or a spot audit of a response actually wants. Reporting the post-truncation
   distribution instead would make every reported number a function of the request's own
   ``top_p``.
3. **Temperature**, then **top-k**, then **top-p**. Top-k before top-p is the usual
   convention and the cheaper order: k is a fixed count and bounds the sort that top-p would
   otherwise do over the whole vocabulary.
4. **Sample**, or take the argmax for greedy rows. Filtering never removes the maximum, so
   greedy rows can be resolved after filtering without a separate code path.

The helper functions are pure: each returns a new tensor and leaves its input alone, so a
test can assert on the effect of one stage in isolation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from turboserve.engine.core.sequence import Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

    from turboserve.engine.core.types import SamplingParams

logger = logging.getLogger(__name__)

__all__ = [
    "Sampler",
    "SamplerOutput",
    "apply_repetition_penalty",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
]


@dataclass(slots=True)
class SamplerOutput:
    """One token per row of the batch, plus optional log-probabilities.

    Plain Python lists rather than tensors: the values are immediately needed on the host
    -- to append to sequences, to check stop conditions and to stream to clients -- so the
    single device synchronisation happens here, once per step, instead of being scattered
    over the call sites.
    """

    token_ids: list[int]
    logprobs: list[float] | None = field(default=None)
    """Log-probability of each chosen token, or ``None`` when no request asked for them.

    Rows whose request did not set ``logprobs`` are ``nan`` so the list stays aligned with
    :attr:`token_ids`.
    """

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)


def apply_repetition_penalty(
    logits: torch.Tensor,
    penalties: SequenceABC[float],
    token_history: SequenceABC[SequenceABC[int]],
) -> torch.Tensor:
    """Divide the logits of already-seen tokens by the request's penalty.

    Follows the convention introduced by CTRL (Keskar et al., 2019) and used by
    ``transformers``: a positive logit is divided by the penalty and a negative one is
    multiplied, so a penalty above 1 always moves the token towards being less likely
    regardless of the sign. Rows with penalty 1.0 or an empty history are untouched.

    The padded index matrix is filled with each row's *first* history token rather than a
    dummy id: ``scatter_`` leaves the result of duplicate indices unspecified, and padding
    with a token whose penalised value is already being written makes every duplicate write
    the same number.
    """
    rows = [
        index
        for index, (penalty, history) in enumerate(zip(penalties, token_history, strict=True))
        if penalty != 1.0 and len(history) > 0
    ]
    if not rows:
        return logits
    device = logits.device
    width = max(len(token_history[index]) for index in rows)
    index_matrix = torch.empty((len(rows), width), dtype=torch.long, device=device)
    for row, index in enumerate(rows):
        history = list(token_history[index])
        index_matrix[row] = history[0]
        index_matrix[row, : len(history)] = torch.tensor(history, dtype=torch.long, device=device)
    row_index = torch.tensor(rows, dtype=torch.long, device=device)
    penalty_column = torch.tensor(
        [float(penalties[index]) for index in rows], dtype=logits.dtype, device=device
    ).unsqueeze(1)
    sub = logits.index_select(0, row_index)
    gathered = sub.gather(1, index_matrix)
    updated = torch.where(gathered < 0, gathered * penalty_column, gathered / penalty_column)
    sub = sub.scatter(1, index_matrix, updated)
    return logits.index_copy(0, row_index, sub)


def apply_temperature(logits: torch.Tensor, temperatures: SequenceABC[float]) -> torch.Tensor:
    """Scale each row by its temperature; rows with temperature 0 (greedy) are left alone.

    Greedy rows are skipped rather than special-cased later because dividing by zero would
    produce infinities that then poison the softmax of the whole batch.
    """
    values = [temp if temp > 0.0 else 1.0 for temp in temperatures]
    if all(value == 1.0 for value in values):
        return logits
    scale = torch.tensor(values, dtype=logits.dtype, device=logits.device).unsqueeze(1)
    return logits / scale


def apply_top_k(logits: torch.Tensor, top_k: SequenceABC[int]) -> torch.Tensor:
    """Mask everything outside each row's ``k`` most likely tokens. ``k == 0`` means no filter."""
    vocab = logits.shape[-1]
    values = [vocab if k <= 0 else min(k, vocab) for k in top_k]
    if all(value >= vocab for value in values):
        return logits
    k_max = max(values)
    kept, _ = torch.topk(logits, k_max, dim=-1)
    k_index = torch.tensor(values, dtype=torch.long, device=logits.device).clamp(max=k_max) - 1
    threshold = kept.gather(1, k_index.unsqueeze(1))
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: SequenceABC[float]) -> torch.Tensor:
    """Keep the smallest set of most likely tokens whose probability mass reaches ``p``.

    The mask is ``cumulative_before_this_token > p``: a token survives when the mass strictly
    ahead of it has not yet reached ``p``, which keeps the token that crosses the threshold
    and guarantees at least the top-1 token survives for any ``p > 0``.
    """
    values = [float(p) for p in top_p]
    if all(value >= 1.0 for value in values):
        return logits
    threshold = torch.tensor(values, dtype=logits.dtype, device=logits.device).unsqueeze(1)
    sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    cumulative_before = probs.cumsum(dim=-1) - probs
    remove_sorted = cumulative_before > threshold
    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(1, sorted_index, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


class Sampler:
    """Batched token sampling with per-request parameters and per-request generators.

    Stateless: one instance can serve every step and every request. It is a class rather
    than a function so the runtime can swap in a different policy (a speculative-decoding
    verifier, for example) behind the same call.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        params: SequenceABC[SamplingParams],
        *,
        generators: SequenceABC[torch.Generator | None] | None = None,
        token_history: SequenceABC[SequenceABC[int]] | None = None,
    ) -> SamplerOutput:
        """Alias of :meth:`forward`."""
        return self.forward(logits, params, generators=generators, token_history=token_history)

    def forward(
        self,
        logits: torch.Tensor,
        params: SequenceABC[SamplingParams],
        *,
        generators: SequenceABC[torch.Generator | None] | None = None,
        token_history: SequenceABC[SequenceABC[int]] | None = None,
    ) -> SamplerOutput:
        """Sample one token per row of ``logits``.

        ``logits`` is ``[batch, vocab]``, one row per sampling sequence in the step, and is
        never modified: the batch's logits belong to the caller, which may also want them
        for speculative verification or for a logprob export.
        """
        if logits.dim() != 2:
            raise ValueError(f"logits must be [batch, vocab], got shape {tuple(logits.shape)}")
        batch = int(logits.shape[0])
        if len(params) != batch:
            raise ValueError(f"{len(params)} sampling params for {batch} rows of logits")
        if generators is not None and len(generators) != batch:
            raise ValueError(f"{len(generators)} generators for {batch} rows of logits")
        if token_history is not None and len(token_history) != batch:
            raise ValueError(f"{len(token_history)} histories for {batch} rows of logits")
        if batch == 0:
            return SamplerOutput(token_ids=[])

        work = logits.detach().to(dtype=torch.float32, copy=True)
        if token_history is not None:
            work = apply_repetition_penalty(
                work, [p.repetition_penalty for p in params], token_history
            )

        want_logprobs = any(p.logprobs for p in params)
        base_logprobs = torch.log_softmax(work, dim=-1) if want_logprobs else None

        work = apply_temperature(work, [p.temperature for p in params])
        work = apply_top_k(work, [p.top_k for p in params])
        work = apply_top_p(work, [p.top_p for p in params])

        tokens = self._draw(work, params, generators)
        chosen = tokens.unsqueeze(1)
        logprob_values: list[float] | None = None
        if base_logprobs is not None:
            selected = base_logprobs.gather(1, chosen).squeeze(1)
            logprob_values = [
                float(value) if p.logprobs else float("nan")
                for value, p in zip(selected.tolist(), params, strict=True)
            ]
        return SamplerOutput(token_ids=[int(t) for t in tokens.tolist()], logprobs=logprob_values)

    def sample_sequences(
        self,
        logits: torch.Tensor,
        sequences: SequenceABC[Sequence],
    ) -> SamplerOutput:
        """Sample for a list of :class:`~turboserve.engine.core.sequence.Sequence` objects.

        Collects each sequence's parameters, its device-matched generator and -- only when
        the request actually uses a repetition penalty -- its token history, so that the
        common case costs no list copying.
        """
        params = [seq.sampling for seq in sequences]
        generators = [seq.ensure_generator(logits.device) for seq in sequences]
        history: list[SequenceABC[int]] = [
            seq.token_ids if seq.sampling.repetition_penalty != 1.0 else () for seq in sequences
        ]
        return self.forward(logits, params, generators=generators, token_history=history)

    def _draw(
        self,
        logits: torch.Tensor,
        params: SequenceABC[SamplingParams],
        generators: SequenceABC[torch.Generator | None] | None,
    ) -> torch.Tensor:
        """Draw one token per row: argmax for greedy rows, multinomial for the rest.

        Rows without their own generator are drawn in a single ``multinomial`` call on the
        global RNG; a row with a seeded generator gets its own call, because a generator
        applies to a whole call and mixing them would make a seeded request's output depend
        on its batch neighbours.
        """
        device = logits.device
        tokens = logits.argmax(dim=-1)
        stochastic = [index for index, p in enumerate(params) if not p.is_greedy]
        if not stochastic:
            return tokens
        probs = logits.softmax(dim=-1)
        seeded: list[tuple[int, torch.Generator]] = []
        shared: list[int] = []
        for index in stochastic:
            generator = generators[index] if generators is not None else None
            if generator is None:
                shared.append(index)
                continue
            if generator.device.type != device.type:
                raise ValueError(
                    f"row {index} has a generator on {generator.device} but the logits are on "
                    f"{device}; build it with Sequence.ensure_generator(logits.device)"
                )
            seeded.append((index, generator))
        if shared:
            rows = torch.tensor(shared, dtype=torch.long, device=device)
            drawn = torch.multinomial(probs.index_select(0, rows), 1).squeeze(1)
            tokens = tokens.index_copy(0, rows, drawn)
        for index, generator in seeded:
            drawn = torch.multinomial(probs[index], 1, generator=generator)
            tokens[index] = drawn[0]
        return tokens
