"""Prompt-lookup drafting: speculate by copying from the text the sequence already contains.

The idea is from prompt-lookup decoding (Saxena, *Prompt Lookup Decoding*, 2023, and the
same mechanism as "n-gram speculative decoding" in vLLM): take the last ``n`` tokens of a
sequence, find where that exact n-gram occurred earlier in the same sequence, and propose
whatever followed it. No model, no weights, no KV cache -- the drafter's entire cost is a
few vectorised comparisons over the sequence's own token ids.

It is a narrow trick and it is honest about being one. It proposes nothing at all unless the
sequence repeats itself, so it is useful exactly where repetition is the point: summarising,
editing or answering questions about a document that is in the prompt, code completion
inside a file, and structured output whose field names recur. Where the continuation is
genuinely novel it proposes nothing and the engine falls straight back to ordinary decoding,
which costs one comparison pass and no forward pass.

The proposal carries no probabilities. That does **not** restrict it to greedy requests: the
verifier reads a missing draft distribution as a point mass on the proposed token, which is
a distribution like any other, so modified rejection sampling still emits tokens distributed
exactly as the target's -- see :mod:`turboserve.engine.spec.verifier`.

Two knobs decide what counts as a match. ``max_ngram`` is tried first and the match is taken
from the *most recent* occurrence, because a longer and more recent context is the better
predictor of what comes next; ``min_ngram`` bounds how little context is allowed to justify a
guess, and lowering it to 1 makes the drafter propose after any repeated token, which is
mostly noise.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from turboserve.engine.spec.drafter import DraftProposal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable
    from collections.abc import Sequence as SequenceABC

    from turboserve.engine.core.sequence import Sequence

logger = logging.getLogger(__name__)

__all__ = ["NgramDrafter", "find_ngram_continuation"]


def find_ngram_continuation(
    tokens: SequenceABC[int],
    *,
    num_tokens: int,
    min_ngram: int = 2,
    max_ngram: int = 4,
) -> list[int]:
    """Return up to ``num_tokens`` tokens that followed the most recent repeat of the tail.

    The tail of length ``n`` is searched for at every earlier offset, longest ``n`` first, and
    the continuation is read from just after the latest occurrence found. An empty list means
    the sequence does not repeat itself in a way this drafter can exploit. Overlapping matches
    are allowed on purpose: a run of one repeated token matches itself shifted by one, and the
    continuation that falls out of that is the repeated token, which is the right guess.

    The search is vectorised: for a candidate length ``n`` the whole sequence is compared
    against the tail with ``n`` shifted equality tests, so the cost is ``O(n * len(tokens))``
    tensor work rather than a Python scan over every offset.
    """
    if num_tokens <= 0 or min_ngram < 1 or max_ngram < min_ngram:
        return []
    total = len(tokens)
    ids = torch.tensor([int(token) for token in tokens], dtype=torch.long)
    for size in range(min(max_ngram, total - 1), min_ngram - 1, -1):
        span = total - size  # number of candidate start offsets, 0 .. span-1
        if span <= 0:
            continue
        tail = ids[total - size :]
        matches = torch.ones(span, dtype=torch.bool)
        for offset in range(size):
            matches &= ids[offset : offset + span] == tail[offset]
        found = torch.nonzero(matches, as_tuple=False)
        if found.numel() == 0:
            continue
        start = int(found[-1, 0]) + size
        return [int(token) for token in ids[start : start + num_tokens].tolist()]
    return []


class NgramDrafter:
    """Propose continuations copied from each sequence's own history.

    Stateless with respect to the KV cache -- there is nothing to allocate, roll back or
    evict -- so the only state it keeps is per-sequence hit accounting, which is what makes
    it possible to tell "speculation did not help" from "speculation was never attempted"
    when reading a benchmark result.
    """

    name = "ngram"

    def __init__(self, *, min_ngram: int = 2, max_ngram: int = 4) -> None:
        if min_ngram < 1:
            raise ValueError(f"min_ngram must be at least 1, got {min_ngram}")
        if max_ngram < min_ngram:
            raise ValueError(f"max_ngram {max_ngram} is below min_ngram {min_ngram}")
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self._num_proposals = 0
        self._num_draft_tokens = 0
        self._num_hits = 0
        self._num_lookups = 0
        self._tracked: set[int] = set()

    def propose(self, seqs: SequenceABC[Sequence], k: int) -> DraftProposal:
        """Look each sequence's tail up in its own history and propose the continuation."""
        if k <= 0 or not seqs:
            return DraftProposal.empty(len(seqs))
        self._num_proposals += 1
        lists: list[list[int]] = []
        for seq in seqs:
            proposed = find_ngram_continuation(
                seq.token_ids,
                num_tokens=k,
                min_ngram=self.min_ngram,
                max_ngram=self.max_ngram,
            )
            self._num_lookups += 1
            self._num_hits += 1 if proposed else 0
            self._tracked.add(seq.seq_id)
            lists.append(proposed)
        proposal = DraftProposal.from_lists(lists)
        self._num_draft_tokens += proposal.total_drafts
        return proposal

    def release(self, seq_id: int) -> None:
        """Stop tracking a finished sequence. The lifetime counters are cumulative and stay."""
        self._tracked.discard(seq_id)

    def prune(self, live_seq_ids: Iterable[int]) -> int:
        """Stop tracking sequences that are no longer in flight. Returns how many."""
        live = set(live_seq_ids)
        stale = [seq_id for seq_id in self._tracked if seq_id not in live]
        self._tracked.difference_update(stale)
        return len(stale)

    def reset(self) -> None:
        """Forget which sequences are in flight; the lifetime counters are kept.

        They are kept because they are what a benchmark reads *after* the run, by which point
        the engine has closed and every sequence has been released. A counter that a normal
        shutdown zeroes is a counter nobody can use.
        """
        self._tracked.clear()

    def stats(self) -> dict[str, int | float]:
        """Counters describing how often the lookup found anything to propose."""
        looked_up = self._num_lookups
        return {
            "draft_proposals": self._num_proposals,
            "draft_tokens_proposed": self._num_draft_tokens,
            "draft_lookups": looked_up,
            "draft_lookup_hits": self._num_hits,
            "draft_lookup_hit_rate": self._num_hits / looked_up if looked_up else 0.0,
            "draft_sequences": len(self._tracked),
        }

    def __repr__(self) -> str:
        return f"NgramDrafter(min_ngram={self.min_ngram}, max_ngram={self.max_ngram})"
