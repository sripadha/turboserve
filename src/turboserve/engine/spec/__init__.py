"""Speculative decoding: drafters, rejection-sampling verification and the spec engine.

Start at :mod:`turboserve.engine.spec.spec_engine` for the step, and at
:mod:`turboserve.engine.spec.verifier` for the guarantee that the step changes nothing a
client can observe. ``docs/speculative-decoding.md`` walks through both.
"""

from __future__ import annotations

from turboserve.engine.spec.drafter import Drafter, DraftProposal, ModelDrafter
from turboserve.engine.spec.ngram import NgramDrafter, find_ngram_continuation
from turboserve.engine.spec.spec_engine import (
    DEFAULT_NUM_SPECULATIVE_TOKENS,
    SpecStats,
    SpeculativeConfig,
    SpeculativeLLMEngine,
)
from turboserve.engine.spec.verifier import (
    Verification,
    processed_logits,
    sampling_probs,
    verify_batch,
    verify_greedy,
    verify_rejection_sampling,
    verify_sequence,
)

__all__ = [
    "DEFAULT_NUM_SPECULATIVE_TOKENS",
    "DraftProposal",
    "Drafter",
    "ModelDrafter",
    "NgramDrafter",
    "SpecStats",
    "SpeculativeConfig",
    "SpeculativeLLMEngine",
    "Verification",
    "find_ngram_continuation",
    "processed_logits",
    "sampling_probs",
    "verify_batch",
    "verify_greedy",
    "verify_rejection_sampling",
    "verify_sequence",
]
