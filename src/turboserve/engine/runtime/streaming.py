"""Incremental detokenization and stop-string handling for streamed responses.

Two problems stand between "the sampler produced token 12345" and "the client should see
these characters now", and both are solved here.

**Multi-byte tokens.** A BPE vocabulary built over bytes contains tokens that are *fragments*
of a UTF-8 character: an emoji or a CJK glyph is commonly two or three tokens, and decoding
the first one alone yields ``U+FFFD REPLACEMENT CHARACTER``. Decoding each token in
isolation therefore corrupts every non-ASCII output. :class:`IncrementalDetokenizer` decodes
a small trailing *window* of tokens instead, emits only the characters that the window grew
by, and withholds a trailing replacement character until the tokens completing it arrive.
The window is bounded, so the cost per token is constant rather than linear in the output
length -- which matters because this runs once per sequence per step.

**Stop strings.** ``SamplingParams.stop`` is defined on text, not tokens, so the engine core
cannot check it: ``"\\n\\n"`` may arrive as one token, as two, or as the tail of a token that
also carries visible text. :class:`StopStringMatcher` buffers the smallest suffix that could
still turn into a stop string, so a stop string that straddles a token boundary is caught,
and the stop string itself is never emitted to the client.

The two are combined by :class:`StreamingDecoder`, which is what the runtime holds per
in-flight request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "REPLACEMENT_CHAR",
    "DecodedDelta",
    "DetokenizerLike",
    "IncrementalDetokenizer",
    "StopStringMatcher",
    "StreamingDecoder",
    "get_tokenizer",
]

#: What a tokenizer emits for an incomplete UTF-8 sequence. A decoded window ending in this
#: character is not yet safe to emit: the bytes that complete the character are in a token
#: that has not been sampled.
REPLACEMENT_CHAR = "�"

#: Tokens kept in the re-decoded window. Large enough to span any real multi-token
#: character or byte-fallback run, small enough that the per-step decode cost does not grow
#: with the length of the response.
_WINDOW_TOKENS = 8


@runtime_checkable
class DetokenizerLike(Protocol):
    """The slice of a Hugging Face tokenizer this module needs.

    Declared structurally so tests can drive the detokenizer with a byte-level stub and so
    the engine does not import ``transformers`` for a type annotation.
    """

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        """Turn token ids into text."""
        ...


@dataclass(slots=True)
class DecodedDelta:
    """What one batch of new tokens produced for the client.

    ``text`` is what may be sent now; ``stop`` names the stop string that terminated the
    response, if any. When ``stop`` is set the text has already been truncated so the stop
    string and everything after it are excluded, which is the OpenAI convention.
    """

    text: str = ""
    stop: str | None = None
    held: str = ""
    """Characters withheld because they could still become a stop string."""

    @property
    def stopped(self) -> bool:
        """Whether a stop string fired on this batch."""
        return self.stop is not None


class IncrementalDetokenizer:
    """Turns a growing list of token ids into a growing string, one delta at a time.

    The algorithm is the one vLLM and TGI converged on. Two offsets are kept into the token
    list: ``prefix_offset``, the start of the re-decoded window, and ``read_offset``, the
    point up to which text has already been emitted. On each call the window
    ``tokens[prefix_offset:read_offset]`` and the window ``tokens[prefix_offset:]`` are both
    decoded; the delta is the difference. Decoding both from the *same* start keeps the
    comparison meaningful for tokenizers that add a leading space or strip one, which a
    naive "decode the new tokens alone" approach gets wrong.

    Why a window at all: decoding the whole sequence every step is O(n) per step and so
    O(n^2) per request, which is visible in a benchmark of long outputs.
    """

    __slots__ = (
        "_decode_kwargs",
        "_prefix_offset",
        "_read_offset",
        "_text",
        "_tokenizer",
        "_tokens",
    )

    def __init__(
        self,
        tokenizer: DetokenizerLike,
        *,
        skip_special_tokens: bool = True,
        initial_token_ids: Iterable[int] = (),
    ) -> None:
        """Create a detokenizer.

        Args:
            tokenizer: anything with a ``decode`` method.
            skip_special_tokens: drop specials (``<|im_end|>``) from the emitted text. The
                engine's stop handling works on token *ids*, so hiding the text of a special
                token never hides a stop condition.
            initial_token_ids: tokens already decoded elsewhere (the prompt, when the caller
                wants prompt-relative decoding). They seed the window without being emitted.
        """
        self._tokenizer = tokenizer
        self._decode_kwargs: dict[str, Any] = {"skip_special_tokens": skip_special_tokens}
        self._tokens: list[int] = list(initial_token_ids)
        self._prefix_offset = max(0, len(self._tokens) - _WINDOW_TOKENS)
        self._read_offset = len(self._tokens)
        self._text = ""

    @property
    def text(self) -> str:
        """Everything emitted so far (excluding any seed tokens)."""
        return self._text

    @property
    def num_tokens(self) -> int:
        """Tokens fed in, seed tokens included."""
        return len(self._tokens)

    def _decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), **self._decode_kwargs)

    def append(self, token_ids: Iterable[int]) -> str:
        """Feed tokens and return the text that became safe to emit.

        Returns ``""`` when the new tokens only extend an incomplete character; the
        characters appear in a later call, once the character is complete.
        """
        new = list(token_ids)
        if not new:
            return ""
        self._tokens.extend(new)
        prefix_text = self._decode(self._tokens[self._prefix_offset : self._read_offset])
        window_text = self._decode(self._tokens[self._prefix_offset :])
        if window_text.endswith(REPLACEMENT_CHAR):
            # The window ends mid-character. Keep both offsets where they are so the next
            # call re-decodes the same window with the completing token appended.
            return ""
        if window_text.startswith(prefix_text):
            delta = window_text[len(prefix_text) :]
        else:
            # Defensive: a tokenizer whose decode is not prefix-stable over a fixed start
            # offset (normalisers that rewrite earlier characters). Fall back to decoding
            # everything and diffing against what has already been emitted, which is always
            # correct and only costs the full decode on such tokenizers.
            full = self._decode(self._tokens)
            delta = full[len(self._text) :] if full.startswith(self._text) else full
            logger.debug("detokenizer window was not prefix-stable; fell back to a full decode")
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._tokens)
        if len(self._tokens) - self._prefix_offset > _WINDOW_TOKENS:
            self._prefix_offset = len(self._tokens) - _WINDOW_TOKENS
        self._text += delta
        return delta

    def append_token(self, token_id: int) -> str:
        """Feed a single token id. Convenience wrapper around :meth:`append`."""
        return self.append((token_id,))

    def flush(self) -> str:
        """Emit whatever remains, even if it ends in an incomplete character.

        Called when a request finishes: holding bytes back forever would truncate the
        response, so the replacement character is preferable to silence.
        """
        if self._read_offset >= len(self._tokens):
            return ""
        window_text = self._decode(self._tokens[self._prefix_offset :])
        prefix_text = self._decode(self._tokens[self._prefix_offset : self._read_offset])
        delta = window_text[len(prefix_text) :] if window_text.startswith(prefix_text) else ""
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._tokens)
        self._text += delta
        return delta


class StopStringMatcher:
    """Withholds the shortest suffix that could still grow into a stop string.

    A stop string is a property of the *text*, and text arrives in chunks whose boundaries
    have nothing to do with it. Emitting every chunk immediately would let ``"</s"`` reach
    the client a step before ``">"`` completes the stop string ``"</s>"``. So the matcher
    keeps a buffer, releases everything that can no longer be part of a match, and reports
    the match with the emitted text already truncated.

    With no stop strings configured the buffer is bypassed entirely, which is the common
    case and must not pay for this.
    """

    __slots__ = ("_buffer", "_max_len", "_stop")

    def __init__(self, stop: Iterable[str] = ()) -> None:
        self._stop: tuple[str, ...] = tuple(s for s in stop if s)
        self._max_len = max((len(s) for s in self._stop), default=0)
        self._buffer = ""

    @property
    def stop_strings(self) -> tuple[str, ...]:
        """The configured stop strings, empty strings dropped."""
        return self._stop

    @property
    def is_active(self) -> bool:
        """Whether any stop string is configured."""
        return bool(self._stop)

    @property
    def buffered(self) -> str:
        """Characters currently withheld."""
        return self._buffer

    def feed(self, text: str) -> DecodedDelta:
        """Add text and return what may be emitted now.

        Returns a :class:`DecodedDelta` whose ``stop`` is set as soon as a complete stop
        string appears; the text is cut immediately before it.
        """
        if not self._stop:
            return DecodedDelta(text=text)
        self._buffer += text
        earliest: int | None = None
        matched: str | None = None
        for candidate in self._stop:
            index = self._buffer.find(candidate)
            if index >= 0 and (earliest is None or index < earliest):
                earliest, matched = index, candidate
        if matched is not None and earliest is not None:
            emit = self._buffer[:earliest]
            self._buffer = ""
            return DecodedDelta(text=emit, stop=matched)
        keep = self._partial_suffix_len()
        emit = self._buffer[: len(self._buffer) - keep] if keep else self._buffer
        self._buffer = self._buffer[len(self._buffer) - keep :] if keep else ""
        return DecodedDelta(text=emit, held=self._buffer)

    def flush(self) -> str:
        """Release the buffer unconditionally (the request ended for another reason)."""
        pending, self._buffer = self._buffer, ""
        return pending

    def _partial_suffix_len(self) -> int:
        """Length of the longest buffer suffix that is a proper prefix of a stop string."""
        limit = min(len(self._buffer), self._max_len - 1)
        for length in range(limit, 0, -1):
            suffix = self._buffer[-length:]
            if any(candidate.startswith(suffix) for candidate in self._stop):
                return length
        return 0


@dataclass(slots=True)
class StreamingDecoder:
    """Per-request detokenization plus stop-string detection.

    This is the object the runtime keeps alive for the lifetime of a request. It owns the
    only mutable text state in the engine, so aborting a request is a dict deletion.
    """

    detokenizer: IncrementalDetokenizer
    matcher: StopStringMatcher = field(default_factory=StopStringMatcher)
    _text: str = field(default="", init=False, repr=False)

    @classmethod
    def create(
        cls,
        tokenizer: DetokenizerLike,
        *,
        stop: Iterable[str] = (),
        skip_special_tokens: bool = True,
        initial_token_ids: Iterable[int] = (),
    ) -> StreamingDecoder:
        """Build a decoder for one request from its sampling parameters."""
        return cls(
            detokenizer=IncrementalDetokenizer(
                tokenizer,
                skip_special_tokens=skip_special_tokens,
                initial_token_ids=initial_token_ids,
            ),
            matcher=StopStringMatcher(stop),
        )

    @property
    def text(self) -> str:
        """Text emitted to the client so far, stop string excluded."""
        return self._text

    def feed(self, token_ids: Iterable[int]) -> DecodedDelta:
        """Detokenize new tokens and apply stop-string matching in one call."""
        delta = self.matcher.feed(self.detokenizer.append(token_ids))
        self._text += delta.text
        return delta

    def finish(self) -> str:
        """Flush both stages when the request ends for a reason other than a stop string."""
        tail = self.detokenizer.flush()
        pending = self.matcher.flush() + tail if self.matcher.is_active else tail
        self._text += pending
        return pending


def get_tokenizer(
    path_or_id: str | Path,
    *,
    local_files_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> Any:
    """Load a Hugging Face tokenizer, with the fast implementation when one exists.

    ``transformers`` is imported here rather than at module import time: the gateway's mock
    backend, the scheduler tests and ``turboserve hwinfo`` all import this package and none
    of them needs a tokenizer.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path_or_id),
        local_files_only=local_files_only,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
