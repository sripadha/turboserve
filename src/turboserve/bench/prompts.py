"""Prompt sources for the benchmark: seeded synthetic text, ShareGPT, and plain files.

A serving benchmark is only reproducible if the *workload* is reproducible, and the part
of a workload that is hardest to pin down is the prompt. Three properties matter here:

* **Exact lengths.** A run advertised as "128-1024 input tokens" must actually send that
  many tokens, or the throughput figures are measuring a different workload than the table
  claims. Random text does not give exact counts, so :class:`SyntheticPromptBuilder`
  samples *token ids* and drives them to a re-tokenisation fixpoint (see
  :func:`_stabilise`), reporting honestly per prompt whether the fixpoint was reached.
* **A controllable shared prefix.** The prefix-cache scenario needs every request to open
  with the same N tokens and diverge afterwards. Built once and prepended verbatim, so the
  shared region is identical at the token level -- which is the level the block hashes in
  the prefix cache are computed at.
* **Determinism.** Everything is driven by one ``random.Random(seed)``; two runs with the
  same profile send byte-identical prompts, so a change in a number is a change in the
  system and not in the input.

The builder needs only ``encode`` and ``decode`` from its tokenizer (see
:class:`TokenizerLike`), so unit tests can drive it with a toy tokenizer and the real runs
pass a Hugging Face ``AutoTokenizer``. Nothing here imports torch.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_STABILISE_ROUNDS",
    "BenchPrompt",
    "PromptError",
    "PromptSource",
    "PromptSpec",
    "SyntheticPromptBuilder",
    "TokenizerLike",
    "build_prompts",
    "load_file_prompts",
    "load_sharegpt_prompts",
]

#: How many re-tokenisation rounds :func:`_stabilise` tries before giving up on a fixpoint.
#: Byte-level BPE tokenizers converge in one or two; the bound exists so a tokenizer that
#: never converges costs a few milliseconds rather than hanging a benchmark.
MAX_STABILISE_ROUNDS = 8

PromptSource = Literal["synthetic", "sharegpt", "file"]


class PromptError(ValueError):
    """A prompt source is unusable: missing file, wrong shape, or impossible lengths."""


@runtime_checkable
class TokenizerLike(Protocol):
    """The two tokenizer methods the prompt builders use.

    Deliberately narrower than ``transformers.PreTrainedTokenizerBase``: the builders need
    to turn ids into text and back, nothing else. A tokenizer is also allowed to expose
    ``vocab_size``/``all_special_ids``, which :func:`_sampling_pool` uses when present.
    """

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Token ids for ``text``."""
        ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = ...) -> str:
        """Text for ``token_ids``."""
        ...


@dataclass(frozen=True, slots=True)
class BenchPrompt:
    """One request's input, ready to be turned into a :class:`GenerateRequest`.

    Both representations are kept. ``token_ids`` is authoritative -- it is what the
    prompt-token counts in the result file mean, and what a prefix-cache experiment must
    send, because re-tokenising ``text`` can merge tokens across the boundary between the
    shared prefix and the unique suffix and destroy the very sharing being measured.
    ``text`` is carried for backends that only accept strings (a remote OpenAI-compatible
    server) and for eyeballing a prompt during debugging.
    """

    prompt_id: str
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    max_tokens: int = 0
    tenant: str = "bench"
    shared_prefix_tokens: int = 0
    source: str = "synthetic"
    round_trip_exact: bool = True
    """Whether ``encode(text) == token_ids``; false means the two views differ in length."""

    @property
    def num_prompt_tokens(self) -> int:
        """Prompt length in tokens, as the engine will see it."""
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        """Compact description for logs and for a run's ``config`` block (no raw ids)."""
        return {
            "prompt_id": self.prompt_id,
            "prompt_tokens": self.num_prompt_tokens,
            "max_tokens": self.max_tokens,
            "tenant": self.tenant,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "source": self.source,
            "round_trip_exact": self.round_trip_exact,
        }


class PromptSpec(BaseModel):
    """A fully specified prompt workload: which source, how many, how long.

    Validated strictly so that a scenario cannot ask for something contradictory -- a
    shared prefix longer than the prompts that are supposed to contain it, or a file source
    with no file.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: PromptSource = "synthetic"
    count: int = Field(ge=1)
    input_tokens: tuple[int, int] = (128, 128)
    output_tokens: tuple[int, int] = (64, 64)
    shared_prefix_tokens: int = Field(default=0, ge=0)
    path: Path | None = None
    seed: int = 0
    tenants: tuple[str, ...] = ("bench",)

    @model_validator(mode="after")
    def _coherent(self) -> PromptSpec:
        for name, (low, high) in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if low < 1 or high < low:
                raise ValueError(f"{name}={low, high} must satisfy 1 <= min <= max")
        if self.shared_prefix_tokens and self.shared_prefix_tokens >= self.input_tokens[0]:
            raise ValueError(
                f"shared_prefix_tokens ({self.shared_prefix_tokens}) must be smaller "
                f"than the shortest prompt ({self.input_tokens[0]} tokens)"
            )
        if self.source == "synthetic" and self.path is not None:
            raise ValueError("the synthetic source takes no path")
        if self.source != "synthetic" and self.path is None:
            raise ValueError(f"the {self.source} source needs a path")
        if not self.tenants:
            raise ValueError("at least one tenant is required")
        return self


def _sampling_pool(tokenizer: TokenizerLike, explicit: Sequence[int] | None) -> list[int]:
    """Token ids that are safe to sample: ordinary vocabulary, no special tokens.

    Special ids are excluded because a randomly placed ``<|endoftext|>`` would truncate the
    prompt inside the model and make the advertised prompt length a lie. When the tokenizer
    does not expose a vocabulary size, the caller must pass ``explicit`` ids.
    """
    if explicit is not None:
        pool = [int(value) for value in explicit]
        if not pool:
            raise PromptError("the explicit token pool is empty")
        return pool
    size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(size, int) or size <= 0:
        raise PromptError(
            "tokenizer exposes no usable vocab_size; pass vocab_ids to the builder instead"
        )
    special = getattr(tokenizer, "all_special_ids", None)
    excluded = set(special) if isinstance(special, list | tuple | set) else set()
    # The first few hundred ids of a byte-level BPE vocabulary are single bytes, many of
    # them control characters that do not survive a decode/encode round trip; starting
    # above them makes the fixpoint in _stabilise converge in one round far more often.
    start = 256 if size > 512 else 0
    pool = [value for value in range(start, size) if value not in excluded]
    if not pool:
        raise PromptError(f"tokenizer vocabulary of {size} ids left nothing to sample")
    return pool


def _stabilise(
    tokenizer: TokenizerLike,
    ids: list[int],
    length: int,
    rng: random.Random,
    pool: Sequence[int],
) -> tuple[list[int], str, bool]:
    """Drive ``ids`` towards ``encode(decode(ids)) == ids`` while keeping ``len == length``.

    Returns the ids (always exactly ``length`` of them), their decoding, and whether the
    fixpoint was reached. The loop is the only honest way to get an exact token count out
    of a sub-word tokenizer: decoding ids to text and re-encoding it is not the identity,
    so the sequence is re-encoded, trimmed or topped up to the target length, and tried
    again. Failure to converge is reported, never hidden: the ids are still exactly the
    requested length, but a backend given the *text* form would see a different count.
    """
    current = list(ids)
    text = ""
    for _ in range(MAX_STABILISE_ROUNDS):
        text = tokenizer.decode(current, skip_special_tokens=True)
        reencoded = list(tokenizer.encode(text, add_special_tokens=False))
        if reencoded == current:
            return current, text, True
        if len(reencoded) > length:
            current = reencoded[:length]
        elif len(reencoded) < length:
            current = reencoded + [rng.choice(pool) for _ in range(length - len(reencoded))]
        else:
            current = reencoded
    text = tokenizer.decode(current, skip_special_tokens=True)
    exact = list(tokenizer.encode(text, add_special_tokens=False)) == current
    if not exact:
        logger.debug(
            "prompt of %d tokens did not reach a re-tokenisation fixpoint in %d rounds",
            length,
            MAX_STABILISE_ROUNDS,
        )
    return current, text, exact


class SyntheticPromptBuilder:
    """Builds seeded prompts of exact token length, optionally sharing a common prefix.

    One builder owns one RNG stream, so a builder constructed with the same seed and asked
    for the same batch produces the same prompts. The shared prefix is generated once on
    first use and reused verbatim by every prompt that asks for one.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        *,
        seed: int = 0,
        vocab_ids: Sequence[int] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._seed = seed
        self._rng = random.Random(seed)
        self._pool = _sampling_pool(tokenizer, vocab_ids)
        self._prefix_cache: dict[int, tuple[list[int], str]] = {}

    @property
    def seed(self) -> int:
        """The seed this builder's stream was started from."""
        return self._seed

    def reset(self) -> None:
        """Restart the RNG stream, so the next batch repeats the previous one.

        The memoised shared prefixes are dropped too: they were drawn from the stream, so
        keeping them would let the second batch skip those draws and diverge from the
        first -- exactly the non-determinism the reset exists to avoid.
        """
        self._rng = random.Random(self._seed)
        self._prefix_cache.clear()

    def shared_prefix(self, num_tokens: int) -> tuple[list[int], str]:
        """The shared prefix of ``num_tokens`` ids, generated once and memoised.

        Memoised per length rather than regenerated because the whole point of the
        prefix-cache experiment is that every request opens with the *identical* tokens.
        """
        if num_tokens <= 0:
            return [], ""
        cached = self._prefix_cache.get(num_tokens)
        if cached is None:
            raw = [self._rng.choice(self._pool) for _ in range(num_tokens)]
            ids, text, exact = _stabilise(self._tokenizer, raw, num_tokens, self._rng, self._pool)
            if not exact:
                logger.debug("shared prefix of %d tokens is not re-tokenisation stable", num_tokens)
            cached = (ids, text)
            self._prefix_cache[num_tokens] = cached
        return list(cached[0]), cached[1]

    def build_one(
        self,
        prompt_id: str,
        *,
        prompt_tokens: int,
        max_tokens: int,
        tenant: str = "bench",
        shared_prefix_tokens: int = 0,
    ) -> BenchPrompt:
        """One prompt of exactly ``prompt_tokens`` ids, opening with the shared prefix."""
        if prompt_tokens < 1:
            raise PromptError(f"prompt_tokens must be >= 1, got {prompt_tokens}")
        if max_tokens < 1:
            raise PromptError(f"max_tokens must be >= 1, got {max_tokens}")
        if shared_prefix_tokens >= prompt_tokens:
            raise PromptError(
                f"shared_prefix_tokens ({shared_prefix_tokens}) must be smaller than "
                f"prompt_tokens ({prompt_tokens})"
            )
        prefix_ids, _ = self.shared_prefix(shared_prefix_tokens)
        suffix_len = prompt_tokens - len(prefix_ids)
        raw = [self._rng.choice(self._pool) for _ in range(suffix_len)]
        suffix_ids, _, _ = _stabilise(self._tokenizer, raw, suffix_len, self._rng, self._pool)
        token_ids = [*prefix_ids, *suffix_ids]
        # Decode the whole sequence rather than concatenating the prefix's and the suffix's
        # own decodings: a sub-word tokenizer's rendering of a token can depend on what
        # precedes it, so gluing two decoded strings together can produce text that does
        # not correspond to these ids at all.
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        exact = list(self._tokenizer.encode(text, add_special_tokens=False)) == token_ids
        return BenchPrompt(
            prompt_id=prompt_id,
            token_ids=token_ids,
            text=text,
            max_tokens=max_tokens,
            tenant=tenant,
            shared_prefix_tokens=len(prefix_ids),
            source="synthetic",
            round_trip_exact=exact,
        )

    def build_batch(
        self,
        count: int,
        *,
        input_tokens: tuple[int, int],
        output_tokens: tuple[int, int],
        shared_prefix_tokens: int = 0,
        tenants: Sequence[str] = ("bench",),
        prefix: str = "synthetic",
    ) -> list[BenchPrompt]:
        """``count`` prompts with lengths drawn uniformly from the two ranges.

        Tenants are assigned round-robin rather than sampled, so a two-tenant run is
        exactly balanced and a per-tenant latency comparison is not confounded by one
        tenant having drawn more requests than the other.
        """
        if count < 1:
            raise PromptError(f"count must be >= 1, got {count}")
        if not tenants:
            raise PromptError("at least one tenant is required")
        low_in, high_in = input_tokens
        low_out, high_out = output_tokens
        if low_in < 1 or high_in < low_in:
            raise PromptError(f"input_tokens={input_tokens} must satisfy 1 <= min <= max")
        if low_out < 1 or high_out < low_out:
            raise PromptError(f"output_tokens={output_tokens} must satisfy 1 <= min <= max")
        prompts: list[BenchPrompt] = []
        for index in range(count):
            prompts.append(
                self.build_one(
                    f"{prefix}-{index:05d}",
                    prompt_tokens=self._rng.randint(low_in, high_in),
                    max_tokens=self._rng.randint(low_out, high_out),
                    tenant=tenants[index % len(tenants)],
                    shared_prefix_tokens=shared_prefix_tokens,
                )
            )
        return prompts


def _count_tokens(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Token ids of ``text`` without special tokens, the form a prompt is sent as."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def load_sharegpt_prompts(
    path: Path | str,
    tokenizer: TokenizerLike,
    *,
    count: int,
    input_tokens: tuple[int, int] = (1, 4096),
    output_tokens: tuple[int, int] = (1, 1024),
    seed: int = 0,
    tenants: Sequence[str] = ("bench",),
) -> list[BenchPrompt]:
    """Load real conversation openings from a local ShareGPT-format JSON sample.

    The dataset is not bundled (it is large and its licensing is not ours to redistribute);
    point this at a downloaded copy. The expected shape is the usual one: a JSON array of
    objects with a ``conversations`` list of ``{"from": "human"|"gpt", "value": str}``
    turns. The first human turn becomes the prompt and the length of the reply that
    followed it becomes ``max_tokens``, which is what makes ShareGPT worth the trouble --
    its joint distribution of input and output lengths is far more skewed than anything
    uniform sampling produces, and tail latency lives in that skew.

    Conversations whose lengths fall outside the requested ranges are skipped; the
    selection is shuffled with ``seed`` so a truncated sample is not the head of the file.
    """
    entries = _read_json(Path(path))
    if not isinstance(entries, list):
        raise PromptError(f"{path} must contain a JSON array of conversations")
    rng = random.Random(seed)
    order = list(range(len(entries)))
    rng.shuffle(order)
    low_in, high_in = input_tokens
    low_out, high_out = output_tokens
    prompts: list[BenchPrompt] = []
    for position in order:
        entry = entries[position]
        turns = entry.get("conversations") if isinstance(entry, dict) else None
        if not isinstance(turns, list) or len(turns) < 2:
            continue
        human = _first_turn(turns, "human")
        reply = _first_turn(turns, "gpt")
        if human is None or reply is None:
            continue
        prompt_ids = _count_tokens(tokenizer, human)
        reply_len = len(_count_tokens(tokenizer, reply))
        if not low_in <= len(prompt_ids) <= high_in:
            continue
        if not low_out <= reply_len <= high_out:
            continue
        index = len(prompts)
        prompts.append(
            BenchPrompt(
                prompt_id=f"sharegpt-{index:05d}",
                token_ids=prompt_ids,
                text=human,
                max_tokens=reply_len,
                tenant=tenants[index % len(tenants)],
                source="sharegpt",
                round_trip_exact=True,
            )
        )
        if len(prompts) == count:
            break
    if len(prompts) < count:
        raise PromptError(
            f"{path} yielded {len(prompts)} usable conversations, needed {count}; widen "
            "input_tokens/output_tokens or use a larger sample"
        )
    return prompts


def _first_turn(turns: list[Any], speaker: str) -> str | None:
    """The first turn spoken by ``speaker``, or ``None``."""
    for turn in turns:
        if isinstance(turn, dict) and turn.get("from") == speaker:
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                return value
    return None


def load_file_prompts(
    path: Path | str,
    tokenizer: TokenizerLike,
    *,
    count: int | None = None,
    default_max_tokens: int = 128,
    tenants: Sequence[str] = ("bench",),
) -> list[BenchPrompt]:
    """Load prompts written by hand or captured from production traffic.

    Three layouts are accepted, chosen by suffix: ``.txt`` (one prompt per non-empty line),
    ``.jsonl`` (one JSON object per line) and ``.json`` (an array of strings or objects).
    An object may set ``prompt`` (required), ``max_tokens``, ``tenant`` and ``prompt_id``;
    anything else in it is ignored, so a file exported from a log with extra columns works
    without editing.
    """
    source = Path(path)
    raw = _read_prompt_file(source)
    prompts: list[BenchPrompt] = []
    for index, item in enumerate(raw):
        if count is not None and len(prompts) == count:
            break
        if isinstance(item, str):
            text, max_tokens, tenant, prompt_id = item, default_max_tokens, None, None
        elif isinstance(item, dict):
            value = item.get("prompt")
            if not isinstance(value, str) or not value:
                raise PromptError(f"{source}: entry {index} has no non-empty 'prompt'")
            text = value
            max_tokens = int(item.get("max_tokens", default_max_tokens))
            tenant = item.get("tenant")
            prompt_id = item.get("prompt_id")
        else:
            raise PromptError(f"{source}: entry {index} is neither a string nor an object")
        if max_tokens < 1:
            raise PromptError(f"{source}: entry {index} has max_tokens < 1")
        position = len(prompts)
        prompts.append(
            BenchPrompt(
                prompt_id=str(prompt_id or f"file-{position:05d}"),
                token_ids=_count_tokens(tokenizer, text),
                text=text,
                max_tokens=max_tokens,
                tenant=str(tenant or tenants[position % len(tenants)]),
                source="file",
                round_trip_exact=True,
            )
        )
    if not prompts:
        raise PromptError(f"{source} contained no prompts")
    if count is not None and len(prompts) < count:
        raise PromptError(f"{source} has {len(prompts)} prompts, needed {count}")
    return prompts


def _read_prompt_file(path: Path) -> list[Any]:
    """Normalise the three accepted file layouts into a list of entries."""
    if not path.is_file():
        raise PromptError(f"prompt file {path} does not exist")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [line for line in (raw.strip() for raw in text.splitlines()) if line]
    if suffix == ".jsonl":
        entries: list[Any] = []
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise PromptError(f"{path}:{number} is not valid JSON: {exc}") from exc
        return entries
    if suffix == ".json":
        data = _read_json(path)
        if not isinstance(data, list):
            raise PromptError(f"{path} must contain a JSON array")
        return data
    raise PromptError(f"unsupported prompt file suffix {path.suffix!r}; use .txt, .jsonl or .json")


def _read_json(path: Path) -> Any:
    """Parse a JSON file with an error that names the file."""
    if not path.is_file():
        raise PromptError(f"prompt file {path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromptError(f"{path} is not valid JSON: {exc}") from exc


def build_prompts(spec: PromptSpec, tokenizer: TokenizerLike) -> list[BenchPrompt]:
    """Build the workload described by ``spec`` from whichever source it names."""
    if spec.source == "synthetic":
        builder = SyntheticPromptBuilder(tokenizer, seed=spec.seed)
        return builder.build_batch(
            spec.count,
            input_tokens=spec.input_tokens,
            output_tokens=spec.output_tokens,
            shared_prefix_tokens=spec.shared_prefix_tokens,
            tenants=spec.tenants,
        )
    if spec.path is None:  # pragma: no cover - PromptSpec validation guarantees a path
        raise PromptError(f"the {spec.source} source needs a path")
    if spec.source == "sharegpt":
        return load_sharegpt_prompts(
            spec.path,
            tokenizer,
            count=spec.count,
            input_tokens=spec.input_tokens,
            output_tokens=spec.output_tokens,
            seed=spec.seed,
            tenants=spec.tenants,
        )
    return load_file_prompts(
        spec.path,
        tokenizer,
        count=spec.count,
        default_max_tokens=spec.output_tokens[1],
        tenants=spec.tenants,
    )
