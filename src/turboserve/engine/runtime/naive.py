"""Baseline engines: what serving looks like without continuous batching.

Every claim the reference engine makes needs something to be compared against, and the
comparison is only fair if both sides run the same model on the same hardware with the same
prompts and the same sampling parameters. These two classes are that other side. They are
built on ``transformers.generate`` with its own KV cache -- the code an engineer writes when
asked to "serve this model" before anyone mentions batching -- and they expose the same
``add_request``/``step`` surface as :class:`~turboserve.engine.runtime.engine.LLMEngine`, so
the benchmark driver cannot accidentally treat them differently.

:class:`NaiveHFEngine`
    One request at a time, to completion, in arrival order. The GPU is idle between the
    end of one request's decode and the start of the next one's prefill, and every decode
    step reads the whole weight matrix to produce a single token.

:class:`StaticBatchHFEngine`
    The first improvement everyone reaches for: collect ``batch_size`` requests, pad them to
    the longest prompt, generate for all of them at once, and return when the *longest* one
    finishes. Throughput improves; the cost is that a short request waits for a long one it
    has nothing to do with, and that padding wastes compute proportional to the spread of
    prompt lengths.

Neither is a straw man: both use the real KV cache, real sampling parameters and the same
stop conditions. What they do not do is admit a request into a batch that has already
started, which is precisely the thing continuous batching adds.

**Interface note.** ``step()`` here returns whole completions rather than single-token
deltas: these engines have no notion of a step, and a fabricated per-token trickle would
misreport time-to-first-token. The first-token timestamp is captured honestly, by a
``StoppingCriteria`` that the generation loop calls after each new token.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from turboserve.engine.core.sequence import DEFAULT_TENANT, Sequence
from turboserve.engine.core.types import (
    NO_LORA,
    EngineConfig,
    FinishReason,
    RequestOutput,
    SamplingParams,
)
from turboserve.engine.runtime.streaming import StreamingDecoder, get_tokenizer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence as SequenceABC

    from turboserve.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["BaselineEngine", "NaiveHFEngine", "StaticBatchHFEngine"]


class _FirstTokenStamp:
    """A ``StoppingCriteria`` that records when the first new token appeared.

    ``transformers.generate`` calls the stopping criteria once per generated token and has
    no other per-token callback that works for a batch, so this is the honest place to take
    the TTFT timestamp: at the first call, the first token exists. It never stops anything.
    """

    def __init__(self) -> None:
        self.t_first: float | None = None

    def __call__(self, input_ids: torch.Tensor, scores: Any = None, **kwargs: Any) -> torch.Tensor:
        if self.t_first is None:
            self.t_first = time.perf_counter()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


@dataclass(slots=True)
class _Pending:
    """A queued request and the clock reading from when it arrived."""

    request_id: str
    token_ids: list[int]
    sampling: SamplingParams
    tenant_id: str
    priority: int
    lora_id: int
    arrival: float = field(default_factory=time.perf_counter)


class BaselineEngine(ABC):
    """Shared plumbing for the ``transformers``-based baselines.

    Holds the model, the tokenizer and the queue, and turns one ``generate`` call into
    :class:`~turboserve.engine.core.types.RequestOutput` objects with the same timing fields
    the reference engine fills in, so the benchmark's metric code is identical for both.
    """

    #: Name recorded in result files, so a table row can be traced back to a class.
    backend_name = "baseline"

    def __init__(
        self,
        config: EngineConfig,
        *,
        model: Any = None,
        tokenizer: Any = None,
        local_files_only: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM

        self.config = config
        self._device = torch.device(config.resolved_device())
        self._dtype = config.resolved_dtype()
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                config.model,
                dtype=self._dtype,
                local_files_only=local_files_only,
            )
            model.to(self._device)
        self.model = model
        self.model.eval()
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else get_tokenizer(config.tokenizer or config.model, local_files_only=local_files_only)
        )
        self._pad_token_id = self._resolve_pad_token_id()
        self._eos_token_ids = self._resolve_eos_token_ids()
        self._queue: list[_Pending] = []
        self._finished: set[str] = set()
        self._aborted: set[str] = set()
        self._num_prompt_tokens = 0
        self._num_generated_tokens = 0
        self._num_batches = 0
        self._closed = False

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> BaselineEngine:
        """Build a baseline from the process-wide ``TURBOSERVE_*`` settings."""
        return cls(EngineConfig.from_settings(settings), **kwargs)

    def _resolve_pad_token_id(self) -> int:
        """Pick a padding id: EOS first, then the tokenizer's own pad id, then zero.

        EOS is preferred because ``generate`` pads a row that finished early, and a row
        padded with EOS is trimmed correctly by the stop check in :meth:`_build_output`
        while a row padded with a distinct pad id would appear to have generated it.

        The fallback chain also exists because tiny test checkpoints ship
        ``pad_token_id = -1`` in their generation config, which ``generate`` will happily
        use as an index and produce garbage from.
        """
        for candidate in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        ):
            if isinstance(candidate, int) and candidate >= 0:
                return candidate
        return 0

    def _resolve_eos_token_ids(self) -> list[int]:
        """Every id that ends a generation, from the tokenizer and the model config."""
        ids: set[int] = set()
        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(tokenizer_eos, int) and tokenizer_eos >= 0:
            ids.add(tokenizer_eos)
        config_eos = getattr(getattr(self.model, "config", None), "eos_token_id", None)
        if isinstance(config_eos, int) and config_eos >= 0:
            ids.add(config_eos)
        elif isinstance(config_eos, list):
            ids.update(int(i) for i in config_eos if int(i) >= 0)
        return sorted(ids)

    # -- request lifecycle ------------------------------------------------------------------

    def encode(self, prompt: str) -> list[int]:
        """Tokenize a text prompt without adding special tokens."""
        return [int(t) for t in self.tokenizer.encode(prompt, add_special_tokens=False)]

    def add_request(
        self,
        request_id: str,
        prompt: str | SequenceABC[int],
        sampling: SamplingParams | None = None,
        *,
        tenant_id: str = DEFAULT_TENANT,
        priority: int = 0,
        lora_id: int = NO_LORA,
        arrival: float | None = None,
    ) -> None:
        """Queue a request. Signature-compatible with :meth:`LLMEngine.add_request`."""
        if self._closed:
            raise RuntimeError("engine is closed")
        if any(item.request_id == request_id for item in self._queue):
            raise ValueError(f"request {request_id!r} is already in flight")
        token_ids = self.encode(prompt) if isinstance(prompt, str) else [int(t) for t in prompt]
        if not token_ids:
            raise ValueError(f"request {request_id!r} has an empty prompt")
        self._queue.append(
            _Pending(
                request_id=request_id,
                token_ids=token_ids,
                sampling=sampling if sampling is not None else SamplingParams(),
                tenant_id=tenant_id,
                priority=priority,
                lora_id=lora_id,
                arrival=time.perf_counter() if arrival is None else arrival,
            )
        )
        self._num_prompt_tokens += len(token_ids)

    def abort(self, request_id: str, *, now: float | None = None) -> bool:
        """Drop a queued request. A request already being generated runs to completion.

        That limitation is the baseline's, not an implementation shortcut:
        ``transformers.generate`` is a blocking call with no cancellation point, which is one
        of the concrete reasons a serving engine does not use it. ``now`` is accepted for
        signature parity with :meth:`LLMEngine.abort` and ignored: nothing here is timed
        against a clock the caller can inject.
        """
        del now
        before = len(self._queue)
        self._queue = [item for item in self._queue if item.request_id != request_id]
        if len(self._queue) != before:
            self._aborted.add(request_id)
            return True
        return False

    def has_unfinished(self) -> bool:
        """Whether any queued request has still to be generated."""
        return bool(self._queue)

    def __len__(self) -> int:
        return len(self._queue)

    # -- stepping ---------------------------------------------------------------------------

    @abstractmethod
    def _take_batch(self) -> list[_Pending]:
        """Choose the requests the next ``generate`` call covers."""

    def step(self) -> list[RequestOutput]:
        """Generate the next batch to completion and return one output per request."""
        if self._closed:
            raise RuntimeError("engine is closed")
        batch = self._take_batch()
        if not batch:
            return []
        self._num_batches += 1
        return self._generate(batch)

    @torch.inference_mode()
    def _generate(self, batch: list[_Pending]) -> list[RequestOutput]:
        """Run one padded ``generate`` call and split the result back into requests."""
        from transformers import StoppingCriteriaList

        max_new = max(item.sampling.max_tokens for item in batch)
        widest = max(len(item.token_ids) for item in batch)
        input_ids = torch.full(
            (len(batch), widest), self._pad_token_id, dtype=torch.long, device=self._device
        )
        attention_mask = torch.zeros((len(batch), widest), dtype=torch.long, device=self._device)
        for row, item in enumerate(batch):
            # Left padding: generation continues from the last column, so every sequence
            # must end there regardless of its prompt length.
            start = widest - len(item.token_ids)
            input_ids[row, start:] = torch.tensor(
                item.token_ids, dtype=torch.long, device=self._device
            )
            attention_mask[row, start:] = 1

        stamp = _FirstTokenStamp()
        if batch[0].sampling.seed is not None:
            # ``generate`` draws from the global RNG; a uniform seed across the batch is the
            # only reproducibility this interface can offer, and the benchmark uses it.
            torch.manual_seed(batch[0].sampling.seed)
        eos_ids = None if batch[0].sampling.ignore_eos else (self._eos_token_ids or None)
        t_start = time.perf_counter()
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            pad_token_id=self._pad_token_id,
            eos_token_id=eos_ids,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([stamp]),
            **self._sampling_kwargs(batch),
        )
        t_finish = time.perf_counter()
        t_first = stamp.t_first if stamp.t_first is not None else t_finish
        return [
            self._build_output(item, generated[row, widest:].tolist(), t_start, t_first, t_finish)
            for row, item in enumerate(batch)
        ]

    def _sampling_kwargs(self, batch: list[_Pending]) -> dict[str, Any]:
        """Translate sampling parameters for ``generate``.

        ``generate`` applies one set of parameters to the whole batch -- it has no per-row
        sampling -- so a heterogeneous batch is rejected rather than silently served with the
        first request's settings. This is a real limitation of static batching and the
        benchmark drives these engines with uniform parameters for exactly that reason.
        """
        first = batch[0].sampling
        for item in batch[1:]:
            if item.sampling != first:
                raise ValueError(
                    "the HF baselines apply one sampling configuration to a whole batch; "
                    f"{item.request_id!r} differs from {batch[0].request_id!r}"
                )
        if first.is_greedy:
            return {"do_sample": False}
        kwargs: dict[str, Any] = {
            "do_sample": True,
            "temperature": first.temperature,
            "top_p": first.top_p,
        }
        if first.top_k > 0:
            kwargs["top_k"] = first.top_k
        if first.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = first.repetition_penalty
        return kwargs

    def _build_output(
        self,
        item: _Pending,
        new_token_ids: list[int],
        t_start: float,
        t_first: float,
        t_finish: float,
    ) -> RequestOutput:
        """Trim a generated row at its stop condition and package it as a RequestOutput."""
        seq = Sequence(
            seq_id=0,
            request_id=item.request_id,
            prompt_token_ids=list(item.token_ids),
            sampling=item.sampling,
            tenant_id=item.tenant_id,
            priority=item.priority,
            lora_id=item.lora_id,
        )
        seq.timing.t_arrival = item.arrival
        seq.timing.t_first_scheduled = t_start
        decoder = StreamingDecoder.create(self.tokenizer, stop=item.sampling.stop)
        reason: FinishReason | None = None
        for token_id in new_token_ids:
            seq.append_token(token_id, now=t_first if seq.num_output_tokens == 0 else t_finish)
            delta = decoder.feed((token_id,))
            if token_id in self._eos_token_ids and not item.sampling.ignore_eos:
                # Kept in the output, exactly as the reference engine and ``generate`` do:
                # a comparison between the two would otherwise differ by one token.
                reason = FinishReason.STOP
                break
            stop = seq.check_stop(token_id, eos_token_id=None)
            if delta.stopped:
                reason = FinishReason.STOP
                break
            if stop is not None:
                reason = stop
                break
        else:
            reason = (
                FinishReason.LENGTH
                if seq.num_output_tokens >= item.sampling.max_tokens
                else FinishReason.STOP
            )
        decoder.finish()
        seq.finish(reason or FinishReason.STOP, now=t_finish)
        self._finished.add(item.request_id)
        self._num_generated_tokens += seq.num_output_tokens
        return seq.make_output(new_token_ids=seq.output_token_ids, text_delta=decoder.text)

    # -- observability ------------------------------------------------------------------------

    def stats(self) -> dict[str, int | float | str]:
        """Counters in the same shape as :meth:`LLMEngine.stats`, minus the KV-pool fields."""
        return {
            "backend": self.backend_name,
            "model": self.config.model,
            "device": str(self._device),
            "dtype": str(self._dtype).removeprefix("torch."),
            "num_waiting": len(self._queue),
            "num_running": 0,
            "num_preempted": 0,
            "num_unfinished": len(self._queue),
            "num_finished": len(self._finished),
            "num_batches": self._num_batches,
            "num_prompt_tokens": self._num_prompt_tokens,
            "num_generated_tokens": self._num_generated_tokens,
        }

    def close(self) -> None:
        """Drop the queue and release device memory. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._queue.clear()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self) -> BaselineEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.config.model!r}, device={self._device}, "
            f"queued={len(self._queue)})"
        )


class NaiveHFEngine(BaselineEngine):
    """One request at a time, run to completion before the next one starts.

    This is the lower bound the whole engine exists to beat, and the shape of the gap is
    structural rather than incidental: a decode step reads every weight of the model to
    produce one token, so serving one sequence per forward pass leaves the arithmetic units
    idle waiting on memory, and the queue behind the current request is not being served at
    all.
    """

    backend_name = "naive_hf"

    def _take_batch(self) -> list[_Pending]:
        return [self._queue.pop(0)] if self._queue else []


class StaticBatchHFEngine(BaselineEngine):
    """Fixed-size batches: wait for ``batch_size`` requests, then generate for all of them.

    Two costs the reference engine does not pay. First, *head-of-line padding*: the batch is
    padded to the longest prompt and generates for the longest requested output, so a short
    request is billed the long one's latency. Second, *no mid-flight admission*: a request
    that arrives one microsecond after the batch starts waits for the whole batch, which is
    what makes the tail latency of static batching so much worse than its average.

    ``flush_partial`` decides what happens when the queue drains below ``batch_size``. The
    benchmark keeps it ``True`` so a run terminates; a server would set it ``False`` and pair
    it with a timeout.
    """

    backend_name = "static_batch"

    def __init__(
        self,
        config: EngineConfig,
        *,
        batch_size: int | None = None,
        flush_partial: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.batch_size = batch_size if batch_size is not None else config.scheduler.max_num_seqs
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        self.flush_partial = flush_partial

    def _take_batch(self) -> list[_Pending]:
        if not self._queue:
            return []
        if len(self._queue) < self.batch_size and not self.flush_partial:
            return []
        take = min(self.batch_size, len(self._queue))
        batch, self._queue = self._queue[:take], self._queue[take:]
        return batch
