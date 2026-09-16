"""The synchronous engine: the step loop that ties scheduler, model runner and tokenizer.

:class:`LLMEngine` is a small object with a large job. It owns nothing that computes --
the scheduler decides, the runner executes, the sampler draws -- and everything that is
*stateful about requests*: the tokenizer, the per-request streaming decoders, the stop
conditions that need text rather than token ids, and the token accounting a gateway has to
report. One call to :meth:`LLMEngine.step` advances every in-flight request by at most one
token and returns the deltas.

The loop is deliberately synchronous and single-threaded. Continuous batching already
extracts the parallelism that matters (many requests share one forward pass), and a step
that mutates the KV pool cannot safely overlap with another step on the same pool. The
asynchronous surface lives in :mod:`turboserve.engine.runtime.async_engine`, which drives
this class from a worker thread; making the engine itself concurrent would buy nothing and
cost the ability to reason about the block allocator.

Extension points, both used by the modules layered on top of this one and neither requiring
a change to this file:

``decode_step_hook``
    Replaces phases 2-4 of a step. :mod:`turboserve.engine.spec` installs its draft/verify
    loop here; returning ``None`` falls back to the ordinary path, which is how a
    speculative engine handles a prefill step with no decode work.
``lora_ctx_builder``
    Supplies the per-token adapter slots for a step. Forwarded to the
    :class:`~turboserve.engine.runtime.worker.ModelRunner`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

import torch
import typer

from turboserve.engine.core.block_manager import BlockManager
from turboserve.engine.core.scheduler import Scheduler
from turboserve.engine.core.sequence import DEFAULT_TENANT
from turboserve.engine.core.types import (
    NO_LORA,
    EngineConfig,
    FinishReason,
    RequestOutput,
    SamplingParams,
    SchedulerConfig,
)
from turboserve.engine.model.model import CausalLM
from turboserve.engine.runtime.memory import KVCacheSizing, build_kv_cache, size_kv_cache
from turboserve.engine.runtime.streaming import StreamingDecoder, get_tokenizer
from turboserve.engine.runtime.worker import ModelRunner, StepOutput

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable
    from collections.abc import Sequence as SequenceABC

    from turboserve.config import Settings
    from turboserve.engine.core.kv_cache import KVCache
    from turboserve.engine.core.scheduler import SchedulerOutput
    from turboserve.engine.core.sequence import Sequence
    from turboserve.engine.core.types import LoRAContext
    from turboserve.engine.runtime.streaming import DetokenizerLike

logger = logging.getLogger(__name__)

__all__ = ["DecodeStepHook", "EngineStats", "LLMEngine", "engine_app"]

type DecodeStepHook = Callable[["LLMEngine", "SchedulerOutput"], StepOutput | None]
"""Replacement for the forward/sample phases of a step.

Called with the engine and the step the scheduler produced. Returning ``None`` means "I did
not handle this step", and the engine runs its own path -- which is what lets a speculative
engine opt out of steps that contain no decoding sequences.
"""


@dataclass(frozen=True, slots=True)
class EngineStats:
    """A snapshot of engine health, in the shape the gateway exports to Prometheus."""

    num_running: int
    num_waiting: int
    num_preempted: int
    num_unfinished: int
    num_finished: int
    num_preemptions: int
    num_steps: int
    kv_utilization: float
    """Fraction of KV blocks currently held by running sequences."""

    prefix_hit_rate: float
    """Fraction of prefix-cache block lookups that hit, over the engine's lifetime."""

    num_prompt_tokens: int
    num_generated_tokens: int
    num_cached_prompt_tokens: int

    def to_dict(self) -> dict[str, int | float]:
        """Plain-data view; the engine's :meth:`LLMEngine.stats` merges it with raw counters."""
        return {
            "num_running": self.num_running,
            "num_waiting": self.num_waiting,
            "num_preempted": self.num_preempted,
            "num_unfinished": self.num_unfinished,
            "num_finished": self.num_finished,
            "num_preemptions": self.num_preemptions,
            "num_steps": self.num_steps,
            "kv_utilization": self.kv_utilization,
            "prefix_hit_rate": self.prefix_hit_rate,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_generated_tokens": self.num_generated_tokens,
            "num_cached_prompt_tokens": self.num_cached_prompt_tokens,
        }


class LLMEngine:
    """Continuous-batching engine over a paged KV cache.

    Build it from an :class:`~turboserve.engine.core.types.EngineConfig` and drive it with
    :meth:`add_request` and :meth:`step` until :meth:`has_unfinished` is false.
    """

    def __init__(
        self,
        config: EngineConfig,
        *,
        model: CausalLM | None = None,
        tokenizer: DetokenizerLike | None = None,
        kv_cache: KVCache | None = None,
        scheduler: Scheduler | None = None,
        runner: ModelRunner | None = None,
        local_files_only: bool = False,
        decode_step_hook: DecodeStepHook | None = None,
        lora_ctx_builder: Callable[[SchedulerOutput, torch.device], LoRAContext | None]
        | None = None,
    ) -> None:
        """Construct an engine, loading whatever was not supplied.

        Every collaborator can be injected. That is not generality for its own sake: a test
        that wants a deterministic 4-block pool, or the speculative engine that wants two
        models sharing one tokenizer, must be able to build the pieces itself, and a
        constructor that always loads from disk would force both to reach into private
        attributes.
        """
        self.config = config
        self._device = torch.device(config.resolved_device())
        self._dtype = config.resolved_dtype()
        self.decode_step_hook = decode_step_hook

        self.model = (
            model
            if model is not None
            else CausalLM.from_pretrained(
                config.model,
                dtype=self._dtype,
                device=self._device,
                local_files_only=local_files_only,
            )
        )
        self._device = self.model.device
        self._dtype = self.model.dtype
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else get_tokenizer(config.tokenizer or config.model, local_files_only=local_files_only)
        )
        self._eos_token_ids = self._resolve_eos_token_ids()

        self.sizing: KVCacheSizing | None = None
        if kv_cache is None:
            self.sizing = size_kv_cache(
                self.model.config,
                config.scheduler,
                device=self._device,
                dtype=self._dtype,
                gpu_memory_utilization=config.gpu_memory_utilization,
                num_blocks=config.num_blocks,
                max_model_len=config.max_model_len,
            )
            kv_cache = build_kv_cache(
                self.model.config, self.sizing, device=self._device, dtype=self._dtype
            )
        self.kv_cache = kv_cache

        if scheduler is None:
            block_manager = BlockManager.create(
                self.kv_cache.num_blocks,
                config.block_size,
                enable_prefix_caching=config.scheduler.enable_prefix_caching,
            )
            scheduler = Scheduler(config.scheduler, block_manager)
        self.scheduler = scheduler
        self.runner = (
            runner
            if runner is not None
            else ModelRunner(self.model, self.kv_cache, lora_ctx_builder=lora_ctx_builder)
        )
        if runner is not None and lora_ctx_builder is not None:
            self.runner.lora_ctx_builder = lora_ctx_builder

        self._decoders: dict[str, StreamingDecoder] = {}
        self._num_generated_tokens = 0
        self._num_prompt_tokens = 0
        self._closed = False
        logger.info(
            "engine ready: %s on %s (%s), %d KV blocks of %d tokens",
            self.model.config.architecture,
            self._device,
            self._dtype,
            self.kv_cache.num_blocks,
            self.kv_cache.block_size,
        )

    # -- construction helpers -----------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> LLMEngine:
        """Build an engine from the process-wide ``TURBOSERVE_*`` settings."""
        return cls(EngineConfig.from_settings(settings), **kwargs)

    def _resolve_eos_token_ids(self) -> frozenset[int]:
        """Every id that ends a generation: the model config's, plus the tokenizer's.

        Both are consulted because they disagree in practice -- a chat checkpoint's
        ``generation_config`` often lists ``<|im_end|>`` alongside ``<|endoftext|>`` while
        the tokenizer reports only the latter -- and stopping on too few of them is the
        failure mode that produces endless self-conversations.
        """
        ids: set[int] = {int(i) for i in self.model.config.eos_token_ids}
        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(tokenizer_eos, int):
            ids.add(tokenizer_eos)
        elif isinstance(tokenizer_eos, list):
            ids.update(int(i) for i in tokenizer_eos)
        return frozenset(ids)

    @property
    def eos_token_ids(self) -> frozenset[int]:
        """Token ids that terminate a generation unless ``ignore_eos`` is set."""
        return self._eos_token_ids

    @property
    def device(self) -> torch.device:
        """Device the model and KV pool live on."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Serving dtype of the model weights and the KV cache."""
        return self._dtype

    @property
    def lora_ctx_builder(
        self,
    ) -> Callable[[SchedulerOutput, torch.device], LoRAContext | None] | None:
        """Hook supplying per-token adapter slots; see :mod:`turboserve.engine.runtime.worker`."""
        return self.runner.lora_ctx_builder

    @lora_ctx_builder.setter
    def lora_ctx_builder(
        self,
        builder: Callable[[SchedulerOutput, torch.device], LoRAContext | None] | None,
    ) -> None:
        self.runner.lora_ctx_builder = builder

    # -- request lifecycle ----------------------------------------------------------------

    def encode(self, prompt: str) -> list[int]:
        """Tokenize a prompt without adding a second BOS.

        ``add_special_tokens=False`` because a chat prompt arriving from the gateway has
        already been through the chat template, which inserts whatever special tokens the
        model expects. Adding another here shifts every position by one and quietly changes
        the model's behaviour.
        """
        encode = getattr(self.tokenizer, "encode", None)
        if encode is None:
            raise TypeError("the configured tokenizer cannot encode text prompts")
        return [int(token) for token in encode(prompt, add_special_tokens=False)]

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
    ) -> Sequence:
        """Queue a request. Returns the sequence the scheduler will drive.

        The prompt may be text or token ids: the gateway sends ids when it has already
        applied a chat template, and text otherwise. Raises ``ValueError`` for a duplicate
        request id, an empty prompt, or a prompt that cannot fit the KV pool at all.
        """
        if self._closed:
            raise RuntimeError("engine is closed")
        token_ids = self.encode(prompt) if isinstance(prompt, str) else [int(t) for t in prompt]
        if not token_ids:
            raise ValueError(f"request {request_id!r} has an empty prompt")
        seq = self.scheduler.add_request(
            request_id,
            token_ids,
            sampling,
            tenant_id=tenant_id,
            priority=priority,
            lora_id=lora_id,
            arrival=arrival,
        )
        self._decoders[request_id] = StreamingDecoder.create(self.tokenizer, stop=seq.sampling.stop)
        self._num_prompt_tokens += len(token_ids)
        logger.debug(
            "queued %s (tenant=%s, %d prompt tokens, lora=%d)",
            request_id,
            tenant_id,
            len(token_ids),
            lora_id,
        )
        return seq

    def abort(self, request_id: str, *, now: float | None = None) -> bool:
        """Cancel a request. Returns whether it was still in flight."""
        aborted = self.scheduler.abort(request_id, now=now)
        self._decoders.pop(request_id, None)
        return aborted

    def has_unfinished(self) -> bool:
        """Whether any request still owes the caller tokens."""
        return self.scheduler.has_unfinished()

    def __len__(self) -> int:
        return self.scheduler.num_unfinished

    # -- the step -----------------------------------------------------------------------

    def step(self, *, now: float | None = None) -> list[RequestOutput]:
        """Advance every scheduled request by one token and return the deltas.

        An empty list means one of three things, all normal: nothing is queued, the step
        consisted only of prefill chunks that have not reached the end of their prompts, or
        every scheduled sequence was preempted. The caller should look at
        :meth:`has_unfinished` rather than at the length of this list to decide whether to
        keep stepping.
        """
        if self._closed:
            raise RuntimeError("engine is closed")
        timestamp = time.perf_counter() if now is None else now
        scheduled = self.scheduler.schedule(now=timestamp)
        if scheduled.is_empty:
            return []
        step_out = self._execute(scheduled)
        return self._process(step_out, now=timestamp)

    def _execute(self, scheduled: SchedulerOutput) -> StepOutput:
        """Run the step, giving ``decode_step_hook`` the first refusal."""
        if self.decode_step_hook is not None:
            replaced = self.decode_step_hook(self, scheduled)
            if replaced is not None:
                return replaced
        return self.runner.execute(scheduled)

    def _process(self, step_out: StepOutput, *, now: float) -> list[RequestOutput]:
        """Feed sampled tokens back: stop checks, detokenization, per-request deltas."""
        outputs: list[RequestOutput] = []
        for item, token_id in step_out.pairs():
            seq = item.seq
            if seq.is_finished:
                continue  # aborted between scheduling and now
            reason = self.scheduler.append_token(seq, token_id, eos_token_id=None, now=now)
            if reason is None and token_id in self._eos_token_ids and not seq.sampling.ignore_eos:
                # The core's stop check only knows the ids in SamplingParams.stop_token_ids;
                # the model's own EOS set lives here because it comes from the checkpoint.
                self.scheduler.finish(seq, FinishReason.STOP, now=now, stop_token_id=token_id)
                reason = FinishReason.STOP
            outputs.append(self._emit(seq, token_id, reason, now=now))
        self._num_generated_tokens += len(outputs)
        return outputs

    def _emit(
        self, seq: Sequence, token_id: int, reason: FinishReason | None, *, now: float
    ) -> RequestOutput:
        """Detokenize one sampled token and build the client-facing delta.

        Stop *strings* are resolved here and nowhere else: they are defined on text, and
        text only exists once the incremental detokenizer has decided the bytes are
        complete. A stop string therefore finishes the sequence one step after the token
        that completed it in the worst case, which is unavoidable and matches every other
        server's behaviour.
        """
        decoder = self._decoders.get(seq.request_id)
        if decoder is None:
            # Aborted and re-added, or injected by a test: decode without stop strings
            # rather than losing the text entirely.
            decoder = StreamingDecoder.create(self.tokenizer, stop=seq.sampling.stop)
            self._decoders[seq.request_id] = decoder
        delta = decoder.feed((token_id,))
        text = delta.text
        if delta.stopped and reason is None:
            self.scheduler.finish(seq, FinishReason.STOP, now=now)
            reason = FinishReason.STOP
        elif reason is not None and not delta.stopped:
            text += decoder.finish()
        output = seq.make_output(new_token_ids=[token_id], text_delta=text)
        if reason is not None:
            self._decoders.pop(seq.request_id, None)
        return output

    # -- convenience --------------------------------------------------------------------

    def generate(
        self,
        prompts: SequenceABC[str | SequenceABC[int]],
        sampling: SamplingParams | None = None,
        *,
        tenant_id: str = DEFAULT_TENANT,
        request_id_prefix: str = "gen",
    ) -> list[RequestOutput]:
        """Run a batch of prompts to completion and return one aggregate result each.

        A blocking convenience for the CLI smoke command and for tests; serving goes through
        :meth:`step` (or :class:`~turboserve.engine.runtime.async_engine.AsyncLLMEngine`) so
        that tokens reach the client as they are produced.
        """
        order: list[str] = []
        accumulated: dict[str, RequestOutput] = {}
        for index, prompt in enumerate(prompts):
            request_id = f"{request_id_prefix}-{index}"
            order.append(request_id)
            self.add_request(request_id, prompt, sampling, tenant_id=tenant_id)
        while self.has_unfinished():
            for output in self.step():
                current = accumulated.get(output.request_id)
                if current is None:
                    accumulated[output.request_id] = output
                    continue
                current.new_token_ids.extend(output.new_token_ids)
                current.text_delta += output.text_delta
                current.finished = output.finished
                current.finish_reason = output.finish_reason
                current.output_tokens = output.output_tokens
                current.prompt_tokens = output.prompt_tokens
                current.cached_prompt_tokens = output.cached_prompt_tokens
                current.timing = output.timing
        return [accumulated[request_id] for request_id in order if request_id in accumulated]

    # -- observability -------------------------------------------------------------------

    def engine_stats(self) -> EngineStats:
        """Structured health snapshot; :meth:`stats` returns it merged with raw counters."""
        raw = self.scheduler.stats()
        return EngineStats(
            num_running=int(raw["num_running"]),
            num_waiting=int(raw["num_waiting"]),
            num_preempted=int(raw["num_preempted"]),
            num_unfinished=int(raw["num_unfinished"]),
            num_finished=int(raw["num_finished"]),
            num_preemptions=int(raw["num_preemptions"]),
            num_steps=int(raw["num_steps"]),
            kv_utilization=float(raw.get("utilization", 0.0)),
            prefix_hit_rate=float(raw.get("prefix_hit_rate", 0.0)),
            num_prompt_tokens=self._num_prompt_tokens,
            num_generated_tokens=self._num_generated_tokens,
            num_cached_prompt_tokens=int(raw.get("num_cached_tokens", 0)),
        )

    def stats(self) -> dict[str, int | float | str]:
        """Everything the scheduler, block pool and prefix cache know, plus engine counters.

        Flat and JSON-safe so the gateway can label Prometheus gauges from it directly and a
        benchmark can embed it in a result file without reshaping.
        """
        out: dict[str, int | float | str] = dict(self.scheduler.stats())
        out.update(self.engine_stats().to_dict())
        out["device"] = str(self._device)
        out["dtype"] = str(self._dtype).removeprefix("torch.")
        out["model"] = self.config.model
        out["num_kv_blocks"] = self.kv_cache.num_blocks
        out["block_size"] = self.kv_cache.block_size
        out["kv_bytes"] = self.kv_cache.total_bytes
        return out

    def close(self) -> None:
        """Release the KV pool and drop every in-flight request. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.scheduler.reset()
        self._decoders.clear()
        self.kv_cache.free()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("engine closed")

    def __enter__(self) -> LLMEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"LLMEngine(model={self.config.model!r}, device={self._device}, "
            f"blocks={self.kv_cache.num_blocks}, unfinished={self.scheduler.num_unfinished})"
        )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

engine_app = typer.Typer(
    name="engine",
    help="Run the reference engine directly, without the gateway.",
    no_args_is_help=True,
)


@engine_app.command("generate")
def generate_command(
    prompt: Annotated[list[str], typer.Argument(help="One or more prompts to complete.")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model repo id or path.")] = "",
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 64,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0)] = 0.0,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 1.0,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed for stochastic sampling.")
    ] = None,
    device: Annotated[str, typer.Option("--device", help="auto, cuda or cpu.")] = "auto",
    dtype: Annotated[
        str, typer.Option("--dtype", help="auto, float16, bfloat16 or float32.")
    ] = "auto",
    num_blocks: Annotated[
        int | None, typer.Option("--num-blocks", min=1, help="Skip memory profiling.")
    ] = None,
    max_num_seqs: Annotated[int, typer.Option("--max-num-seqs", min=1)] = 8,
    max_num_batched_tokens: Annotated[int, typer.Option("--max-num-batched-tokens", min=1)] = 2048,
    block_size: Annotated[int, typer.Option("--block-size", min=1)] = 16,
    prefix_caching: Annotated[bool, typer.Option("--prefix-caching/--no-prefix-caching")] = True,
    offline: Annotated[
        bool, typer.Option("--offline", help="Never contact the Hub; use the local cache only.")
    ] = False,
    show_stats: Annotated[bool, typer.Option("--stats/--no-stats")] = False,
) -> None:
    """Complete one or more prompts and print the results.

    A smoke command, not a serving path: it runs every prompt to completion in one engine
    and prints the text. Use it to check that a checkpoint loads, that the KV pool sizes
    sensibly on a given device, and that continuous batching produces sane tokens.
    """
    from rich.console import Console

    from turboserve.config import get_settings

    settings = get_settings()
    config = EngineConfig(
        model=model or settings.model,
        device=device,  # type: ignore[arg-type]  # typer cannot express a Literal option
        dtype=dtype,  # type: ignore[arg-type]
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
            num_blocks=num_blocks,
            enable_prefix_caching=prefix_caching,
        ),
        seed=seed,
    )
    console = Console()
    sampling = SamplingParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
    )
    with LLMEngine(config, local_files_only=offline) as engine:
        started = time.perf_counter()
        outputs = engine.generate(list(prompt), sampling)
        elapsed = time.perf_counter() - started
        for index, output in enumerate(outputs):
            console.print(f"[bold cyan]prompt {index}[/bold cyan]: {prompt[index]}")
            console.print(f"[bold green]output {index}[/bold green]: {output.text_delta}")
            console.print(f"  {output.output_tokens} tokens, finish_reason={output.finish_reason}")
        console.print(f"[dim]{len(outputs)} completions in {elapsed:.2f}s[/dim]")
        if show_stats:
            console.print_json(data=engine.stats())


@engine_app.command("kv-size")
def kv_size_command(
    model: Annotated[str, typer.Option("--model", "-m", help="Model repo id or path.")] = "",
    device: Annotated[str, typer.Option("--device", help="auto, cuda or cpu.")] = "auto",
    dtype: Annotated[
        str, typer.Option("--dtype", help="auto, float16, bfloat16 or float32.")
    ] = "auto",
    block_size: Annotated[int, typer.Option("--block-size", min=1)] = 16,
    max_num_seqs: Annotated[int, typer.Option("--max-num-seqs", min=1)] = 64,
    max_num_batched_tokens: Annotated[int, typer.Option("--max-num-batched-tokens", min=1)] = 2048,
    gpu_memory_utilization: Annotated[
        float, typer.Option("--gpu-memory-utilization", min=0.01, max=1.0)
    ] = 0.9,
    max_model_len: Annotated[int | None, typer.Option("--max-model-len", min=1)] = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Never contact the Hub; use the local cache only.")
    ] = False,
) -> None:
    """Report how many KV blocks this configuration would get, and why.

    Capacity planning without loading a checkpoint: only ``config.json`` is read, and the
    answer is the same arithmetic the engine performs at startup. Use it to see what a
    block size or a token budget costs before renting a GPU for it.
    """
    from rich.console import Console

    from turboserve.config import get_settings
    from turboserve.engine.model.model_config import ModelConfig
    from turboserve.engine.runtime.memory import size_kv_cache

    settings = get_settings()
    config = EngineConfig(
        model=model or settings.model,
        device=device,  # type: ignore[arg-type]
        dtype=dtype,  # type: ignore[arg-type]
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
        ),
    )
    model_config = ModelConfig.from_hf(config.model, local_files_only=offline)
    sizing = size_kv_cache(
        model_config,
        config.scheduler,
        device=config.resolved_device(),
        dtype=config.resolved_dtype(),
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
    )
    Console().print_json(data=sizing.to_dict())
