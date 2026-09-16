"""Sizing the KV block pool from the memory the device actually has left.

The number of KV blocks is the single most consequential number in the engine: it sets how
many sequences can be resident, which sets the batch size, which sets throughput. Too many
blocks and the first long prompt drives the allocator into the driver's out-of-memory path;
too few and the scheduler preempts constantly. vLLM decides this by profiling; so does this
module, with one simplification that matters for a reference engine: instead of running a
synthetic worst-case batch and watching the allocator's high-water mark, it reads the free
memory *after the weights are loaded* and subtracts an analytic estimate of the transient
activations one scheduler step can need.

The trade is deliberate. A profiling run needs the model, a tokenizer and a device that can
hold a worst-case batch, so it cannot be unit-tested on a laptop, and it makes engine
construction depend on a forward pass that may itself fail. The analytic estimate is a
closed-form function of numbers the config already carries, so every branch here is testable
on CPU, and the one thing it cannot know -- fragmentation inside the caching allocator -- is
covered by ``gpu_memory_utilization`` being a fraction well below 1 by default.

Nothing in this module is a performance measurement. Every function returns byte counts
derived from the model's shape and the device's reported capacity.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from turboserve.engine.core.kv_cache import KVCache

if TYPE_CHECKING:  # pragma: no cover - typing only
    from turboserve.engine.core.types import SchedulerConfig
    from turboserve.engine.model.model_config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CPU_MEMORY_BYTES",
    "ACTIVATION_ELEMENTS_PER_TOKEN",
    "KVCacheSizing",
    "MemoryProbe",
    "MemoryProfileError",
    "activation_headroom_bytes",
    "build_kv_cache",
    "probe_memory",
    "size_kv_cache",
]

#: Hidden-state copies a single decoder layer holds alive at once during a step: the layer
#: input kept for the residual, the normalised copy, the attention output, the second
#: residual, and the two SwiGLU branches plus their product. Multiplying this by
#: ``hidden_size`` and the step's token count gives the transient activation footprint that
#: has to stay *outside* the KV pool. It is an upper bound on what torch keeps alive
#: simultaneously, not a measurement.
ACTIVATION_ELEMENTS_PER_TOKEN = 8

#: Fallback capacity used when the host's memory cannot be read (a non-Linux kernel, or a
#: ``/proc`` that is not mounted). Small on purpose: under-sizing the pool degrades
#: throughput, over-sizing it kills the process.
DEFAULT_CPU_MEMORY_BYTES = 2 * 1024**3

_MEMINFO = "/proc/meminfo"


class MemoryProfileError(RuntimeError):
    """Raised when the device cannot hold even one KV block under the configured budget."""


@dataclass(frozen=True, slots=True)
class MemoryProbe:
    """What the device reports about its memory at one instant.

    ``free_bytes`` is what is available *now*, i.e. after the model weights have been
    allocated, which is why the engine builds the model before sizing the pool.
    """

    device: str
    """``"cuda"``, ``"cuda:0"``, ``"cpu"`` -- the device this probe describes."""

    total_bytes: int
    """Physical capacity of the device."""

    free_bytes: int
    """Capacity not currently committed to anything."""

    source: str
    """Where the numbers came from: ``cuda``, ``proc-meminfo`` or ``fallback``."""

    @property
    def used_bytes(self) -> int:
        """Capacity currently committed, weights included."""
        return max(0, self.total_bytes - self.free_bytes)

    def to_dict(self) -> dict[str, int | str]:
        """Plain-data view for logs and for a result file's ``config`` block."""
        return {
            "device": self.device,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "used_bytes": self.used_bytes,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class KVCacheSizing:
    """The decision: how many blocks, and the arithmetic that produced it."""

    num_blocks: int
    block_size: int
    bytes_per_block: int
    """Bytes one block occupies across every layer, keys and values together."""

    budget_bytes: int
    """Bytes the KV pool was allowed to use."""

    activation_bytes: int
    """Bytes reserved for one step's transient activations, outside the pool."""

    probe: MemoryProbe
    reason: str
    """Which constraint decided the final count: ``explicit``, ``memory`` or ``max_model_len``."""

    @property
    def kv_bytes(self) -> int:
        """Bytes the chosen pool will actually occupy."""
        return self.num_blocks * self.bytes_per_block

    @property
    def num_slots(self) -> int:
        """Token slots in the pool: the total context the engine can hold at once."""
        return self.num_blocks * self.block_size

    def to_dict(self) -> dict[str, object]:
        """Plain-data view, embedded in the engine's ``stats()`` and in result files."""
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "bytes_per_block": self.bytes_per_block,
            "kv_bytes": self.kv_bytes,
            "num_slots": self.num_slots,
            "budget_bytes": self.budget_bytes,
            "activation_bytes": self.activation_bytes,
            "reason": self.reason,
            "probe": self.probe.to_dict(),
        }


def _read_meminfo_bytes(key: str) -> int | None:
    """Read one ``/proc/meminfo`` field in bytes, or ``None`` when it is unavailable."""
    try:
        with open(_MEMINFO, encoding="ascii") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                if name != key:
                    continue
                parts = rest.split()
                if not parts:
                    return None
                value = int(parts[0])
                unit = parts[1].lower() if len(parts) > 1 else "kb"
                return value * 1024 if unit == "kb" else value
    except (OSError, ValueError):
        return None
    return None


def probe_memory(device: torch.device | str) -> MemoryProbe:
    """Report total and free memory for ``device``.

    CUDA numbers come from the driver (``torch.cuda.mem_get_info``), which sees the whole
    device including memory held by other processes -- the right view when a benchmark
    shares a GPU with a stray notebook. Host numbers come from ``/proc/meminfo``'s
    ``MemAvailable``, which is the kernel's own estimate of what a new allocation can get
    without swapping, rather than ``MemFree``, which excludes reclaimable page cache and
    would under-size the pool severely on a machine that has been running a while.
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        return MemoryProbe(
            device=str(dev), total_bytes=int(total), free_bytes=int(free), source="cuda"
        )
    reported_total = _read_meminfo_bytes("MemTotal")
    reported_free = _read_meminfo_bytes("MemAvailable")
    if reported_total is None or reported_free is None:
        names = getattr(os, "sysconf_names", {})
        pages, page_size = names.get("SC_PHYS_PAGES"), names.get("SC_PAGE_SIZE")
        if pages is not None and page_size is not None:
            total = os.sysconf(pages) * os.sysconf(page_size)
        else:
            total = DEFAULT_CPU_MEMORY_BYTES
        return MemoryProbe(
            device=str(dev), total_bytes=int(total), free_bytes=int(total), source="fallback"
        )
    return MemoryProbe(
        device=str(dev),
        total_bytes=int(reported_total),
        free_bytes=int(reported_free),
        source="proc-meminfo",
    )


def activation_headroom_bytes(
    model_config: ModelConfig,
    *,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    dtype: torch.dtype,
) -> int:
    """Bytes one scheduler step can need for transients, outside the KV pool.

    Two terms, both linear in quantities the scheduler config already bounds:

    * hidden states: ``max_num_batched_tokens`` tokens times
      :data:`ACTIVATION_ELEMENTS_PER_TOKEN` copies of ``hidden_size``, plus the attention
      scores' Q/K/V staging, which is why the intermediate size enters through the MLP's
      two projections rather than being counted separately;
    * logits: one fp32 row of ``vocab_size`` per sampling sequence. For a 150k-token
      vocabulary this is the larger term at small batch sizes, and it is the one people
      forget.

    The result is an upper bound on the simultaneously-live transients, not a measurement of
    peak allocator usage: torch may hold more because of fragmentation, which is what the
    utilization fraction absorbs.
    """
    if max_num_batched_tokens < 1 or max_num_seqs < 1:
        raise ValueError("max_num_batched_tokens and max_num_seqs must be positive")
    element = dtype.itemsize
    widest = max(model_config.hidden_size, model_config.intermediate_size)
    hidden = max_num_batched_tokens * ACTIVATION_ELEMENTS_PER_TOKEN * widest * element
    qkv = (
        max_num_batched_tokens
        * (model_config.q_proj_size + 2 * model_config.kv_proj_size)
        * element
    )
    logits = max_num_seqs * model_config.vocab_size * torch.float32.itemsize
    return int(hidden + qkv + logits)


def size_kv_cache(
    model_config: ModelConfig,
    scheduler_config: SchedulerConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    gpu_memory_utilization: float = 0.9,
    num_blocks: int | None = None,
    max_model_len: int | None = None,
    probe: MemoryProbe | None = None,
) -> KVCacheSizing:
    """Decide how many KV blocks the engine gets.

    Precedence, highest first:

    1. an explicit ``num_blocks`` (from ``TURBOSERVE_NUM_BLOCKS`` or a benchmark profile),
       which is what makes a preemption test reproducible;
    2. the memory budget, ``utilization x total - already used - activations``;
    3. a cap at ``max_model_len``'s worth of blocks per the sequence budget, so a config
       that can never use the pool does not reserve it.

    ``utilization x total - used`` is the vLLM convention and is not the same as
    ``utilization x free``: it treats the fraction as a ceiling on the engine's share of the
    *whole* device, so another process already holding memory shrinks this engine's pool
    rather than being quietly overcommitted.

    Raises:
        MemoryProfileError: when the budget cannot pay for a single block. The message
            carries every term, because the fix is always to change one of them.
    """
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}")
    block_size = scheduler_config.block_size
    bytes_per_block = KVCache.bytes_per_block_for(
        num_layers=model_config.num_hidden_layers,
        block_size=block_size,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        dtype=dtype,
    )
    measured = probe if probe is not None else probe_memory(device)
    activations = activation_headroom_bytes(
        model_config,
        max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
        max_num_seqs=scheduler_config.max_num_seqs,
        dtype=dtype,
    )
    budget = int(measured.total_bytes * gpu_memory_utilization) - measured.used_bytes - activations
    budget = max(0, budget)

    if num_blocks is not None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        chosen, reason = num_blocks, "explicit"
        if num_blocks * bytes_per_block > budget:
            logger.warning(
                "explicit num_blocks=%d needs %d bytes but only %d are budgeted on %s; "
                "the allocator may run out of memory",
                num_blocks,
                num_blocks * bytes_per_block,
                budget,
                measured.device,
            )
    else:
        chosen = KVCache.blocks_for_memory(
            budget,
            num_layers=model_config.num_hidden_layers,
            block_size=block_size,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            dtype=dtype,
        )
        reason = "memory"
        if chosen < 1:
            raise MemoryProfileError(
                f"no room for a KV block on {measured.device}: "
                f"{measured.total_bytes} bytes total, {measured.used_bytes} already used, "
                f"{activations} reserved for activations, utilization "
                f"{gpu_memory_utilization} leaves {budget} bytes but one block needs "
                f"{bytes_per_block}. Lower max_num_batched_tokens/max_num_seqs, raise "
                f"gpu_memory_utilization, or use a smaller model."
            )
        cap = _max_useful_blocks(scheduler_config, block_size, max_model_len)
        if cap is not None and cap < chosen:
            chosen, reason = cap, "max_model_len"

    logger.info(
        "kv cache: %d blocks of %d tokens (%d bytes/block, %d bytes total) on %s [%s]",
        chosen,
        block_size,
        bytes_per_block,
        chosen * bytes_per_block,
        measured.device,
        reason,
    )
    return KVCacheSizing(
        num_blocks=chosen,
        block_size=block_size,
        bytes_per_block=bytes_per_block,
        budget_bytes=budget,
        activation_bytes=activations,
        probe=measured,
        reason=reason,
    )


def _max_useful_blocks(
    scheduler_config: SchedulerConfig, block_size: int, max_model_len: int | None
) -> int | None:
    """Blocks needed to hold ``max_num_seqs`` sequences of ``max_model_len`` tokens."""
    if max_model_len is None:
        return None
    per_seq = math.ceil(max_model_len / block_size)
    return max(1, per_seq * scheduler_config.max_num_seqs)


def build_kv_cache(
    model_config: ModelConfig,
    sizing: KVCacheSizing,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    allocate: bool = True,
) -> KVCache:
    """Construct (and by default materialise) the block pool described by ``sizing``."""
    cache = KVCache(
        num_layers=model_config.num_hidden_layers,
        num_blocks=sizing.num_blocks,
        block_size=sizing.block_size,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        dtype=dtype,
        device=device,
    )
    if allocate:
        cache.allocate()
    return cache
