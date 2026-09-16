"""Model-free core: sequences, block/KV allocation, prefix cache, scheduler and sampler.

Everything in this package is pure Python and tensor bookkeeping -- no model, no tokenizer,
no device assumptions beyond what the caller passes in -- which is what makes the engine's
scheduling behaviour testable in milliseconds on CPU. The layering runs one way only::

    types  ->  kv_cache  ->  prefix_cache  ->  block_manager  ->  scheduler
      \\-----> sequence --------^                                   ^
               sampler --------------------------------------------/

``types`` holds the contracts the rest of the engine, the gateway and the benchmark client
share; ``kv_cache`` owns the memory; ``prefix_cache`` decides what is worth keeping;
``block_manager`` joins the two to sequences; ``scheduler`` decides what runs next; and
``sampler`` turns a step's logits back into tokens.

This module re-exports the names other packages are expected to import, so that
``from turboserve.engine.core import Scheduler, SchedulerOutput`` keeps working if a class
moves between modules inside the package.
"""

from turboserve.engine.core.block_manager import BlockManager
from turboserve.engine.core.kv_cache import (
    BlockAllocator,
    BlockAllocatorError,
    BlockRecycler,
    InvalidBlockError,
    KVCache,
    OutOfBlocksError,
)
from turboserve.engine.core.prefix_cache import (
    ROOT_HASH,
    PrefixCache,
    PrefixMatch,
    block_hash,
    block_hash_chain,
)
from turboserve.engine.core.sampler import (
    Sampler,
    SamplerOutput,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
)
from turboserve.engine.core.scheduler import ScheduledSeq, Scheduler, SchedulerOutput
from turboserve.engine.core.sequence import DEFAULT_TENANT, SeqStatus, Sequence
from turboserve.engine.core.types import (
    DTYPE_BY_NAME,
    NO_LORA,
    PAD_BLOCK,
    AttnMetadata,
    DeviceName,
    DTypeName,
    EngineConfig,
    FinishReason,
    LoRAContext,
    RequestOutput,
    RequestTiming,
    SamplingParams,
    SchedulerConfig,
    SchedulerPolicy,
    build_query_start_loc,
    pad_block_tables,
    resolve_dtype,
)

__all__ = [
    "DEFAULT_TENANT",
    "DTYPE_BY_NAME",
    "NO_LORA",
    "PAD_BLOCK",
    "ROOT_HASH",
    "AttnMetadata",
    "BlockAllocator",
    "BlockAllocatorError",
    "BlockManager",
    "BlockRecycler",
    "DTypeName",
    "DeviceName",
    "EngineConfig",
    "FinishReason",
    "InvalidBlockError",
    "KVCache",
    "LoRAContext",
    "OutOfBlocksError",
    "PrefixCache",
    "PrefixMatch",
    "RequestOutput",
    "RequestTiming",
    "Sampler",
    "SamplerOutput",
    "SamplingParams",
    "ScheduledSeq",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    "SchedulerPolicy",
    "SeqStatus",
    "Sequence",
    "apply_repetition_penalty",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "block_hash",
    "block_hash_chain",
    "build_query_start_loc",
    "pad_block_tables",
    "resolve_dtype",
]
