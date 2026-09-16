"""Multi-tenant LoRA: many adapters over one base model, mixed freely within a step.

Read the modules in this order:

* :mod:`~turboserve.engine.lora.adapter` -- what a PEFT adapter directory contains and how
  it becomes tensors.
* :mod:`~turboserve.engine.lora.layers` -- the projection that applies a different adapter
  to different rows of one batch (SGMV), and how it is installed into a model.
* :mod:`~turboserve.engine.lora.triton_bgmv` -- the decode-time kernel.
* :mod:`~turboserve.engine.lora.registry` -- which adapters exist, which are resident, and
  what that costs in VRAM.

The whole thing is two calls at the engine's edge::

    registry = LoRARegistry(max_gpu_adapters=32, max_lora_rank=16)
    registry.register("tenant-0", "adapters/tenant-0")
    install_lora(engine, registry)

after which any request whose ``lora_id`` is a registered adapter id is served with that
adapter, and any request without one is served by the base model, in the same step.

``docs/multi-lora.md`` covers the design, the SGMV grouping and the VRAM arithmetic.
"""

from __future__ import annotations

from turboserve.engine.lora.adapter import (
    AdapterError,
    LoRAAdapter,
    LoRAWeights,
    adapter_scaling,
    discover_adapters,
    linear_shapes,
    load_peft_adapter,
    merged_state_dict,
    peft_module_name,
)
from turboserve.engine.lora.layers import (
    DEFAULT_BGMV_MAX_TOKENS,
    LoRABatch,
    LoRALinear,
    install_lora,
    iter_lora_linears,
    lora_linear_factory,
    lora_segments,
    uninstall_lora,
)
from turboserve.engine.lora.registry import (
    LoRACapacityError,
    LoRAOptions,
    LoRARegistry,
    LoRARegistryError,
    RegistryStats,
    SlotInfo,
    UnknownAdapterError,
    setup_lora,
)
from turboserve.engine.lora.triton_bgmv import (
    MAX_BGMV_RANK,
    TRITON_AVAILABLE,
    bgmv_delta,
    can_use_bgmv,
)

__all__ = [
    "DEFAULT_BGMV_MAX_TOKENS",
    "MAX_BGMV_RANK",
    "TRITON_AVAILABLE",
    "AdapterError",
    "LoRAAdapter",
    "LoRABatch",
    "LoRACapacityError",
    "LoRALinear",
    "LoRAOptions",
    "LoRARegistry",
    "LoRARegistryError",
    "LoRAWeights",
    "RegistryStats",
    "SlotInfo",
    "UnknownAdapterError",
    "adapter_scaling",
    "bgmv_delta",
    "can_use_bgmv",
    "discover_adapters",
    "install_lora",
    "iter_lora_linears",
    "linear_shapes",
    "load_peft_adapter",
    "lora_linear_factory",
    "lora_segments",
    "merged_state_dict",
    "peft_module_name",
    "setup_lora",
    "uninstall_lora",
]
