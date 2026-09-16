"""Process-wide settings, read from ``TURBOSERVE_*`` environment variables.

Every knob that the engine, the gateway and the benchmark drivers agree on lives here so
that a scenario can be reproduced from the environment alone; the result files written by
``bench/`` embed the dump of this object. Per-tenant and per-model data (quotas, prices,
adapter paths) is intentionally *not* here: it is fleet data that changes without a
restart and lives in the YAML files named by :attr:`Settings.tenants_file` and
:attr:`Settings.models_file`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DeviceName = Literal["auto", "cuda", "cpu"]
DTypeName = Literal["auto", "float16", "bfloat16", "float32"]
EngineName = Literal["reference", "mock", "vllm"]
SchedulerPolicy = Literal["fcfs", "tenant_fair"]

ENV_PREFIX = "TURBOSERVE_"


class Settings(BaseSettings):
    """Runtime configuration for one turboserve process.

    Invariants enforced here rather than at every call site: the token budget of a
    scheduler step is at least one block, ``gpu_memory_utilization`` is a fraction, and
    ``block_size`` is a positive power of two (paged attention indexes a slot as
    ``block * block_size + offset``, and the kernels assume a power-of-two stride).
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- process -----------------------------------------------------------------
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # --- model and placement ------------------------------------------------------
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: DeviceName = "auto"
    dtype: DTypeName = "auto"

    # --- engine -------------------------------------------------------------------
    engine: EngineName = "reference"
    block_size: int = Field(default=16, ge=1)
    num_blocks: int | None = Field(default=None, ge=1)
    gpu_memory_utilization: float = Field(default=0.85, gt=0.0, le=1.0)
    max_num_seqs: int = Field(default=64, ge=1)
    max_num_batched_tokens: int = Field(default=2048, ge=1)
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    scheduler_policy: SchedulerPolicy = "fcfs"

    # --- speculative decoding -----------------------------------------------------
    speculative_model: str | None = None
    num_speculative_tokens: int = Field(default=4, ge=1)

    # --- multi-LoRA ---------------------------------------------------------------
    enable_lora: bool = False
    max_loras: int = Field(default=8, ge=1)
    max_lora_rank: int = Field(default=16, ge=1)
    adapters_dir: Path = Path("adapters")

    # --- fleet data and outputs ---------------------------------------------------
    tenants_file: Path = Path("configs/tenants.yaml")
    models_file: Path = Path("configs/models.yaml")
    results_dir: Path = Path("results")

    @field_validator("block_size")
    @classmethod
    def _block_size_is_power_of_two(cls, value: int) -> int:
        if value & (value - 1):
            raise ValueError(f"block_size must be a power of two, got {value}")
        return value

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        import logging

        upper = value.strip().upper()
        if upper not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level: {value!r}")
        return upper

    def resolved_device(self) -> str:
        """Return ``"cuda"`` or ``"cpu"``, resolving ``"auto"`` by probing torch.

        torch is imported lazily so that ``turboserve version`` and config-only tests do
        not pay the multi-second import.
        """
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_dtype(self) -> str:
        """Return a concrete dtype name.

        ``auto`` means fp16 on CUDA and fp32 on CPU. bf16 is never chosen automatically:
        the target GPU here is Turing (sm_75), which has no bf16 tensor cores, and CPU
        bf16 matmuls in torch are slower than fp32.
        """
        if self.dtype != "auto":
            return self.dtype
        return "float16" if self.resolved_device() == "cuda" else "float32"

    @property
    def max_batched_blocks(self) -> int:
        """Blocks a single scheduler step can touch, used to size staging buffers."""
        return -(-self.max_num_batched_tokens // self.block_size)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, built once from the environment."""
    return Settings()
