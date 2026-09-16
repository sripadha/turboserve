"""Settings tests: environment prefix, validation invariants and derived values."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from turboserve.config import ENV_PREFIX, Settings, get_settings


def test_defaults_match_the_spec() -> None:
    settings = Settings()
    assert settings.engine == "reference"
    assert settings.block_size == 16
    assert settings.max_num_seqs == 64
    assert settings.max_num_batched_tokens == 2048
    assert settings.enable_chunked_prefill is True
    assert settings.enable_prefix_caching is True
    assert settings.scheduler_policy == "fcfs"
    assert settings.tenants_file == Path("configs/tenants.yaml")


def test_env_prefix_is_turboserve(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ENV_PREFIX == "TURBOSERVE_"
    monkeypatch.setenv("TURBOSERVE_PORT", "9123")
    monkeypatch.setenv("TURBOSERVE_MAX_NUM_SEQS", "7")
    monkeypatch.setenv("TURBOSERVE_ENABLE_PREFIX_CACHING", "false")
    settings = Settings(_env_file=None)
    assert settings.port == 9123
    assert settings.max_num_seqs == 7
    assert settings.enable_prefix_caching is False


def test_log_level_is_normalised_and_validated() -> None:
    assert Settings(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        Settings(log_level="chatty")


@pytest.mark.parametrize("bad", [3, 12, 100])
def test_block_size_must_be_a_power_of_two(bad: int) -> None:
    with pytest.raises(ValidationError):
        Settings(block_size=bad)


@pytest.mark.parametrize("good", [1, 8, 16, 32])
def test_power_of_two_block_sizes_are_accepted(good: int) -> None:
    assert Settings(block_size=good).block_size == good


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_gpu_memory_utilization_is_a_fraction(bad: float) -> None:
    with pytest.raises(ValidationError):
        Settings(gpu_memory_utilization=bad)


def test_settings_are_frozen() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.port = 1234  # type: ignore[misc]


def test_max_batched_blocks_rounds_up() -> None:
    settings = Settings(block_size=16, max_num_batched_tokens=33)
    assert settings.max_batched_blocks == 3


def test_resolved_device_and_dtype_agree() -> None:
    settings = Settings(device="cpu", dtype="auto")
    assert settings.resolved_device() == "cpu"
    assert settings.resolved_dtype() == "float32"
    explicit = Settings(device="cuda", dtype="bfloat16")
    assert explicit.resolved_device() == "cuda"
    assert explicit.resolved_dtype() == "bfloat16"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    assert get_settings() is first
    get_settings.cache_clear()
