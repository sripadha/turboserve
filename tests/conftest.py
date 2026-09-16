"""Shared pytest fixtures: markers, cached tiny models, device selection, CUDA guard.

The default suite must run in seconds on CPU and must never touch the network (see
``CONTRIBUTING.md``): the tiny-random models below are resolved from the local Hugging
Face cache with ``local_files_only=True`` and the test is skipped when they are absent,
so a fresh clone degrades to "skipped", never to a multi-gigabyte download.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Tiny random Qwen2 checkpoints, tried in order. Both are a handful of megabytes and
#: carry a real Qwen2 tokenizer, which is what the engine's Qwen2 path is tested against.
TINY_QWEN2_REPOS: tuple[str, ...] = (
    "hf-tiny-v2/tiny-random-Qwen2ForCausalLM",
    "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
)

#: Tiny random Llama checkpoint, used for the second supported architecture.
TINY_LLAMA_REPOS: tuple[str, ...] = ("hf-internal-testing/tiny-random-LlamaForCausalLM",)


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers so ``-W error::pytest.PytestUnknownMarkWarning`` stays clean.

    They are also declared in ``pyproject.toml``; registering them here keeps the markers
    available when a module owner runs pytest against a single test file with ``-p
    no:cacheprovider`` or from another working directory.
    """
    config.addinivalue_line(
        "markers", "slow: loads a real (non-tiny) model or otherwise takes minutes"
    )
    config.addinivalue_line("markers", "gpu: requires a CUDA device")


def _cached_snapshot(repo_ids: tuple[str, ...]) -> Path:
    """Return the local snapshot dir of the first cached repo, or skip the test.

    ``local_files_only=True`` is deliberate: unit tests never download. To enable these
    tests on a fresh machine, populate the cache once outside the test run (for example
    ``hf download hf-tiny-v2/tiny-random-Qwen2ForCausalLM``) or point ``HF_HOME`` at a
    cache that already has them.
    """
    from huggingface_hub import snapshot_download

    tried: list[str] = []
    for repo_id in repo_ids:
        try:
            return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
        except (OSError, ValueError) as exc:  # not cached, or the cache entry is incomplete
            tried.append(f"{repo_id}: {type(exc).__name__}")
    pytest.skip("no cached tiny model among " + ", ".join(tried))


@pytest.fixture(scope="session")
def tiny_qwen2_path() -> Path:
    """Path to a cached tiny-random Qwen2 checkpoint (config + weights + tokenizer)."""
    return _cached_snapshot(TINY_QWEN2_REPOS)


@pytest.fixture(scope="session")
def tiny_llama_path() -> Path:
    """Path to a cached tiny-random Llama checkpoint (config + weights + tokenizer)."""
    return _cached_snapshot(TINY_LLAMA_REPOS)


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Whether torch reports a usable CUDA device in this process."""
    import torch

    return bool(torch.cuda.is_available())


@pytest.fixture
def device(cuda_available: bool) -> str:
    """``"cuda"`` when available and not disabled, else ``"cpu"``.

    ``TURBOSERVE_TEST_DEVICE=cpu`` forces CPU, which is how the CPU-only CI job runs the
    same tests on a machine that happens to have a driver.
    """
    forced = os.environ.get("TURBOSERVE_TEST_DEVICE")
    if forced:
        return forced
    return "cuda" if cuda_available else "cpu"


@pytest.fixture(autouse=True)
def _skip_gpu_tests_without_cuda(request: pytest.FixtureRequest) -> None:
    """Skip ``gpu``-marked tests when there is no CUDA device.

    The default ``addopts`` already deselect them; this guard covers the explicit
    ``pytest -m gpu`` run on a machine without a GPU (a laptop, or CI).
    """
    if request.node.get_closest_marker("gpu") is None:
        return
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for gpu-marked tests")
