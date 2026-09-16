"""Fixtures for the gateway-to-engine end-to-end tests.

The app builder and the SSE parser live in ``_support.py`` next to this file; only the
fixtures are here. Both integration modules run the same assertions through the same
builder and differ only in the checkpoint behind it — the cached tiny-random Qwen2 for the
default suite, a real Qwen2.5-0.5B for the ``slow`` one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _support import SLOW_MODEL_REPOS, running_gateway

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture(scope="session")
def qwen_05b_path() -> Path:
    """Local snapshot of a real 0.5B instruct checkpoint, or skip.

    ``local_files_only=True`` for the same reason the tiny fixtures use it: a test never
    downloads. Populate the cache once outside the test run to enable the ``slow`` tier.
    """
    from pathlib import Path as _Path

    from huggingface_hub import snapshot_download

    tried: list[str] = []
    for repo_id in SLOW_MODEL_REPOS:
        try:
            return _Path(snapshot_download(repo_id=repo_id, local_files_only=True))
        except (OSError, ValueError) as exc:
            tried.append(f"{repo_id}: {type(exc).__name__}")
    pytest.skip("no cached 0.5B checkpoint among " + ", ".join(tried))


@pytest.fixture
async def gateway(tiny_qwen2_path: Path) -> AsyncIterator[dict[str, Any]]:
    """A gateway over the cached tiny-random Qwen2: the default suite's fixture."""
    async for value in running_gateway(tiny_qwen2_path):
        yield value


@pytest.fixture
async def real_gateway(qwen_05b_path: Path) -> AsyncIterator[dict[str, Any]]:
    """A gateway over a real 0.5B instruct checkpoint: the ``slow`` suite's fixture."""
    async for value in running_gateway(qwen_05b_path):
        yield value
