"""GPU smoke tests: the device this repo will be measured on is usable and reported honestly.

These are deliberately tiny (a few kilobytes of VRAM, milliseconds) so that ``make test-gpu``
is a real, non-vacuous check on any CUDA box, including the 6 GB development GPU. Engine and
kernel tests that need more than that belong to their module owners' suites.
"""

from __future__ import annotations

import pytest

from turboserve import hwinfo


@pytest.mark.gpu
def test_small_allocation_round_trips(device: str) -> None:
    """A modest allocation must survive a device round trip (far under the 1.5 GB budget)."""
    import torch

    assert device == "cuda", "gpu-marked tests must not be forced onto CPU"
    x = torch.arange(1024, device="cuda", dtype=torch.float16)
    y = (x * 2).cpu()
    assert y[-1].item() == pytest.approx(2046.0)
    del x, y
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_hwinfo_reports_the_same_device_as_torch() -> None:
    """hwinfo is embedded in every result file, so it must not disagree with torch."""
    import torch

    payload = hwinfo.collect()
    assert payload["torch"]["cuda_available"] is True
    assert payload["torch"]["devices"], "CUDA is available but no devices were reported"
    assert payload["torch"]["devices"][0]["name"] == torch.cuda.get_device_properties(0).name
    assert payload["gpu_name"] == torch.cuda.get_device_properties(0).name


@pytest.mark.gpu
def test_nvidia_smi_agrees_with_torch_on_the_device_name() -> None:
    """The driver and the torch runtime must be describing the same card."""
    import torch

    smi = hwinfo.nvidia_smi_info()
    if not smi.get("gpus"):
        pytest.skip("nvidia-smi is unavailable or reported no GPU")
    assert smi["gpus"][0]["name"] == torch.cuda.get_device_properties(0).name


@pytest.mark.gpu
def test_paged_kv_cache_shaped_allocation_fits_the_test_budget() -> None:
    """Allocate a KV-cache-shaped block table on device: the engine's core allocation pattern.

    Sized to ~64 MB so it is safe next to anything else on a shared development GPU, while
    still exercising the contiguous (num_blocks, block_size, num_kv_heads, head_dim) layout
    the paged attention kernels index into.
    """
    import torch

    num_blocks, block_size, num_kv_heads, head_dim = 512, 16, 4, 64
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    k = torch.zeros(
        (num_blocks, block_size, num_kv_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    v = torch.zeros_like(k)
    assert k.is_contiguous() and v.is_contiguous()
    assert k.numel() == num_blocks * block_size * num_kv_heads * head_dim
    used = torch.cuda.memory_allocated() - before
    assert used < 1.5 * 1024**3, f"gpu test budget exceeded: {used} bytes"
    del k, v
    torch.cuda.empty_cache()
