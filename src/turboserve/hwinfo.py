"""Collect the hardware/software facts that every result file must carry.

A benchmark number is meaningless without the machine and the software stack that
produced it, so :func:`collect` is called by the benchmark drivers and embedded verbatim
in each ``results/*.json``. It must never raise: on a machine without an NVIDIA driver,
without git, or without ``/proc``, the corresponding fields are ``None`` instead.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Every external command below is given this many seconds before it is abandoned.
COMMAND_TIMEOUT_S = 10.0

SCHEMA_VERSION = 1


def _run(args: list[str], *, cwd: Path | None = None) -> str | None:
    """Run a command and return its stripped stdout, or ``None`` if it cannot run."""
    executable = shutil.which(args[0])
    if executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [executable, *args[1:]],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            check=False,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("command %s failed: %s", args, exc)
        return None
    if completed.returncode != 0:
        logger.debug(
            "command %s exited %d: %s", args, completed.returncode, completed.stderr.strip()
        )
        return None
    return completed.stdout.strip() or None


def nvidia_smi_info() -> dict[str, Any]:
    """Driver and per-GPU facts from ``nvidia-smi``; empty dict when it is absent."""
    header = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if header is None:
        return {}
    gpus: list[dict[str, Any]] = []
    driver_version: str | None = None
    for line in header.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, driver, memory_mib, compute_cap = parts
        driver_version = driver
        gpus.append(
            {
                "name": name,
                "memory_total_bytes": int(float(memory_mib)) * 1024 * 1024,
                "compute_capability": compute_cap,
            }
        )
    info: dict[str, Any] = {"driver_version": driver_version, "gpus": gpus}
    cuda_line = _run(["nvidia-smi", "--query"])
    if cuda_line is not None:
        for line in cuda_line.splitlines():
            if line.startswith("CUDA Version"):
                info["driver_cuda_version"] = line.split(":", 1)[1].strip()
                break
    return info


def torch_info() -> dict[str, Any]:
    """torch build facts plus the CUDA device list as torch itself sees it."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
    }
    if not torch.cuda.is_available():
        return info
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": int(props.total_memory),
                "multi_processor_count": int(props.multi_processor_count),
            }
        )
    info["devices"] = devices
    return info


def triton_version() -> str | None:
    """Version of the Triton compiler shipped with torch, if importable."""
    try:
        import triton
    except ImportError:
        return None
    return str(triton.__version__)


def memory_total_bytes() -> int | None:
    """Total system RAM, read from ``/proc/meminfo`` with a ``sysconf`` fallback."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None


def git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Commit sha, branch and dirty flag of the checkout this code runs from."""
    root = repo_root or Path(__file__).resolve().parents[2]
    sha = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if sha is None:
        return {"sha": None, "branch": None, "dirty": None}
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain"], cwd=root)
    return {"sha": sha, "branch": branch, "dirty": bool(status)}


def collect(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Return the full hardware/software record embedded in every result file."""
    torch_facts = torch_info()
    nvidia = nvidia_smi_info()
    gpu_name: str | None = None
    if torch_facts.get("devices"):
        gpu_name = torch_facts["devices"][0]["name"]
    elif nvidia.get("gpus"):
        gpu_name = nvidia["gpus"][0]["name"]
    return {
        "schema_version": SCHEMA_VERSION,
        "collected_at": datetime.now(UTC).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "cpu_count_affinity": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "memory_total_bytes": memory_total_bytes(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "gpu_name": gpu_name,
        "nvidia_smi": nvidia,
        "torch": torch_facts,
        "triton_version": triton_version(),
        "git": git_info(repo_root),
        "package_version": _package_version(),
    }


def _package_version() -> str:
    from turboserve._version import __version__

    return __version__


def to_json(info: dict[str, Any] | None = None, *, indent: int = 2) -> str:
    """Serialise a record (collecting a fresh one when not given) as JSON text."""
    return json.dumps(info if info is not None else collect(), indent=indent, sort_keys=True)
