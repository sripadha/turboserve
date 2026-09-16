"""Streaming safetensors loader: Hugging Face checkpoint names to engine parameter names.

Weights are the largest thing the engine touches, so they are never materialised twice.
:func:`load_weights` opens each shard with ``safetensors``' lazy reader, copies one tensor
at a time straight into the already-allocated parameter (casting dtype and moving device
in the same ``copy_``), and never builds a full ``state_dict`` in host memory. For a 7B
checkpoint that is the difference between one copy of the weights resident and three.

Two details matter for correctness rather than memory:

* **Name mapping.** The engine's module tree is deliberately named like the HF one
  (``model.layers.N.self_attn.q_proj.weight``), so the map is mostly the identity -- but
  it must still *drop* the non-parameter tensors some older checkpoints ship (rotary
  ``inv_freq`` buffers, causal-mask buffers), which would otherwise be reported as
  unexpected keys and mask a genuine mismatch.
* **Tied embeddings.** When ``tie_word_embeddings`` is set the checkpoint has no
  ``lm_head.weight``; the caller ties the parameter object before loading and this module
  simply must not complain about the missing key. :class:`LoadReport` therefore
  distinguishes "missing" from "tied".

Only safetensors are supported. A ``pytorch_model.bin`` checkpoint raises: loading it
means ``torch.load`` on a pickle from an arbitrary repo, and every model this engine
targets publishes safetensors.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

logger = logging.getLogger(__name__)

#: Single-file checkpoint name.
SAFETENSORS_FILE = "model.safetensors"
#: Sharded checkpoint index name (maps parameter -> shard filename).
SAFETENSORS_INDEX = "model.safetensors.index.json"

#: Checkpoint tensors that are recomputed at construction time rather than loaded.
#: ``inv_freq`` is the rotary frequency table (derived from ``rope_theta``); the mask
#: buffers are artefacts of very old exports.
_SKIP_SUFFIXES: tuple[str, ...] = (
    ".rotary_emb.inv_freq",
    ".attn.bias",
    ".attn.masked_bias",
    ".masked_bias",
)

#: Prefixes some exporters put in front of the decoder stack.
_PREFIX_REWRITES: tuple[tuple[str, str], ...] = (
    ("transformer.", "model."),
    ("language_model.model.", "model."),
    ("language_model.lm_head.", "lm_head."),
)

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class WeightLoadError(RuntimeError):
    """A checkpoint could not be read, or does not match the model that was built."""


@dataclass(slots=True)
class LoadReport:
    """What :func:`load_weights` actually did, for logging and for tests.

    Tests assert ``missing == []`` and ``unexpected == []``: a silently unloaded
    ``o_proj.weight`` leaves a randomly initialised layer in the middle of the model and
    the only symptom is degraded output quality, which no amount of shape checking finds.
    """

    num_tensors: int = 0
    num_bytes: int = 0
    files: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    """Parameters of the model that no checkpoint tensor supplied."""

    unexpected: list[str] = field(default_factory=list)
    """Checkpoint tensors with no matching parameter (after mapping and skipping)."""

    tied: list[str] = field(default_factory=list)
    """Parameters intentionally satisfied by weight tying rather than by a tensor."""

    @property
    def ok(self) -> bool:
        """Whether every parameter was filled and every tensor consumed."""
        return not self.missing and not self.unexpected

    def summary(self) -> str:
        """One-line human-readable form used by the debug log."""
        return (
            f"{self.num_tensors} tensors ({self.num_bytes} bytes) from "
            f"{len(self.files)} file(s); missing={len(self.missing)} "
            f"unexpected={len(self.unexpected)} tied={len(self.tied)}"
        )


def resolve_model_path(
    path_or_id: str | Path,
    *,
    local_files_only: bool = False,
    revision: str | None = None,
) -> Path:
    """Return a local directory for a checkpoint path or a Hub repo id.

    A path that exists on disk is returned as-is (no Hub call at all), which is what the
    unit tests and the vast.ai runbook both rely on: models are staged onto the instance
    once and every later run is offline.
    """
    candidate = Path(path_or_id).expanduser()
    if candidate.is_dir():
        return candidate
    if candidate.exists():
        raise WeightLoadError(f"{candidate} is a file; expected a checkpoint directory")

    from huggingface_hub import snapshot_download

    try:
        local = snapshot_download(
            repo_id=str(path_or_id), revision=revision, local_files_only=local_files_only
        )
    except Exception as exc:
        raise WeightLoadError(
            f"could not resolve {path_or_id!r} to a local checkpoint "
            f"(local_files_only={local_files_only}): {exc}"
        ) from exc
    return Path(local)


def safetensors_files(model_dir: Path) -> list[Path]:
    """List the checkpoint's safetensors shards, in index order when sharded.

    Shard order is taken from ``model.safetensors.index.json`` rather than from a glob so
    that a checkpoint whose shards are numbered ``00002-of-00011`` is not read in
    lexicographic order, and so that a shard named in the index but absent on disk is a
    loud error instead of a set of missing parameters.
    """
    if not model_dir.is_dir():
        raise WeightLoadError(f"{model_dir} is not a directory")

    index_path = model_dir / SAFETENSORS_INDEX
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise WeightLoadError(f"{index_path} has no usable weight_map")
        names: list[str] = []
        for shard in weight_map.values():
            if shard not in names:
                names.append(str(shard))
        files = [model_dir / name for name in sorted(names)]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise WeightLoadError(f"shards listed in {index_path} are absent: {missing}")
        return files

    single = model_dir / SAFETENSORS_FILE
    if single.is_file():
        return [single]

    shards = sorted(model_dir.glob("*.safetensors"))
    if shards:
        return shards
    legacy = sorted(model_dir.glob("*.bin")) + sorted(model_dir.glob("*.pth"))
    if legacy:
        raise WeightLoadError(
            f"{model_dir} ships only pickle checkpoints ({legacy[0].name}); turboserve "
            "loads safetensors exclusively. Convert with "
            "`transformers` (`save_pretrained(..., safe_serialization=True)`) first."
        )
    raise WeightLoadError(f"no safetensors checkpoint under {model_dir}")


def map_weight_name(name: str) -> str | None:
    """Map a checkpoint tensor name to an engine parameter name, or ``None`` to skip it.

    The engine's parameter names mirror the HF ones, so the interesting cases are the
    rewrites (``transformer.h.0`` style prefixes) and the skips (derived buffers). Keeping
    this a pure function makes the mapping table testable without a checkpoint.
    """
    if any(name.endswith(suffix) for suffix in _SKIP_SUFFIXES):
        return None
    for old, new in _PREFIX_REWRITES:
        if name.startswith(old):
            name = new + name[len(old) :]
            break
    if name.startswith("model.decoder."):
        name = "model." + name[len("model.decoder.") :]
    return name


def iter_checkpoint_tensors(
    files: Sequence[Path], *, device: torch.device | str = "cpu"
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(mapped_name, tensor)`` for every parameter tensor in ``files``.

    Tensors are produced lazily, one at a time, so peak host memory is one shard's largest
    tensor rather than the whole checkpoint.
    """
    target = torch.device(device)
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for raw_name in reader.keys():  # noqa: SIM118 - safe_open is not a Mapping
                mapped = map_weight_name(raw_name)
                if mapped is None:
                    logger.debug("skipping derived checkpoint tensor %s", raw_name)
                    continue
                tensor = reader.get_tensor(raw_name)
                yield mapped, tensor if target.type == "cpu" else tensor.to(target)


def load_weights(
    module: nn.Module,
    files: Sequence[Path],
    *,
    dtype: torch.dtype | None = None,
    tied_parameters: Sequence[str] = (),
    strict: bool = True,
) -> LoadReport:
    """Copy every checkpoint tensor in ``files`` into ``module``'s parameters.

    ``module`` must already be constructed with the right shapes: this loader copies into
    existing storage (``param.data.copy_``) rather than replacing parameter objects, which
    is what keeps weight tying (``lm_head.weight is embed_tokens.weight``) and any LoRA
    wrapper that captured a reference to the base linear valid after loading.

    ``tied_parameters`` names parameters that are expected to be absent from the
    checkpoint because they share storage with another one; they are reported under
    :attr:`LoadReport.tied` instead of :attr:`LoadReport.missing`.
    """
    params: dict[str, torch.Tensor] = dict(module.named_parameters())
    # Persistent buffers are part of the checkpoint; non-persistent ones (the rotary
    # tables, which are derived from rope_theta) are not, and must not be reported missing.
    persistent = set(module.state_dict().keys())
    for buffer_name, buffer in module.named_buffers():
        if buffer_name in persistent:
            params[buffer_name] = buffer
    seen: set[str] = set()
    report = LoadReport(files=[str(path) for path in files])

    for name, tensor in iter_checkpoint_tensors(files):
        target = params.get(name)
        if target is None:
            report.unexpected.append(name)
            continue
        if tuple(target.shape) != tuple(tensor.shape):
            raise WeightLoadError(
                f"shape mismatch for {name}: model has {tuple(target.shape)}, "
                f"checkpoint has {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            target.copy_(tensor.to(dtype=dtype or target.dtype))
        seen.add(name)
        report.num_tensors += 1
        report.num_bytes += tensor.numel() * tensor.element_size()

    tied = set(tied_parameters)
    for name in params:
        if name in seen:
            continue
        (report.tied if name in tied else report.missing).append(name)

    logger.debug("loaded weights into %s: %s", type(module).__name__, report.summary())
    if strict and not report.ok:
        raise WeightLoadError(
            f"checkpoint does not match the model: missing={report.missing[:8]} "
            f"unexpected={report.unexpected[:8]}"
        )
    return report


def layer_index_of(name: str) -> int | None:
    """Decoder-layer index encoded in a parameter name, or ``None`` for a global weight.

    Used by the LoRA registry and by debug logging to group per-layer tensors without
    re-deriving the naming convention in three places.
    """
    match = _LAYER_RE.match(name)
    return int(match.group(1)) if match else None
