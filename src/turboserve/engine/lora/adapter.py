"""Reading a PEFT LoRA adapter directory into tensors the engine can stack.

A LoRA adapter (Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models*,
https://arxiv.org/abs/2106.09685) replaces a weight update with the product of two thin
matrices: ``W' = W + (alpha / r) * B @ A`` with ``A`` of shape ``[r, in]`` and ``B`` of
shape ``[out, r]``. Serving many tenants from one base model means keeping ``W`` once and
every tenant's ``(A, B)`` beside it, so this module's job is to turn the on-disk PEFT
layout into exactly that pair per target projection, checked against the model it will be
applied to.

Three decisions are worth stating because they are the ones that bite:

* **Safetensors only.** ``adapter_model.bin`` is a pickle, and a serving process that
  loads tenant-supplied pickles executes tenant-supplied code. An adapter saved as
  ``.bin`` is rejected with a message telling the operator how to convert it.
* **Scaling is resolved per module, not per adapter.** PEFT's ``rank_pattern`` and
  ``alpha_pattern`` let one adapter carry different ranks per projection, and ``use_rslora``
  changes the denominator from ``r`` to ``sqrt(r)``. The engine stores one scalar per
  (slot, projection) anyway, so folding all of that into a per-module ``scaling`` here
  keeps every later stage ignorant of PEFT's option matrix.
* **Validation happens at registration, not at the first token.** An adapter whose
  ``q_proj`` is 4096 wide against a 3584-wide model is a configuration mistake; finding it
  when a tenant's request is already in flight turns it into an outage.

The unsupported PEFT variants (DoRA, ``modules_to_save``, LoRA on embeddings, biased
adapters) are refused explicitly rather than silently ignored: dropping ``modules_to_save``
would serve a tenant an adapter that is quietly not the one they trained.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Mapping, Sequence

    from torch import nn

logger = logging.getLogger(__name__)

__all__ = [
    "PEFT_CONFIG_FILE",
    "PEFT_PICKLE_FILE",
    "PEFT_WEIGHTS_FILE",
    "AdapterError",
    "LoRAAdapter",
    "LoRAWeights",
    "adapter_scaling",
    "discover_adapters",
    "linear_shapes",
    "load_peft_adapter",
    "merged_state_dict",
    "peft_module_name",
]

#: Metadata file every PEFT adapter directory carries.
PEFT_CONFIG_FILE = "adapter_config.json"

#: The tensor file this loader accepts.
PEFT_WEIGHTS_FILE = "adapter_model.safetensors"

#: The tensor file this loader refuses (see the module docstring).
PEFT_PICKLE_FILE = "adapter_model.bin"

#: Prefixes PEFT puts in front of the base model's module path.
_PEFT_PREFIXES: tuple[str, ...] = ("base_model.model.", "base_model.")

#: Suffixes of PEFT tensor keys this loader understands.
_A_MARKERS = ("lora_A",)
_B_MARKERS = ("lora_B",)

#: Key fragments that mean a PEFT feature the engine does not implement.
_UNSUPPORTED_MARKERS: tuple[tuple[str, str], ...] = (
    ("lora_embedding_A", "LoRA on embedding layers"),
    ("lora_embedding_B", "LoRA on embedding layers"),
    ("lora_magnitude_vector", "DoRA (weight-decomposed LoRA)"),
    ("lora_bias", "biased LoRA adapters"),
)


class AdapterError(ValueError):
    """An adapter directory is unreadable, malformed, or uses an unsupported variant."""


def adapter_scaling(rank: int, alpha: float, *, use_rslora: bool = False) -> float:
    """The scalar multiplying ``B @ A``.

    ``alpha / r`` is the original formulation; rank-stabilised LoRA (Kalajdzievski, 2023,
    https://arxiv.org/abs/2312.03732) uses ``alpha / sqrt(r)`` so that the update's
    magnitude does not shrink as the rank grows. Getting this wrong does not crash
    anything -- it scales every adapter's effect by a constant, which is exactly the kind
    of error that shows up as "the adapter does nothing" or "the model speaks nonsense".
    """
    if rank < 1:
        raise AdapterError(f"LoRA rank must be positive, got {rank}")
    denominator = math.sqrt(rank) if use_rslora else float(rank)
    return float(alpha) / denominator


def peft_module_name(key: str) -> tuple[str, str] | None:
    """Split a PEFT state-dict key into ``(engine module name, "A" | "B")``.

    Returns ``None`` for a key that is not a LoRA factor (PEFT writes a few bookkeeping
    tensors). Raises :class:`AdapterError` for a key that names a feature the engine does
    not implement, because silently skipping it would serve a different adapter than the
    one on disk.

    The ``.default`` in ``...lora_A.default.weight`` is the adapter name PEFT uses when a
    model carries several adapters at once; the engine keeps one adapter per directory, so
    any such component between the marker and ``weight`` is dropped.
    """
    for marker, feature in _UNSUPPORTED_MARKERS:
        if marker in key:
            raise AdapterError(f"unsupported PEFT feature in adapter weights: {feature} ({key})")
    parts = key.split(".")
    for index, part in enumerate(parts):
        if part in _A_MARKERS:
            side = "A"
        elif part in _B_MARKERS:
            side = "B"
        else:
            continue
        module = ".".join(parts[:index])
        for prefix in _PEFT_PREFIXES:
            if module.startswith(prefix):
                module = module[len(prefix) :]
                break
        if not module:
            raise AdapterError(f"adapter key {key!r} names no module")
        return module, side
    return None


@dataclass(frozen=True, slots=True)
class LoRAWeights:
    """One projection's adapter: ``A[r, in]``, ``B[out, r]`` and its scaling.

    Stored in PEFT's orientation (``A`` maps input to rank, ``B`` rank to output) so that
    the delta is ``x @ A.T @ B.T * scaling`` -- two ``F.linear`` calls with no transposes
    at serving time.
    """

    module: str
    """Dotted module path in the engine's model, e.g. ``model.layers.3.self_attn.q_proj``."""

    a: torch.Tensor
    """``[rank, in_features]``."""

    b: torch.Tensor
    """``[out_features, rank]``."""

    scaling: float
    """``alpha / r`` (or ``alpha / sqrt(r)`` under ``use_rslora``), folded per module."""

    def __post_init__(self) -> None:
        if self.a.dim() != 2 or self.b.dim() != 2:
            raise AdapterError(
                f"{self.module}: LoRA factors must be 2-D, got A{tuple(self.a.shape)} "
                f"B{tuple(self.b.shape)}"
            )
        if self.a.shape[0] != self.b.shape[1]:
            raise AdapterError(
                f"{self.module}: rank mismatch between A{tuple(self.a.shape)} and "
                f"B{tuple(self.b.shape)}"
            )

    @property
    def rank(self) -> int:
        """Inner dimension shared by ``A`` and ``B``."""
        return int(self.a.shape[0])

    @property
    def in_features(self) -> int:
        """Width of the projection's input."""
        return int(self.a.shape[1])

    @property
    def out_features(self) -> int:
        """Width of the projection's output."""
        return int(self.b.shape[0])

    @property
    def nbytes(self) -> int:
        """Bytes occupied by the two factors."""
        return self.a.numel() * self.a.element_size() + self.b.numel() * self.b.element_size()

    def to(
        self, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> LoRAWeights:
        """Return these factors cast and/or moved (``self`` when nothing changes)."""
        target = None if device is None else torch.device(device)
        if (dtype is None or dtype == self.a.dtype) and (target is None or target == self.a.device):
            return self
        return replace(
            self,
            a=self.a.to(dtype=dtype or self.a.dtype, device=target or self.a.device),
            b=self.b.to(dtype=dtype or self.b.dtype, device=target or self.b.device),
        )

    def merged_delta(self) -> torch.Tensor:
        """``scaling * B @ A`` -- the dense weight update this adapter represents.

        Only used by tests and by tooling that compares against a merged model; the
        serving path never materialises it, which is the entire point of LoRA.
        """
        return (self.b.to(torch.float32) @ self.a.to(torch.float32)) * self.scaling


@dataclass(frozen=True, slots=True)
class LoRAAdapter:
    """A whole adapter: every targeted projection's factors, plus its identity.

    Immutable because it is shared: one CPU-resident copy answers every activation of the
    adapter into a GPU slot, and a mutable object shared across those would make an
    eviction able to corrupt a live slot.
    """

    name: str
    """Name tenants and the gateway use for this adapter."""

    weights: dict[str, LoRAWeights]
    """Engine module path to that projection's factors."""

    rank: int
    """Largest rank across the adapter's modules (what a slot must be wide enough for)."""

    alpha: float
    """The ``lora_alpha`` from ``adapter_config.json``, kept for reporting."""

    target_modules: tuple[str, ...]
    """Projection suffixes the adapter was trained on, as recorded by PEFT."""

    base_model: str = ""
    """``base_model_name_or_path`` from the adapter config; "" when the file omits it."""

    path: Path | None = None
    """Directory it was loaded from; ``None`` for adapters built in memory."""

    use_rslora: bool = False
    """Whether scaling used the rank-stabilised denominator."""

    def __post_init__(self) -> None:
        if not self.name:
            raise AdapterError("adapter name must not be empty")
        if not self.weights:
            raise AdapterError(f"adapter {self.name!r} carries no LoRA factors")

    @property
    def modules(self) -> tuple[str, ...]:
        """Sorted module paths this adapter touches."""
        return tuple(sorted(self.weights))

    @property
    def nbytes(self) -> int:
        """Total bytes of every factor, i.e. what one resident copy costs."""
        return sum(weight.nbytes for weight in self.weights.values())

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the factors (they are stored uniformly)."""
        return next(iter(self.weights.values())).a.dtype

    @property
    def device(self) -> torch.device:
        """Device the factors currently live on."""
        return next(iter(self.weights.values())).a.device

    def __iter__(self) -> Iterator[LoRAWeights]:
        """Iterate the per-module factors in sorted module order."""
        for module in self.modules:
            yield self.weights[module]

    def __contains__(self, module: str) -> bool:
        """Whether this adapter has factors for ``module``."""
        return module in self.weights

    def to(
        self, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> LoRAAdapter:
        """Return the adapter with every factor cast and/or moved."""
        moved = {
            name: weight.to(dtype=dtype, device=device) for name, weight in self.weights.items()
        }
        if all(moved[name] is weight for name, weight in self.weights.items()):
            return self
        return replace(self, weights=moved)

    def validate_against(
        self, shapes: Mapping[str, tuple[int, int]], *, strict: bool = True
    ) -> None:
        """Check every factor against the model's projection shapes.

        ``shapes`` maps a module path to ``(in_features, out_features)``; build it with
        :func:`linear_shapes`. With ``strict`` the adapter must target only modules the
        model has -- the default, because an adapter trained for another architecture whose
        extra modules are quietly dropped produces a model that answers but is not the one
        the tenant asked for.
        """
        unknown = [module for module in self.modules if module not in shapes]
        if unknown and strict:
            raise AdapterError(
                f"adapter {self.name!r} targets modules the model does not have: "
                f"{unknown[:5]}{'...' if len(unknown) > 5 else ''}"
            )
        for module, weight in self.weights.items():
            expected = shapes.get(module)
            if expected is None:
                continue
            if (weight.in_features, weight.out_features) != expected:
                raise AdapterError(
                    f"adapter {self.name!r} module {module}: factors are "
                    f"{weight.in_features}->{weight.out_features}, model is "
                    f"{expected[0]}->{expected[1]}"
                )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe description, for result files and ``stats()`` blocks."""
        return {
            "name": self.name,
            "rank": self.rank,
            "alpha": self.alpha,
            "use_rslora": self.use_rslora,
            "target_modules": list(self.target_modules),
            "num_modules": len(self.weights),
            "bytes": self.nbytes,
            "dtype": str(self.dtype).removeprefix("torch."),
            "base_model": self.base_model,
            "path": None if self.path is None else str(self.path),
        }

    @classmethod
    def from_directory(
        cls,
        path: Path | str,
        *,
        name: str | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> LoRAAdapter:
        """Load a PEFT adapter directory. See :func:`load_peft_adapter`."""
        return load_peft_adapter(path, name=name, dtype=dtype, device=device)


def _read_config(directory: Path) -> dict[str, Any]:
    """Parse ``adapter_config.json``, rejecting variants the engine cannot serve."""
    config_path = directory / PEFT_CONFIG_FILE
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdapterError(f"{directory} is not a PEFT adapter: no {PEFT_CONFIG_FILE}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AdapterError(f"{config_path} must contain a JSON object")

    peft_type = str(data.get("peft_type", "LORA")).upper()
    if peft_type != "LORA":
        raise AdapterError(f"{config_path}: peft_type {peft_type!r} is not supported (need LORA)")
    if data.get("use_dora"):
        raise AdapterError(f"{config_path}: DoRA adapters are not supported")
    if data.get("modules_to_save"):
        raise AdapterError(
            f"{config_path}: modules_to_save={data['modules_to_save']} needs full replacement "
            "modules, which the stacked-slot layout cannot hold"
        )
    if data.get("target_parameters"):
        raise AdapterError(f"{config_path}: parameter-targeted LoRA is not supported")
    bias = str(data.get("bias", "none")).lower()
    if bias != "none":
        raise AdapterError(f"{config_path}: bias={bias!r} adapters are not supported (need 'none')")
    if data.get("fan_in_fan_out"):
        raise AdapterError(f"{config_path}: fan_in_fan_out adapters are not supported")
    return data


def _read_tensors(directory: Path) -> dict[str, torch.Tensor]:
    """Load ``adapter_model.safetensors``, refusing the pickle variant."""
    weights_path = directory / PEFT_WEIGHTS_FILE
    if not weights_path.is_file():
        if (directory / PEFT_PICKLE_FILE).is_file():
            raise AdapterError(
                f"{directory} stores its weights as {PEFT_PICKLE_FILE}, which is a pickle and is "
                "refused; re-save it with safetensors "
                "(PeftModel.save_pretrained(..., safe_serialization=True))"
            )
        raise AdapterError(f"{directory}: no {PEFT_WEIGHTS_FILE}")
    from safetensors.torch import load_file

    try:
        return load_file(str(weights_path), device="cpu")
    except OSError as exc:
        raise AdapterError(f"cannot read {weights_path}: {exc}") from exc


def load_peft_adapter(
    path: Path | str,
    *,
    name: str | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> LoRAAdapter:
    """Read a PEFT adapter directory into a :class:`LoRAAdapter`.

    ``name`` defaults to the directory's own name, which is what
    ``scripts/make_lora_adapters.py`` and the Helm adapter-sync init container both rely
    on: an object-store prefix of ``tenant-0/``, ``tenant-1/`` becomes the adapter names
    tenants address.

    ``dtype`` casts the factors on load. The registry passes the serving dtype so that the
    per-slot copy is a plain ``copy_`` with no conversion in the request path.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise AdapterError(f"{directory} is not a directory")
    config = _read_config(directory)
    tensors = _read_tensors(directory)

    base_rank = int(config.get("r", 0) or 0)
    base_alpha = float(config.get("lora_alpha", base_rank) or 0.0)
    use_rslora = bool(config.get("use_rslora", False))
    rank_pattern: Mapping[str, Any] = config.get("rank_pattern") or {}
    alpha_pattern: Mapping[str, Any] = config.get("alpha_pattern") or {}

    factors: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        parsed = peft_module_name(key)
        if parsed is None:
            logger.debug("ignoring non-LoRA tensor %s in %s", key, directory)
            continue
        module, side = parsed
        factors.setdefault(module, {})[side] = tensor

    weights: dict[str, LoRAWeights] = {}
    target = torch.device(device)
    for module, pair in sorted(factors.items()):
        if "A" not in pair or "B" not in pair:
            missing = "B" if "A" in pair else "A"
            raise AdapterError(f"{directory}: module {module} is missing its lora_{missing} factor")
        a = pair["A"].to(dtype=dtype or pair["A"].dtype, device=target)
        b = pair["B"].to(dtype=dtype or pair["B"].dtype, device=target)
        rank = int(a.shape[0])
        alpha = float(_pattern_lookup(alpha_pattern, module, base_alpha))
        weights[module] = LoRAWeights(
            module=module,
            a=a,
            b=b,
            scaling=adapter_scaling(rank, alpha, use_rslora=use_rslora),
        )
        pattern_rank = _pattern_lookup(rank_pattern, module, rank)
        if int(pattern_rank) != rank:
            raise AdapterError(
                f"{directory}: module {module} has rank {rank} on disk but rank_pattern says "
                f"{int(pattern_rank)}"
            )

    if not weights:
        raise AdapterError(f"{directory}: {PEFT_WEIGHTS_FILE} contains no LoRA factors")

    adapter = LoRAAdapter(
        name=name or directory.name,
        weights=weights,
        rank=max(weight.rank for weight in weights.values()),
        alpha=base_alpha,
        target_modules=tuple(sorted(str(m) for m in config.get("target_modules") or ())),
        base_model=str(config.get("base_model_name_or_path") or ""),
        path=directory,
        use_rslora=use_rslora,
    )
    logger.debug(
        "loaded adapter %s: rank %d over %d modules (%d bytes)",
        adapter.name,
        adapter.rank,
        len(adapter.weights),
        adapter.nbytes,
    )
    return adapter


def _pattern_lookup(pattern: Mapping[str, Any], module: str, default: float) -> float:
    """PEFT's per-module override lookup: exact key, or a dotted suffix match."""
    if module in pattern:
        return float(pattern[module])
    for key, value in pattern.items():
        if module.endswith(f".{key}") or module == key:
            return float(value)
    return float(default)


def discover_adapters(root: Path | str) -> list[Path]:
    """Every PEFT adapter directory directly under ``root``, sorted by name.

    Sorted rather than in directory order so that a benchmark taking "the first 16
    adapters" takes the same 16 on every machine.
    """
    base = Path(root)
    if not base.is_dir():
        raise AdapterError(f"{base} is not a directory")
    found = [child for child in sorted(base.iterdir()) if (child / PEFT_CONFIG_FILE).is_file()]
    if not found and (base / PEFT_CONFIG_FILE).is_file():
        return [base]
    return found


def linear_shapes(model: nn.Module) -> dict[str, tuple[int, int]]:
    """``{module path: (in_features, out_features)}`` for every projection in ``model``.

    Built from :func:`~turboserve.engine.model.layers.named_linears` so it sees exactly
    the modules a LoRA layer can wrap, including ones already wrapped.
    """
    from turboserve.engine.model.layers import named_linears

    return {
        name: (linear.in_features, linear.out_features) for name, linear in named_linears(model)
    }


def merged_state_dict(
    adapter: LoRAAdapter, base: Mapping[str, torch.Tensor], *, modules: Sequence[str] | None = None
) -> dict[str, torch.Tensor]:
    """``{module: W + scaling * B @ A}`` for the adapter's modules.

    The reference used by ``tests/unit/test_lora_layers.py`` to check the batched path
    against a merged model, and by nothing in the serving path.
    """
    wanted = set(modules) if modules is not None else set(adapter.weights)
    merged: dict[str, torch.Tensor] = {}
    for module, weight in adapter.weights.items():
        if module not in wanted:
            continue
        original = base.get(module)
        if original is None:
            raise AdapterError(f"no base weight for {module}")
        merged[module] = original.to(torch.float32) + weight.merged_delta()
    return merged
