"""The batched LoRA projection: one base GEMM plus one grouped delta per active adapter.

The serving requirement that shapes this file is that a step may mix adapters. Continuous
batching only works if the scheduler is free to put whichever sequences are runnable into
the same step; forcing adapter-homogeneous batches would mean a tenant's request waits for
other requests using the same adapter, which is the queueing behaviour multi-tenant serving
exists to avoid. So the batch carries a per-token slot vector
(:class:`~turboserve.engine.core.types.LoRAContext`) and the projection applies a different
low-rank update to different rows of the same input.

Two ways to do that, both implemented:

**SGMV (the default).** Segmented Gather Matrix-Vector, the grouping S-LoRA (Sheng et al.,
2023, https://arxiv.org/abs/2311.03285) and Punica (Chen et al., 2023,
https://arxiv.org/abs/2310.18547) describe: sort the token indices by slot, and run one
pair of ``[n_slot, in] x [in, r]`` and ``[n_slot, r] x [r, out]`` GEMMs per active slot,
scattering the results back with ``index_add_``. The cost is one pair of dense GEMMs per
*active adapter*, not per token and not per registered adapter, and it is pure PyTorch, so
it runs identically on CPU (where the tests check it) and on any GPU.

**BGMV (decode only, CUDA).** When the step is a handful of tokens -- decode, or a
speculative verification batch -- those per-slot GEMMs degenerate into matrix-vector
products and the launch overhead dominates. :mod:`turboserve.engine.lora.triton_bgmv` then
does the whole batch in two kernels.

The grouping itself is computed **once per step, on the host**, by
:class:`LoRABatch`. This is the detail that decides whether multi-LoRA is usable at all: a
28-layer model has 196 wrapped projections, and deriving the segments inside each of them
from the device-side slot tensor would be 196 device-to-host synchronisations per decoded
token. The scheduler already knows every token's adapter as Python ints, so the sort is a
few microseconds of ``list.sort`` and one upload.

``LoRALinear`` deliberately subclasses
:class:`~turboserve.engine.model.layers.LinearBase` and *adopts* the base projection's
parameter objects rather than holding it as a child module. If it held a child, the base
weight's name would change from ``model.layers.0.self_attn.q_proj.weight`` to
``...q_proj.base.weight`` and the checkpoint loader would no longer find it -- adapters
must be installable before *or* after the weights are loaded, and both orders are used
(the factory path during construction, ``install_lora`` for an engine that is already up).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from turboserve.engine.core.types import NO_LORA, LoRAContext
from turboserve.engine.lora.triton_bgmv import bgmv_delta, can_use_bgmv
from turboserve.engine.model.layers import LORA_TARGET_MODULES, LinearBase, named_linears

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator, Sequence

    from turboserve.engine.lora.registry import LoRARegistry

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BGMV_MAX_TOKENS",
    "LoRABatch",
    "LoRALinear",
    "install_lora",
    "iter_lora_linears",
    "lora_linear_factory",
    "lora_segments",
    "uninstall_lora",
]

#: Above this many tokens in a step the grouped GEMM path is used even on CUDA: BGMV is a
#: matrix-vector kernel and a long prefill chunk has enough work per adapter to amortise a
#: pair of real GEMMs. It is an attribute of every :class:`LoRALinear`, so a deployment can
#: move it without editing this file.
DEFAULT_BGMV_MAX_TOKENS = 256


@dataclass(slots=True)
class LoRABatch(LoRAContext):
    """A :class:`LoRAContext` that also carries the step's SGMV grouping.

    ``order`` lists the positions of the adapter-using tokens, sorted by slot; ``segments``
    gives ``(slot, start, end)`` half-open ranges into ``order``. Base-model tokens appear
    in neither, so a batch that is mostly base traffic costs the LoRA layers nothing beyond
    the (already required) base GEMM.

    Every consumer accepts a plain ``LoRAContext`` too -- :func:`lora_segments` derives the
    same grouping from the device tensor when it has to. That path is correct and
    synchronising, and exists so that a caller which builds contexts itself (the engine's
    default builder, a test) still works.
    """

    order: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), compare=False
    )
    """``[num_adapter_tokens]`` int64 token positions, grouped by slot."""

    segments: tuple[tuple[int, int, int], ...] = ()
    """``(slot, start, end)`` per active slot, indexing :attr:`order`."""

    @classmethod
    def from_token_slots(
        cls, slots: Sequence[int], *, device: torch.device | str = "cpu"
    ) -> LoRABatch:
        """Build the context and its grouping from the step's per-token slot list.

        ``slots`` is a host list -- the scheduler produces it as Python ints -- so the sort
        and the segment boundaries are computed without reading anything back from the
        device.
        """
        target = torch.device(device)
        buckets: dict[int, list[int]] = {}
        for position, slot in enumerate(slots):
            if slot == NO_LORA:
                continue
            buckets.setdefault(int(slot), []).append(position)

        order: list[int] = []
        segments: list[tuple[int, int, int]] = []
        for slot in sorted(buckets):
            positions = buckets[slot]
            start = len(order)
            order.extend(positions)
            segments.append((slot, start, len(order)))

        return cls(
            token_lora_slot=torch.tensor(list(slots), dtype=torch.long, device=target),
            active_slots=sorted(buckets),
            order=torch.tensor(order, dtype=torch.long, device=target),
            segments=tuple(segments),
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> LoRABatch:
        """Move both the slot vector and the grouping (``self`` when already there)."""
        target = torch.device(device)
        if self.token_lora_slot.device == target and self.order.device == target:
            return self
        return replace(
            self,
            token_lora_slot=self.token_lora_slot.to(target, non_blocking=non_blocking),
            order=self.order.to(target, non_blocking=non_blocking),
        )

    def validate(self) -> None:
        """Check the context invariants plus the grouping's consistency with them."""
        # Explicit base call rather than ``super()``: ``@dataclass(slots=True)`` returns a
        # *new* class object, which leaves the zero-argument ``super()`` closure cell
        # pointing at the undecorated class and makes it raise at runtime.
        LoRAContext.validate(self)
        if self.order.dim() != 1 or self.order.dtype != torch.long:
            raise ValueError(
                f"order must be a 1-D int64 tensor, got shape {tuple(self.order.shape)} "
                f"dtype {self.order.dtype}"
            )
        covered = sum(end - start for _, start, end in self.segments)
        if covered != int(self.order.shape[0]):
            raise ValueError(
                f"segments cover {covered} positions but order holds {int(self.order.shape[0])}"
            )
        if [slot for slot, _, _ in self.segments] != self.active_slots:
            raise ValueError(
                f"segments name slots {[s for s, _, _ in self.segments]} but active_slots is "
                f"{self.active_slots}"
            )


def lora_segments(ctx: LoRAContext) -> tuple[torch.Tensor, tuple[tuple[int, int, int], ...]]:
    """Return ``(order, segments)`` for ``ctx``, deriving them if they are not carried.

    The derivation reads ``token_lora_slot`` back to the host, which synchronises with the
    device. That is why :class:`LoRABatch` exists and why the registry's context builder
    produces one: on the serving path this fallback must not be reached.
    """
    if isinstance(ctx, LoRABatch):
        return ctx.order, ctx.segments
    slots = [int(slot) for slot in ctx.token_lora_slot.tolist()]
    batch = LoRABatch.from_token_slots(slots, device=ctx.token_lora_slot.device)
    logger.debug(
        "derived LoRA segments on the host for a plain LoRAContext (%d tokens)", len(slots)
    )
    return batch.order, batch.segments


class LoRALinear(LinearBase):
    """A projection with ``num_slots`` stacked adapters applied per token.

    Holds ``A[num_slots, max_rank, in]``, ``B[num_slots, out, max_rank]`` and
    ``scaling[num_slots]``, all preallocated: the slots are a fixed GPU budget the registry
    fills and evicts, so serving never allocates in the request path and the VRAM cost of
    "N adapters resident" is a number known before the first request.

    Slot indices are 1-based in the batch (``0`` is
    :data:`~turboserve.engine.core.types.NO_LORA`, the base model), and row ``slot - 1``
    holds that slot's factors. Storing a dead row 0 would be simpler by one subtraction and
    would waste ``1/num_slots`` of the adapter pool, which is real VRAM in exactly the
    configuration this module exists to make cheap.

    The stacked tensors are registered as **non-persistent** buffers: they are runtime slot
    storage, not part of the model's checkpoint, and a persistent buffer would make
    ``load_weights(strict=True)`` report them missing from every checkpoint.
    """

    lora_a: torch.Tensor
    lora_b: torch.Tensor
    lora_scaling: torch.Tensor

    def __init__(
        self,
        base: LinearBase,
        *,
        num_slots: int,
        max_rank: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        bgmv_max_tokens: int = DEFAULT_BGMV_MAX_TOKENS,
        prefer_triton: bool = True,
    ) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        if max_rank < 1:
            raise ValueError(f"max_rank must be positive, got {max_rank}")
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.name = base.name
        # Adopt the base projection's parameters rather than nesting it, so the parameter
        # path a checkpoint uses is unchanged by wrapping. See the module docstring.
        self.weight = base.weight
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = base.bias

        self.num_slots = num_slots
        self.max_rank = max_rank
        self.bgmv_max_tokens = bgmv_max_tokens
        self.prefer_triton = prefer_triton
        weight_dtype = dtype if dtype is not None else self.weight.dtype
        weight_device = torch.device(device) if device is not None else self.weight.device
        self.register_buffer(
            "lora_a",
            torch.zeros(
                (num_slots, max_rank, self.in_features), dtype=weight_dtype, device=weight_device
            ),
            persistent=False,
        )
        self.register_buffer(
            "lora_b",
            torch.zeros(
                (num_slots, self.out_features, max_rank), dtype=weight_dtype, device=weight_device
            ),
            persistent=False,
        )
        self.register_buffer(
            "lora_scaling",
            torch.zeros((num_slots,), dtype=torch.float32, device=weight_device),
            persistent=False,
        )
        self._slot_rank: list[int] = [0] * num_slots
        self._slot_scaling: list[float] = [0.0] * num_slots

    # -- slot management ----------------------------------------------------------------

    @property
    def bytes_per_slot(self) -> int:
        """Bytes one slot occupies in this projection's stacked buffers."""
        element = self.lora_a.element_size()
        return (self.max_rank * self.in_features + self.out_features * self.max_rank) * element

    @property
    def bytes_reserved(self) -> int:
        """Bytes the whole adapter pool of this projection occupies, slots included."""
        return self.num_slots * self.bytes_per_slot + self.lora_scaling.numel() * 4

    def slot_rank(self, slot: int) -> int:
        """Rank currently loaded in ``slot`` (``0`` when the slot is empty)."""
        return self._slot_rank[self._row(slot)]

    def _row(self, slot: int) -> int:
        """Storage row of a 1-based batch slot, with a real error for a bad one."""
        if not 1 <= slot <= self.num_slots:
            raise ValueError(
                f"{self.name or type(self).__name__}: slot {slot} outside 1..{self.num_slots}"
            )
        return slot - 1

    @torch.no_grad()
    def load_slot(self, slot: int, a: torch.Tensor, b: torch.Tensor, scaling: float) -> None:
        """Copy one adapter's factors for this projection into ``slot``.

        Columns beyond the adapter's rank are zeroed rather than left stale, so a rank-8
        adapter loaded over a rank-16 one cannot pick up the evicted tenant's numbers
        through the padded region that the BGMV kernel reads unconditionally.
        """
        row = self._row(slot)
        if a.shape[1] != self.in_features or b.shape[0] != self.out_features:
            raise ValueError(
                f"{self.name}: adapter factors are {a.shape[1]}->{b.shape[0]}, projection is "
                f"{self.in_features}->{self.out_features}"
            )
        rank = int(a.shape[0])
        if rank != int(b.shape[1]):
            raise ValueError(f"{self.name}: A rank {rank} != B rank {int(b.shape[1])}")
        if rank > self.max_rank:
            raise ValueError(
                f"{self.name}: adapter rank {rank} exceeds the pool's max_rank {self.max_rank}"
            )
        self.lora_a[row, :rank].copy_(a)
        self.lora_a[row, rank:].zero_()
        self.lora_b[row, :, :rank].copy_(b)
        self.lora_b[row, :, rank:].zero_()
        self.lora_scaling[row] = scaling
        self._slot_rank[row] = rank
        self._slot_scaling[row] = float(scaling)

    @torch.no_grad()
    def clear_slot(self, slot: int) -> None:
        """Zero a slot so an adapter that does not target this projection is a no-op."""
        row = self._row(slot)
        self.lora_a[row].zero_()
        self.lora_b[row].zero_()
        self.lora_scaling[row] = 0.0
        self._slot_rank[row] = 0
        self._slot_scaling[row] = 0.0

    # -- forward ------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, lora_ctx: LoRAContext | None = None) -> torch.Tensor:
        """Base projection plus the per-token adapter delta.

        With no context, or a context in which every token uses the base model, this is
        exactly ``LinearBase.forward`` -- one ``F.linear`` and no adapter memory traffic.
        """
        out = F.linear(x, self.weight, self.bias)
        if lora_ctx is None or lora_ctx.is_base_only:
            return out
        flat_x = x.reshape(-1, self.in_features)
        flat_out = out.reshape(-1, self.out_features)
        if lora_ctx.num_tokens != flat_x.shape[0]:
            raise ValueError(
                f"{self.name}: LoRA context describes {lora_ctx.num_tokens} tokens but the batch "
                f"has {flat_x.shape[0]}"
            )
        self._add_delta(flat_x, flat_out, lora_ctx)
        return out

    def _add_delta(self, x: torch.Tensor, out: torch.Tensor, lora_ctx: LoRAContext) -> None:
        """Accumulate the adapter deltas into ``out`` (a flat view of the result)."""
        max_slot = max(lora_ctx.active_slots)
        if max_slot > self.num_slots:
            raise ValueError(
                f"{self.name}: batch uses slot {max_slot} but the pool has {self.num_slots} slots"
            )
        contiguous = x if x.is_contiguous() else x.contiguous()
        if self.prefer_triton and can_use_bgmv(
            contiguous, rank=self.max_rank, max_tokens=self.bgmv_max_tokens
        ):
            bgmv_delta(
                contiguous,
                out,
                self.lora_a,
                self.lora_b,
                self.lora_scaling,
                lora_ctx.token_lora_slot,
                rank=self.max_rank,
            )
            return
        order, segments = lora_segments(lora_ctx)
        for slot, start, end in segments:
            if end <= start:
                continue
            row = slot - 1
            rank = self._slot_rank[row]
            if rank == 0:
                continue
            index = order[start:end]
            rows = x.index_select(0, index)
            shrunk = F.linear(rows, self.lora_a[row, :rank])
            delta = F.linear(shrunk, self.lora_b[row, :, :rank])
            out.index_add_(0, index, delta.mul_(self._slot_scaling[row]).to(out.dtype))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, name={self.name!r}, "
            f"num_slots={self.num_slots}, max_rank={self.max_rank}"
        )


def _matches(name: str, target_modules: Sequence[str]) -> bool:
    """Whether a dotted module path ends in one of the targeted projection names."""
    leaf = name.rsplit(".", 1)[-1]
    return leaf in target_modules


def lora_linear_factory(
    *,
    num_slots: int,
    max_rank: int,
    target_modules: Sequence[str] = LORA_TARGET_MODULES,
    on_create: Callable[[str, LoRALinear], None] | None = None,
    bgmv_max_tokens: int = DEFAULT_BGMV_MAX_TOKENS,
    prefer_triton: bool = True,
) -> Callable[..., LinearBase]:
    """A factory for :func:`~turboserve.engine.model.layers.use_linear_factory`.

    Installing adapters at construction time (rather than swapping modules afterwards) is
    the path that matters when the model is built directly, because the wrapper exists
    before ``load_checkpoint`` runs and the adopted parameter is the one the loader fills.

    ``on_create`` is called with ``(dotted name, layer)`` for every wrapped projection; the
    registry passes its own ``attach`` so it learns the layers without a second walk.
    """

    def factory(
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        name: str = "",
    ) -> LinearBase:
        base = LinearBase(
            in_features, out_features, bias=bias, dtype=dtype, device=device, name=name
        )
        if not _matches(name, target_modules):
            return base
        layer = LoRALinear(
            base,
            num_slots=num_slots,
            max_rank=max_rank,
            dtype=dtype,
            device=device,
            bgmv_max_tokens=bgmv_max_tokens,
            prefer_triton=prefer_triton,
        )
        if on_create is not None:
            on_create(name, layer)
        return layer

    return factory


def _resolve_model(target: object) -> nn.Module:
    """The ``nn.Module`` to wrap, whether given a model or something owning one."""
    if isinstance(target, nn.Module):
        return target
    model = getattr(target, "model", None)
    if isinstance(model, nn.Module):
        return model
    raise TypeError(f"{type(target).__name__} is neither a module nor an object owning a .model")


def install_lora(
    target: object,
    registry: LoRARegistry,
    target_modules: Sequence[str] | None = None,
) -> list[str]:
    """Swap every targeted projection for a :class:`LoRALinear` and wire the engine hook.

    Args:
        target: an :class:`~turboserve.engine.runtime.engine.LLMEngine`, an
            :class:`~turboserve.engine.runtime.async_engine.AsyncLLMEngine`, or a bare
            model. Anything exposing ``.model`` is unwrapped until a module is found.
        registry: a :class:`~turboserve.engine.lora.registry.LoRARegistry`; supplies the
            slot budget and rank, learns the layers through ``attach``, and becomes the
            engine's ``lora_ctx_builder``.
        target_modules: projection names to wrap; defaults to the registry's own
            configuration (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``, ``gate_proj``,
            ``up_proj``, ``down_proj``).

    Returns the dotted names that were wrapped, sorted. Wrapping is idempotent: a
    projection that is already a :class:`LoRALinear` is re-attached to the registry and
    left alone, so installing twice does not stack two adapters' worth of buffers.

    Setting ``lora_ctx_builder`` on the engine is what makes the adapters take effect: the
    engine's default builder maps ``Sequence.lora_id`` straight to a slot, which is only
    correct when residency never changes. The registry's builder resolves each request's
    adapter id to whichever slot it currently occupies, activating it first.
    """
    from turboserve.engine.lora.registry import LoRARegistry

    if not isinstance(registry, LoRARegistry):
        raise TypeError(f"registry must be a LoRARegistry, got {type(registry).__name__}")
    model = _resolve_model(target)
    modules = tuple(target_modules if target_modules is not None else registry.target_modules)

    wrapped: list[str] = []
    for name, linear in list(named_linears(model)):
        if not _matches(name, modules):
            continue
        if isinstance(linear, LoRALinear):
            registry.attach(name, linear)
            wrapped.append(name)
            continue
        layer = LoRALinear(
            linear,
            num_slots=registry.num_slots,
            max_rank=registry.max_lora_rank,
            dtype=registry.dtype,
            device=registry.device,
        )
        _replace_child(model, name, layer)
        registry.attach(name, layer)
        wrapped.append(name)

    if not wrapped:
        raise ValueError(
            f"no projection in {type(model).__name__} matches target_modules {list(modules)}"
        )
    registry.bind(model)
    if not isinstance(target, nn.Module):
        _install_builder(target, registry)
    logger.info(
        "installed LoRA on %d projections (%d slots, max rank %d)",
        len(wrapped),
        registry.num_slots,
        registry.max_lora_rank,
    )
    return sorted(wrapped)


def _install_builder(target: object, registry: LoRARegistry) -> None:
    """Point an engine's ``lora_ctx_builder`` at the registry, unwrapping async engines."""
    holder = target
    for _ in range(4):
        if hasattr(holder, "lora_ctx_builder"):
            holder.lora_ctx_builder = registry.build_context
            _warn_on_slot_budget(holder, registry)
            return
        inner = getattr(holder, "engine", None)
        if inner is None:
            break
        holder = inner
    raise TypeError(
        f"{type(target).__name__} exposes no lora_ctx_builder; pass the LLMEngine itself"
    )


def _warn_on_slot_budget(engine: object, registry: LoRARegistry) -> None:
    """Warn when the pool is smaller than the widest step the scheduler may build.

    A step holds at most one adapter per sequence, so ``num_slots >= max_num_seqs`` makes
    :class:`~turboserve.engine.lora.registry.LoRACapacityError` unreachable. Falling below
    it is a legitimate memory trade -- adapters are usually shared across sequences -- but
    it is a trade the operator should make knowingly.
    """
    config = getattr(getattr(engine, "scheduler", None), "config", None)
    max_num_seqs = getattr(config, "max_num_seqs", None)
    if isinstance(max_num_seqs, int) and registry.num_slots < max_num_seqs:
        logger.warning(
            "LoRA pool has %d slots but the scheduler may run %d sequences per step; a step "
            "using more than %d distinct adapters will raise LoRACapacityError",
            registry.num_slots,
            max_num_seqs,
            registry.num_slots,
        )


def _replace_child(root: nn.Module, dotted: str, layer: nn.Module) -> None:
    """Rebind ``root.<dotted>`` to ``layer``."""
    parent_path, _, leaf = dotted.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, layer)


def uninstall_lora(target: object) -> list[str]:
    """Replace every :class:`LoRALinear` under ``target`` with a plain projection.

    Frees the stacked buffers (the adapter pool is the larger half of a multi-LoRA
    deployment's non-KV memory) and restores base-only serving. Returns the names restored.
    """
    model = _resolve_model(target)
    restored: list[str] = []
    for name, linear in list(named_linears(model)):
        if not isinstance(linear, LoRALinear):
            continue
        base = LinearBase(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
            name=linear.name,
        )
        base.weight = linear.weight
        if linear.bias is not None:
            base.bias = linear.bias
        _replace_child(model, name, base)
        restored.append(name)
    if restored and not isinstance(target, nn.Module) and hasattr(target, "lora_ctx_builder"):
        target.lora_ctx_builder = None
    return sorted(restored)


def iter_lora_linears(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    """Yield ``(dotted name, layer)`` for every installed :class:`LoRALinear`."""
    for name, linear in named_linears(model):
        if isinstance(linear, LoRALinear):
            yield name, linear
