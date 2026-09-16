"""Which adapters exist, which are on the GPU right now, and what that costs.

Serving 100 adapters does not mean holding 100 adapters in VRAM. The registry keeps every
registered adapter in host memory and treats the GPU-side stacked buffers of
:class:`~turboserve.engine.lora.layers.LoRALinear` as a small fixed cache: ``max_gpu_adapters``
slots, filled on demand, evicted least-recently-used. That is the design S-LoRA (Sheng et
al., 2023, https://arxiv.org/abs/2311.03285) calls unified paging for adapters, reduced here
to its essential part -- adapters are small and uniform, so a slot array beats a paged
allocator and keeps the kernels' indexing trivial.

Two identifiers, and keeping them apart is the whole point of this module:

``adapter id``
    Stable, assigned at :meth:`LoRARegistry.register`, carried by a request all the way
    into :attr:`Sequence.lora_id <turboserve.engine.core.sequence.Sequence.lora_id>`. It
    never changes while the process lives.
``slot``
    Where that adapter *currently* sits in the GPU buffers, or nothing at all. It changes
    every time the working set changes.

The engine's default context builder equates the two, which is only correct when residency
never moves. :meth:`LoRARegistry.build_context` is the replacement
``lora_ctx_builder``: it resolves each token's adapter id to a live slot, activating (and
if necessary evicting) first, so the tensors the layers read are guaranteed to hold the
adapter the token's tenant asked for.

Capacity, stated plainly because it is the one way to misconfigure this: a step can contain
at most one adapter per sequence, so setting ``max_gpu_adapters >= scheduler.max_num_seqs``
makes "more distinct adapters in a step than there are slots" impossible.
:func:`~turboserve.engine.lora.layers.install_lora` warns when the configuration does not
have that property, and a step that does overflow raises :class:`LoRACapacityError` rather
than quietly serving some tenant the base model.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from turboserve.engine.core.types import NO_LORA
from turboserve.engine.lora.adapter import (
    AdapterError,
    LoRAAdapter,
    linear_shapes,
    load_peft_adapter,
)
from turboserve.engine.lora.layers import LoRABatch, LoRALinear
from turboserve.engine.model.layers import LORA_TARGET_MODULES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence

    from turboserve.engine.core.scheduler import SchedulerOutput
    from turboserve.engine.core.types import LoRAContext

logger = logging.getLogger(__name__)

__all__ = [
    "LoRACapacityError",
    "LoRAOptions",
    "LoRARegistry",
    "LoRARegistryError",
    "RegistryStats",
    "SlotInfo",
    "UnknownAdapterError",
    "setup_lora",
]


class LoRARegistryError(RuntimeError):
    """The registry was asked for something it cannot do."""


class UnknownAdapterError(LoRARegistryError):
    """A request named an adapter id or name that was never registered.

    The gateway rejects unknown adapter *names* at admission
    (``AdapterNotFoundError``, 403), so reaching this in a running engine means an id was
    fabricated somewhere between the gateway and the scheduler.
    """


class LoRACapacityError(LoRARegistryError):
    """One step needs more distinct adapters than the GPU pool has slots.

    Not a transient condition: it means ``max_gpu_adapters`` is smaller than the number of
    adapters the scheduler is willing to put in one step. See the module docstring.
    """


@dataclass(slots=True)
class RegistryStats:
    """Counters worth putting next to the engine's own in a result file.

    ``hits``/``misses`` are per activation request, not per token: a step that uses three
    adapters, two of them already resident, records two hits and one miss.
    """

    registrations: int = 0
    activations: int = 0
    hits: int = 0
    misses: int = 0
    loads: int = 0
    evictions: int = 0
    contexts: int = 0
    tokens: int = 0
    adapter_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of adapter activations already resident; ``0.0`` before any."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def adapter_token_fraction(self) -> float:
        """Fraction of scheduled tokens that carried an adapter."""
        return self.adapter_tokens / self.tokens if self.tokens else 0.0

    def to_dict(self) -> dict[str, int | float]:
        """Flat, JSON-safe counters plus the two derived rates."""
        data: dict[str, int | float] = dict(asdict(self))
        data["hit_rate"] = self.hit_rate
        data["adapter_token_fraction"] = self.adapter_token_fraction
        return data


@dataclass(frozen=True, slots=True)
class SlotInfo:
    """What occupies one GPU slot."""

    slot: int
    adapter_id: int
    name: str
    rank: int
    pinned: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe description for ``stats()`` and the CLI."""
        return asdict(self)


class LoRARegistry:
    """Host-resident adapter store with an LRU of GPU slots.

    Not thread-safe and not re-entrant: it is owned by the engine's step loop, the single
    place that decides what the next step contains.
    """

    def __init__(
        self,
        max_gpu_adapters: int,
        model: nn.Module | None = None,
        *,
        max_lora_rank: int = 16,
        target_modules: Sequence[str] = LORA_TARGET_MODULES,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Size the GPU pool and decide what the adapters will be stored as.

        Args:
            max_gpu_adapters: number of slots, i.e. how many adapters may be resident at
                once. See the module docstring for the relationship to ``max_num_seqs``.
            model: the model the adapters will be applied to. Supplying it here makes
                registration validate shapes and gives the slots the model's dtype and
                device; it can also be supplied later through :meth:`bind`.
            max_lora_rank: the widest adapter the pool can hold. Registering a wider one
                is an error rather than a silent truncation.
            target_modules: projection names :func:`install_lora` wraps.
            dtype: slot storage dtype; defaults to the model's.
            device: slot storage device; defaults to the model's.
        """
        if max_gpu_adapters < 1:
            raise ValueError(f"max_gpu_adapters must be positive, got {max_gpu_adapters}")
        if max_lora_rank < 1:
            raise ValueError(f"max_lora_rank must be positive, got {max_lora_rank}")
        self._num_slots = int(max_gpu_adapters)
        self._max_rank = int(max_lora_rank)
        self._target_modules = tuple(target_modules)
        self._dtype = dtype
        self._device = None if device is None else torch.device(device)
        self._model: nn.Module | None = None
        self._shapes: dict[str, tuple[int, int]] = {}

        self._adapters: dict[int, LoRAAdapter] = {}
        self._ids_by_name: dict[str, int] = {}
        self._next_id = NO_LORA + 1

        self._slot_of: dict[int, int] = {}
        self._adapter_of: dict[int, int] = {}
        self._free_slots: list[int] = list(range(1, self._num_slots + 1))
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._pinned: set[int] = set()

        self._layers: dict[str, LoRALinear] = {}
        self.stats = RegistryStats()
        if model is not None:
            self.bind(model)

    # -- configuration ------------------------------------------------------------------

    @property
    def num_slots(self) -> int:
        """How many adapters may be GPU-resident at once."""
        return self._num_slots

    @property
    def max_lora_rank(self) -> int:
        """Widest adapter rank the stacked buffers can hold."""
        return self._max_rank

    @property
    def target_modules(self) -> tuple[str, ...]:
        """Projection names :func:`install_lora` wraps by default."""
        return self._target_modules

    @property
    def dtype(self) -> torch.dtype | None:
        """Storage dtype for the slots (``None`` = follow the base projection)."""
        return self._dtype

    @property
    def device(self) -> torch.device | None:
        """Storage device for the slots (``None`` = follow the base projection)."""
        return self._device

    @property
    def model(self) -> nn.Module | None:
        """The bound model, once :meth:`bind` has seen one."""
        return self._model

    def bind(self, model: nn.Module) -> None:
        """Record the model adapters are validated against, and adopt its dtype/device.

        Called by :func:`~turboserve.engine.lora.layers.install_lora`; calling it directly
        before registering adapters is what makes registration-time shape validation
        possible for a registry built without a model.
        """
        self._model = model
        self._shapes = linear_shapes(model)
        if self._dtype is None or self._device is None:
            reference = next((p for p in model.parameters()), None)
            if reference is not None:
                self._dtype = self._dtype or reference.dtype
                self._device = self._device or reference.device
        for adapter in self._adapters.values():
            adapter.validate_against(self._shapes)

    def attach(self, name: str, layer: LoRALinear) -> None:
        """Register one wrapped projection so slot loads reach it.

        Rejects a layer whose pool geometry differs from the registry's: two different
        slot counts in one model would make a slot index mean different things in
        different layers, which is the kind of bug that shows up as one tenant's adapter
        leaking into another's output for three of seven projections.
        """
        if layer.num_slots != self._num_slots or layer.max_rank != self._max_rank:
            raise LoRARegistryError(
                f"{name}: layer pool is {layer.num_slots} slots of rank {layer.max_rank}, "
                f"registry is {self._num_slots} of rank {self._max_rank}"
            )
        self._layers[name] = layer

    @property
    def layers(self) -> Mapping[str, LoRALinear]:
        """The wrapped projections, by dotted module name."""
        return self._layers

    # -- registration -------------------------------------------------------------------

    def register(self, name: str, path: Path | str) -> int:
        """Load a PEFT adapter directory into host memory and give it a stable id."""
        adapter = load_peft_adapter(path, name=name, dtype=self._dtype, device="cpu")
        return self.register_adapter(adapter)

    def register_adapter(self, adapter: LoRAAdapter) -> int:
        """Register an already-loaded adapter; returns its stable id.

        Validation happens here and not at first use: an adapter whose rank exceeds the
        pool, or whose projections do not match the model, is a deployment error and must
        surface when the operator adds it, not when a tenant sends a request.
        """
        if adapter.name in self._ids_by_name:
            raise LoRARegistryError(f"adapter {adapter.name!r} is already registered")
        if adapter.rank > self._max_rank:
            raise LoRARegistryError(
                f"adapter {adapter.name!r} has rank {adapter.rank}, pool max_lora_rank is "
                f"{self._max_rank}"
            )
        if self._shapes:
            adapter.validate_against(self._shapes)
        if self._layers:
            untargeted = [m for m in adapter.modules if m not in self._layers]
            if untargeted:
                raise LoRARegistryError(
                    f"adapter {adapter.name!r} targets {untargeted[:3]}, which LoRA is not "
                    "installed on; widen target_modules or retrain the adapter"
                )
        if self._dtype is not None:
            adapter = adapter.to(dtype=self._dtype)
        adapter_id = self._next_id
        self._next_id += 1
        self._adapters[adapter_id] = adapter
        self._ids_by_name[adapter.name] = adapter_id
        self.stats.registrations += 1
        logger.debug(
            "registered adapter %s as id %d (rank %d, %d bytes)",
            adapter.name,
            adapter_id,
            adapter.rank,
            adapter.nbytes,
        )
        return adapter_id

    def register_directory(self, root: Path | str, *, limit: int | None = None) -> dict[str, int]:
        """Register every PEFT adapter directory under ``root``, sorted by name.

        ``limit`` takes the first ``n``, which is how the benchmark asks for "the first 16
        of these 64 adapters" and gets the same 16 on every machine.
        """
        from turboserve.engine.lora.adapter import discover_adapters

        found = discover_adapters(root)
        if limit is not None:
            if limit < 1:
                raise ValueError(f"limit must be positive, got {limit}")
            if len(found) < limit:
                raise AdapterError(f"{root} holds {len(found)} adapters, {limit} were requested")
            found = found[:limit]
        return {path.name: self.register(path.name, path) for path in found}

    def __len__(self) -> int:
        """Number of registered adapters."""
        return len(self._adapters)

    def __contains__(self, key: str | int) -> bool:
        """Whether an adapter name or id is registered."""
        return key in self._ids_by_name if isinstance(key, str) else key in self._adapters

    def names(self) -> list[str]:
        """Registered adapter names, in registration order."""
        return [adapter.name for adapter in self._adapters.values()]

    def ids(self) -> list[int]:
        """Registered adapter ids, in registration order."""
        return list(self._adapters)

    def name_to_id(self) -> dict[str, int]:
        """The mapping a gateway backend's ``adapters`` option needs."""
        return dict(self._ids_by_name)

    def id_for(self, name: str) -> int:
        """Stable id of a registered adapter name."""
        try:
            return self._ids_by_name[name]
        except KeyError as exc:
            raise UnknownAdapterError(f"no adapter named {name!r}") from exc

    def name_for(self, adapter_id: int) -> str:
        """Name of a registered adapter id."""
        return self.adapter(adapter_id).name

    def adapter(self, key: str | int) -> LoRAAdapter:
        """The host-resident adapter for a name or id."""
        adapter_id = self.id_for(key) if isinstance(key, str) else int(key)
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise UnknownAdapterError(f"no adapter with id {adapter_id}") from exc

    # -- residency ----------------------------------------------------------------------

    @property
    def num_resident(self) -> int:
        """Adapters currently occupying a GPU slot."""
        return len(self._slot_of)

    @property
    def num_free_slots(self) -> int:
        """Slots holding no adapter."""
        return len(self._free_slots)

    def slot_for(self, key: str | int) -> int | None:
        """The slot an adapter occupies, or ``None`` when it is not resident."""
        adapter_id = self.id_for(key) if isinstance(key, str) else int(key)
        return self._slot_of.get(adapter_id)

    def resident_ids(self) -> list[int]:
        """Resident adapter ids, least recently used first."""
        return list(self._lru)

    def slots(self) -> list[SlotInfo]:
        """A description of every occupied slot, ordered by slot index."""
        return [
            SlotInfo(
                slot=slot,
                adapter_id=adapter_id,
                name=self._adapters[adapter_id].name,
                rank=self._adapters[adapter_id].rank,
                pinned=adapter_id in self._pinned,
            )
            for slot, adapter_id in sorted(self._adapter_of.items())
        ]

    def pin(self, key: str | int) -> int:
        """Make an adapter resident and exempt from eviction; returns its slot.

        Pinning is for the adapters a deployment always serves (a default persona, the
        house model): without it a burst of one-off tenants can evict them and make every
        subsequent request pay a slot load.
        """
        adapter_id = self.id_for(key) if isinstance(key, str) else int(key)
        slot = self.activate([adapter_id])[adapter_id]
        self._pinned.add(adapter_id)
        return slot

    def unpin(self, key: str | int) -> bool:
        """Allow an adapter to be evicted again; ``False`` if it was not pinned."""
        adapter_id = self.id_for(key) if isinstance(key, str) else int(key)
        if adapter_id not in self._pinned:
            return False
        self._pinned.discard(adapter_id)
        return True

    def is_pinned(self, key: str | int) -> bool:
        """Whether an adapter is exempt from eviction."""
        adapter_id = self.id_for(key) if isinstance(key, str) else int(key)
        return adapter_id in self._pinned

    def activate(self, ids: Iterable[int]) -> dict[int, int]:
        """Make every adapter in ``ids`` resident, evicting LRU adapters as needed.

        Returns ``{adapter id: slot}`` for exactly the requested ids. Adapters requested in
        the same call are never each other's eviction victims, so a step whose working set
        fits in the pool always succeeds regardless of the order the ids arrive in.
        """
        wanted = sorted({int(i) for i in ids if int(i) != NO_LORA})
        if not wanted:
            return {}
        unknown = [i for i in wanted if i not in self._adapters]
        if unknown:
            raise UnknownAdapterError(f"adapter ids {unknown} are not registered")
        if len(wanted) > self._num_slots:
            raise LoRACapacityError(
                f"{len(wanted)} distinct adapters in one step but the GPU pool has "
                f"{self._num_slots} slots; raise max_gpu_adapters (at least to the "
                "scheduler's max_num_seqs) or lower max_num_seqs"
            )
        self.stats.activations += 1

        protected = set(wanted)
        mapping: dict[int, int] = {}
        misses: list[int] = []
        for adapter_id in wanted:
            slot = self._slot_of.get(adapter_id)
            if slot is None:
                misses.append(adapter_id)
                continue
            mapping[adapter_id] = slot
            self._lru.move_to_end(adapter_id)
            self.stats.hits += 1

        for adapter_id in misses:
            self.stats.misses += 1
            slot = self._take_slot(protected)
            self._load(adapter_id, slot)
            mapping[adapter_id] = slot
        return mapping

    def activate_names(self, names: Iterable[str]) -> dict[str, int]:
        """:meth:`activate` addressed by adapter name; returns ``{name: slot}``."""
        ids = {name: self.id_for(name) for name in names}
        slots = self.activate(ids.values())
        return {name: slots[adapter_id] for name, adapter_id in ids.items()}

    def deactivate(self, key: str | int) -> bool:
        """Evict an adapter from its slot; ``False`` when it was not resident.

        Refuses to evict a pinned adapter -- unpin it first, so that "this adapter is
        always warm" cannot be undone by a stray call.
        """
        adapter_id = self.id_for(key) if isinstance(key, str) else int(key)
        if adapter_id in self._pinned:
            raise LoRARegistryError(f"adapter {self.name_for(adapter_id)!r} is pinned")
        slot = self._slot_of.get(adapter_id)
        if slot is None:
            return False
        self._evict_slot(slot)
        return True

    def _take_slot(self, protected: set[int]) -> int:
        """A free slot, evicting the least recently used unprotected adapter if needed."""
        if self._free_slots:
            return self._free_slots.pop(0)
        for adapter_id in list(self._lru):
            if adapter_id in protected or adapter_id in self._pinned:
                continue
            self._evict_slot(self._slot_of[adapter_id])
            return self._free_slots.pop(0)
        raise LoRACapacityError(
            f"every one of the {self._num_slots} slots is pinned or in use by this step; "
            f"{len(self._pinned)} adapters are pinned"
        )

    def _evict_slot(self, slot: int) -> None:
        """Release a slot and forget what was in it (the layers keep the stale bytes)."""
        adapter_id = self._adapter_of.pop(slot)
        self._slot_of.pop(adapter_id, None)
        self._lru.pop(adapter_id, None)
        self._free_slots.append(slot)
        self.stats.evictions += 1
        logger.debug("evicted adapter %s from slot %d", self._adapters[adapter_id].name, slot)

    def _load(self, adapter_id: int, slot: int) -> None:
        """Copy one adapter's factors into every wrapped projection's ``slot``."""
        adapter = self._adapters[adapter_id]
        for name, layer in self._layers.items():
            weights = adapter.weights.get(name)
            if weights is None:
                layer.clear_slot(slot)
                continue
            moved = weights.to(dtype=layer.lora_a.dtype, device=layer.lora_a.device)
            layer.load_slot(slot, moved.a, moved.b, moved.scaling)
        self._slot_of[adapter_id] = slot
        self._adapter_of[slot] = adapter_id
        self._lru[adapter_id] = None
        self._lru.move_to_end(adapter_id)
        self.stats.loads += 1
        logger.debug("loaded adapter %s into slot %d", adapter.name, slot)

    def reset_slots(self) -> None:
        """Evict everything (pinned included) and zero the GPU buffers."""
        for layer in self._layers.values():
            for slot in range(1, self._num_slots + 1):
                layer.clear_slot(slot)
        self._slot_of.clear()
        self._adapter_of.clear()
        self._lru.clear()
        self._pinned.clear()
        self._free_slots = list(range(1, self._num_slots + 1))

    # -- the engine hook ----------------------------------------------------------------

    def build_context(
        self, out: SchedulerOutput, device: torch.device | str = "cpu"
    ) -> LoRAContext | None:
        """The ``lora_ctx_builder`` the engine calls once per step.

        Reads each scheduled sequence's adapter *id*, ensures every one of them is resident
        (this is the only place activation happens on the serving path), and returns a
        :class:`~turboserve.engine.lora.layers.LoRABatch` whose per-token slots are the live
        ones. ``None`` means "no adapter in this step", which makes every wrapped projection
        a plain ``F.linear``.
        """
        ids = out.token_lora_ids()
        self.stats.contexts += 1
        self.stats.tokens += len(ids)
        distinct = {i for i in ids if i != NO_LORA}
        if not distinct:
            return None
        mapping = self.activate(distinct)
        slots = [mapping[i] if i != NO_LORA else NO_LORA for i in ids]
        self.stats.adapter_tokens += sum(1 for slot in slots if slot != NO_LORA)
        return LoRABatch.from_token_slots(slots, device=device)

    # -- accounting ---------------------------------------------------------------------

    @property
    def bytes_per_slot(self) -> int:
        """GPU bytes one slot costs across every wrapped projection."""
        return sum(layer.bytes_per_slot for layer in self._layers.values())

    @property
    def bytes_reserved(self) -> int:
        """GPU bytes the whole adapter pool occupies, empty slots included.

        This is the honest number for "what do adapters cost me": the buffers are
        preallocated, so an idle slot costs exactly as much as a busy one.
        """
        return sum(layer.bytes_reserved for layer in self._layers.values())

    @property
    def bytes_resident(self) -> int:
        """GPU bytes of the slots that currently hold an adapter."""
        return self.num_resident * self.bytes_per_slot

    @property
    def bytes_host(self) -> int:
        """Host bytes of every registered adapter (the CPU-resident store)."""
        return sum(adapter.nbytes for adapter in self._adapters.values())

    def base_weight_bytes(self) -> int:
        """Bytes of the bound model's parameters, counting tied storage once.

        The denominator of the VRAM comparison: serving ``N`` adapters by merging would
        need ``N`` copies of exactly this.
        """
        if self._model is None:
            raise LoRARegistryError("no model is bound; call bind(model) or install_lora first")
        seen: set[int] = set()
        total = 0
        for param in self._model.parameters():
            storage = param.data_ptr()
            if storage in seen:
                continue
            seen.add(storage)
            total += param.numel() * param.element_size()
        return total

    def vram_report(
        self, *, num_adapters: int | None = None, base_bytes: int | None = None
    ) -> dict[str, int | float]:
        """What the adapters cost against the merged-copies alternative.

        ``merged_bytes`` is what ``num_adapters`` separately merged models would occupy
        (``N * base``); ``lora_bytes`` is what this process occupies for the same coverage
        (one base plus the whole preallocated pool). ``saved_pct`` is the reduction, and is
        a property of the configuration -- it is arithmetic over measured tensor sizes, not
        a benchmark result.
        """
        count = len(self._adapters) if num_adapters is None else int(num_adapters)
        if count < 1:
            raise ValueError(f"num_adapters must be positive, got {count}")
        base = self.base_weight_bytes() if base_bytes is None else int(base_bytes)
        merged = base * count
        lora = base + self.bytes_reserved
        return {
            "num_adapters": count,
            "num_slots": self._num_slots,
            "base_bytes": base,
            "adapter_pool_bytes": self.bytes_reserved,
            "bytes_per_slot": self.bytes_per_slot,
            "resident_bytes": self.bytes_resident,
            "host_adapter_bytes": self.bytes_host,
            "merged_bytes": merged,
            "lora_bytes": lora,
            "saved_bytes": merged - lora,
            "saved_pct": 100.0 * (merged - lora) / merged if merged else 0.0,
        }

    def stats_dict(self) -> dict[str, int | float]:
        """Flat counters and sizes, prefixed ``lora_``, for the engine's ``stats()``."""
        data: dict[str, int | float] = {
            f"lora_{key}": value for key, value in self.stats.to_dict().items()
        }
        data.update(
            {
                "lora_num_adapters": len(self._adapters),
                "lora_num_slots": self._num_slots,
                "lora_num_resident": self.num_resident,
                "lora_num_pinned": len(self._pinned),
                "lora_max_rank": self._max_rank,
                "lora_bytes_per_slot": self.bytes_per_slot,
                "lora_bytes_reserved": self.bytes_reserved,
                "lora_bytes_resident": self.bytes_resident,
                "lora_bytes_host": self.bytes_host,
            }
        )
        return data

    def check_invariants(self) -> None:
        """Assert the slot bookkeeping is self-consistent (tests and debugging)."""
        if len(self._slot_of) != len(self._adapter_of):
            raise LoRARegistryError("slot maps disagree in size")
        for adapter_id, slot in self._slot_of.items():
            if self._adapter_of.get(slot) != adapter_id:
                raise LoRARegistryError(f"slot {slot} does not point back at adapter {adapter_id}")
        occupied = set(self._adapter_of)
        free = set(self._free_slots)
        if occupied & free:
            raise LoRARegistryError(f"slots {sorted(occupied & free)} are both free and occupied")
        if occupied | free != set(range(1, self._num_slots + 1)):
            raise LoRARegistryError("slot accounting does not cover the pool")
        if len(self._free_slots) != len(free):
            raise LoRARegistryError("the free list holds a duplicate")
        if set(self._lru) != set(self._slot_of):
            raise LoRARegistryError("the LRU and the residency map disagree")
        if not self._pinned <= set(self._slot_of):
            raise LoRARegistryError("a pinned adapter is not resident")

    def __repr__(self) -> str:
        return (
            f"LoRARegistry(adapters={len(self._adapters)}, slots={self._num_slots}, "
            f"resident={self.num_resident}, max_rank={self._max_rank})"
        )


class LoRAOptions(BaseModel):
    """The validated shape of :attr:`EngineConfig.lora`.

    :class:`~turboserve.engine.core.types.EngineConfig` carries ``lora`` as an untyped dict
    precisely so that this module can define and validate its own options without editing
    the shared config file. ``extra="forbid"`` means a typo in a deployment's YAML is a
    startup error rather than a silently ignored setting.
    """

    model_config = ConfigDict(extra="forbid")

    max_loras: int = Field(default=8, ge=1)
    """GPU slots, i.e. how many adapters may be resident at once."""

    max_lora_rank: int = Field(default=16, ge=1)
    """Widest adapter rank the slots can hold."""

    target_modules: list[str] = Field(default_factory=lambda: list(LORA_TARGET_MODULES))
    """Projection names to wrap."""

    adapters: dict[str, str] = Field(default_factory=dict)
    """Explicit ``name -> directory`` adapters, registered in the order given."""

    adapters_dir: str | None = None
    """Directory whose immediate subdirectories are each a PEFT adapter."""

    pinned: list[str] = Field(default_factory=list)
    """Adapter names to keep resident regardless of use."""

    @model_validator(mode="after")
    def _check(self) -> LoRAOptions:
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if not self.adapters and self.adapters_dir is None:
            raise ValueError("configure adapters or adapters_dir; a LoRA engine with none is idle")
        return self

    @classmethod
    def from_engine_config(cls, config: Any) -> LoRAOptions | None:
        """Parse ``config.lora``; ``None`` when the engine is configured without adapters."""
        raw = getattr(config, "lora", None)
        if raw is None:
            return None
        return cls.model_validate(raw)


def setup_lora(engine: Any, options: LoRAOptions | Mapping[str, Any]) -> LoRARegistry:
    """Build a registry from ``options``, register its adapters and install it on ``engine``.

    The one call ``turboserve serve --engine reference`` and the multi-adapter benchmark
    both use: it is the only place that knows the order of operations (build the pool,
    wrap the projections, register adapters against the now-known shapes, pin what should
    stay warm), and getting that order wrong produces confusing errors rather than wrong
    answers.
    """
    from turboserve.engine.lora.layers import install_lora

    resolved = options if isinstance(options, LoRAOptions) else LoRAOptions.model_validate(options)
    registry = LoRARegistry(
        resolved.max_loras,
        max_lora_rank=resolved.max_lora_rank,
        target_modules=tuple(resolved.target_modules),
    )
    install_lora(engine, registry)
    for name, path in resolved.adapters.items():
        registry.register(name, path)
    if resolved.adapters_dir is not None:
        registry.register_directory(resolved.adapters_dir)
    for name in resolved.pinned:
        registry.pin(name)
    logger.info(
        "LoRA ready: %d adapters registered, %d slots of rank %d (%d bytes reserved)",
        len(registry),
        registry.num_slots,
        registry.max_lora_rank,
        registry.bytes_reserved,
    )
    return registry
