"""Slot residency, eviction, pinning, VRAM accounting and the engine hook.

Most of this file runs against a deliberately small stand-in model (two "layers" of two
projections) with adapters built in memory: residency policy has nothing to do with
transformers, and testing it on a real checkpoint would only make the failures slower to
read. The last section uses the tiny Qwen2 checkpoint and real PEFT directories, because
``setup_lora`` and ``build_context`` are exactly the places where the policy meets the
engine.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from turboserve.engine.core.scheduler import Scheduler
from turboserve.engine.core.types import NO_LORA, SchedulerConfig
from turboserve.engine.lora.adapter import AdapterError, LoRAAdapter, LoRAWeights
from turboserve.engine.lora.layers import LoRABatch, LoRALinear, install_lora
from turboserve.engine.lora.registry import (
    LoRACapacityError,
    LoRAOptions,
    LoRARegistry,
    LoRARegistryError,
    UnknownAdapterError,
    setup_lora,
)
from turboserve.engine.model.layers import LinearBase

HIDDEN = 8
TARGETS = ("q_proj", "v_proj")
MODULES = tuple(f"layers.{index}.{name}" for index in range(2) for name in TARGETS)


class StandInModel(nn.Module):
    """Two blocks of three projections, of which only :data:`TARGETS` are wrapped.

    The third (``o_proj``) is there on purpose: a model always has projections LoRA is not
    installed on, and the registry must tell "the model does not have this module" apart
    from "the model has it but it is not wrapped".
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Module() for _ in range(2))
        for index, block in enumerate(self.layers):
            for name in (*TARGETS, "o_proj"):
                block.add_module(name, LinearBase(HIDDEN, HIDDEN, name=f"layers.{index}.{name}"))


def make_registry(num_slots: int, *, max_rank: int = 4) -> tuple[LoRARegistry, StandInModel]:
    """A registry installed on a stand-in model."""
    model = StandInModel()
    registry = LoRARegistry(num_slots, max_lora_rank=max_rank, target_modules=TARGETS)
    install_lora(model, registry)
    return registry, model


def make_adapter(
    name: str, *, rank: int = 2, seed: int = 0, modules: tuple[str, ...] = MODULES
) -> LoRAAdapter:
    """An in-memory adapter with reproducible factors for ``modules``."""
    generator = torch.Generator().manual_seed(seed)
    weights = {
        module: LoRAWeights(
            module=module,
            a=torch.randn(rank, HIDDEN, generator=generator) * 0.1,
            b=torch.randn(HIDDEN, rank, generator=generator) * 0.1,
            scaling=2.0,
        )
        for module in modules
    }
    return LoRAAdapter(
        name=name, weights=weights, rank=rank, alpha=2.0 * rank, target_modules=TARGETS
    )


def load_adapter_script() -> Any:
    """Return the synthetic-adapter trainer module.

    The trainer used to be a bare script under ``scripts/``; it now lives in the package as
    :mod:`turboserve.engine.lora.make_adapters` and is exposed as
    ``turboserve lora make-adapters``, so loading it is a plain import. The helper is kept
    so the fixtures below read the same as they did while it was a script.
    """
    return importlib.import_module("turboserve.engine.lora.make_adapters")


@pytest.fixture(scope="session")
def adapters_dir(tmp_path_factory: pytest.TempPathFactory, tiny_qwen2_path: Path) -> Path:
    """Three real PEFT adapters over the tiny Qwen2 checkpoint, trained for two steps."""
    module = load_adapter_script()
    out = tmp_path_factory.mktemp("adapters")
    module.make_adapters(
        str(tiny_qwen2_path),
        out,
        count=3,
        rank=4,
        alpha=64,
        steps=2,
        batch_size=2,
        learning_rate=0.05,
        max_length=32,
        device="cpu",
        local_files_only=True,
    )
    return out


# -- registration --------------------------------------------------------------------------


def test_ids_are_stable_and_never_collide_with_the_base_model() -> None:
    registry, _ = make_registry(2)
    first = registry.register_adapter(make_adapter("a"))
    second = registry.register_adapter(make_adapter("b", seed=1))
    assert first > NO_LORA and second > first
    assert registry.id_for("a") == first
    assert registry.name_for(second) == "b"
    assert registry.name_to_id() == {"a": first, "b": second}
    assert "a" in registry and first in registry and len(registry) == 2


def test_a_duplicate_name_is_refused() -> None:
    registry, _ = make_registry(2)
    registry.register_adapter(make_adapter("a"))
    with pytest.raises(LoRARegistryError, match="already registered"):
        registry.register_adapter(make_adapter("a", seed=9))


def test_an_adapter_wider_than_the_pool_is_refused_at_registration() -> None:
    registry, _ = make_registry(2, max_rank=2)
    with pytest.raises(LoRARegistryError, match="max_lora_rank"):
        registry.register_adapter(make_adapter("wide", rank=8))


def test_an_adapter_targeting_an_unwrapped_projection_is_refused() -> None:
    registry, _ = make_registry(2)
    stray = make_adapter("stray", modules=("layers.0.o_proj",))
    with pytest.raises(LoRARegistryError, match="LoRA is not installed"):
        registry.register_adapter(stray)


def test_unknown_names_and_ids_raise() -> None:
    registry, _ = make_registry(1)
    with pytest.raises(UnknownAdapterError):
        registry.id_for("nope")
    with pytest.raises(UnknownAdapterError):
        registry.adapter(42)


# -- residency -----------------------------------------------------------------------------


def test_activate_places_adapters_and_counts_hits_and_misses() -> None:
    registry, _ = make_registry(3)
    ids = [registry.register_adapter(make_adapter(f"a{i}", seed=i)) for i in range(3)]
    mapping = registry.activate(ids)
    assert sorted(mapping.values()) == [1, 2, 3]
    assert registry.stats.misses == 3 and registry.stats.hits == 0
    again = registry.activate(ids)
    assert again == mapping
    assert registry.stats.hits == 3
    assert registry.stats.loads == 3, "a hit must not reload the factors"
    registry.check_invariants()


def test_the_slot_really_holds_that_adapters_factors() -> None:
    registry, model = make_registry(2)
    adapter = make_adapter("a", rank=2, seed=3)
    slot = registry.activate([registry.register_adapter(adapter)])[1]
    layer = model.layers[0].q_proj
    assert isinstance(layer, LoRALinear)
    torch.testing.assert_close(layer.lora_a[slot - 1, :2], adapter.weights["layers.0.q_proj"].a)
    torch.testing.assert_close(layer.lora_b[slot - 1, :, :2], adapter.weights["layers.0.q_proj"].b)


def test_an_adapter_that_skips_a_projection_clears_that_slot() -> None:
    registry, model = make_registry(1)
    partial = make_adapter("partial", modules=("layers.0.q_proj",))
    slot = registry.activate([registry.register_adapter(partial)])[1]
    untouched = model.layers[1].v_proj
    assert isinstance(untouched, LoRALinear)
    assert untouched.slot_rank(slot) == 0
    assert float(untouched.lora_a[slot - 1].abs().max()) == 0.0


def test_eviction_takes_the_least_recently_used() -> None:
    registry, _ = make_registry(2)
    a, b, c = (registry.register_adapter(make_adapter(f"a{i}", seed=i)) for i in range(3))
    registry.activate([a])
    registry.activate([b])
    registry.activate([a])  # a becomes the most recent
    registry.activate([c])  # so b is the victim
    assert registry.slot_for(b) is None
    assert registry.slot_for(a) is not None
    assert registry.slot_for(c) is not None
    assert registry.stats.evictions == 1
    registry.check_invariants()


def test_adapters_requested_together_never_evict_each_other() -> None:
    registry, _ = make_registry(2)
    ids = [registry.register_adapter(make_adapter(f"a{i}", seed=i)) for i in range(2)]
    registry.activate([ids[0]])
    mapping = registry.activate(ids)
    assert len(set(mapping.values())) == 2
    registry.check_invariants()


def test_a_step_wider_than_the_pool_is_a_capacity_error() -> None:
    registry, _ = make_registry(2)
    ids = [registry.register_adapter(make_adapter(f"a{i}", seed=i)) for i in range(3)]
    with pytest.raises(LoRACapacityError, match="max_gpu_adapters"):
        registry.activate(ids)


def test_activating_an_unregistered_id_is_refused() -> None:
    registry, _ = make_registry(2)
    with pytest.raises(UnknownAdapterError, match="not registered"):
        registry.activate([7])


def test_the_base_model_is_never_a_slot() -> None:
    registry, _ = make_registry(1)
    assert registry.activate([NO_LORA]) == {}
    assert registry.num_resident == 0


def test_pinning_survives_pressure_and_blocks_eviction() -> None:
    registry, _ = make_registry(2)
    keep = registry.register_adapter(make_adapter("keep"))
    others = [registry.register_adapter(make_adapter(f"x{i}", seed=i + 10)) for i in range(3)]
    slot = registry.pin(keep)
    assert registry.is_pinned(keep)
    for adapter_id in others:
        registry.activate([adapter_id])
    assert registry.slot_for(keep) == slot
    with pytest.raises(LoRARegistryError, match="pinned"):
        registry.deactivate(keep)
    assert registry.unpin(keep) is True
    assert registry.unpin(keep) is False
    assert registry.deactivate(keep) is True
    assert registry.deactivate(keep) is False
    registry.check_invariants()


def test_a_fully_pinned_pool_reports_why_it_cannot_make_room() -> None:
    registry, _ = make_registry(1)
    first = registry.register_adapter(make_adapter("pinned"))
    second = registry.register_adapter(make_adapter("other", seed=2))
    registry.pin(first)
    with pytest.raises(LoRACapacityError, match="pinned"):
        registry.activate([second])


def test_activate_names_and_slot_descriptions() -> None:
    registry, _ = make_registry(2)
    registry.register_adapter(make_adapter("alpha"))
    registry.register_adapter(make_adapter("beta", seed=4))
    slots = registry.activate_names(["alpha", "beta"])
    assert set(slots) == {"alpha", "beta"}
    described = registry.slots()
    assert [info.name for info in described] == sorted(
        ["alpha", "beta"], key=lambda name: slots[name]
    )
    assert described[0].to_dict()["pinned"] is False


def test_reset_slots_empties_the_pool_and_the_buffers() -> None:
    registry, model = make_registry(2)
    adapter_id = registry.register_adapter(make_adapter("a"))
    registry.pin(adapter_id)
    registry.reset_slots()
    assert registry.num_resident == 0
    assert registry.num_free_slots == 2
    assert not registry.is_pinned(adapter_id)
    layer = model.layers[0].q_proj
    assert float(layer.lora_a.abs().max()) == 0.0
    registry.check_invariants()


def test_attach_refuses_a_layer_with_a_different_pool_geometry() -> None:
    registry, _ = make_registry(2)
    foreign = LoRALinear(LinearBase(HIDDEN, HIDDEN, name="x"), num_slots=5, max_rank=4)
    with pytest.raises(LoRARegistryError, match="registry is"):
        registry.attach("x", foreign)


# -- accounting ----------------------------------------------------------------------------


def test_vram_report_compares_against_n_merged_copies() -> None:
    registry, model = make_registry(4, max_rank=4)
    for index in range(4):
        registry.register_adapter(make_adapter(f"a{index}", seed=index))
    base = registry.base_weight_bytes()
    assert base == 6 * HIDDEN * HIDDEN * 4, "six fp32 projections of HIDDEN x HIDDEN"

    report = registry.vram_report(num_adapters=4)
    assert report["base_bytes"] == base
    assert report["merged_bytes"] == 4 * base
    assert report["lora_bytes"] == base + registry.bytes_reserved
    assert report["saved_bytes"] == report["merged_bytes"] - report["lora_bytes"]
    assert report["saved_pct"] == pytest.approx(
        100.0 * report["saved_bytes"] / report["merged_bytes"]
    )
    # Four wrapped projections, each holding A[4, 8] and B[8, 4] per slot, fp32.
    assert registry.bytes_per_slot == 4 * (4 * HIDDEN + HIDDEN * 4) * 4
    assert registry.bytes_reserved == 4 * registry.bytes_per_slot + 4 * 4 * 4
    del model


def test_resident_bytes_track_occupancy() -> None:
    registry, _ = make_registry(4)
    ids = [registry.register_adapter(make_adapter(f"a{i}", seed=i)) for i in range(2)]
    assert registry.bytes_resident == 0
    registry.activate(ids)
    assert registry.bytes_resident == 2 * registry.bytes_per_slot
    assert registry.bytes_host == sum(registry.adapter(i).nbytes for i in ids)


def test_vram_report_needs_a_model_and_a_positive_count() -> None:
    registry = LoRARegistry(2, max_lora_rank=4, target_modules=TARGETS)
    with pytest.raises(LoRARegistryError, match="no model is bound"):
        registry.base_weight_bytes()
    bound, _ = make_registry(2)
    with pytest.raises(ValueError, match="num_adapters"):
        bound.vram_report(num_adapters=0)


def test_stats_dict_is_flat_and_prefixed() -> None:
    registry, _ = make_registry(2)
    registry.activate([registry.register_adapter(make_adapter("a"))])
    stats = registry.stats_dict()
    assert stats["lora_num_adapters"] == 1
    assert stats["lora_num_resident"] == 1
    assert stats["lora_loads"] == 1
    assert stats["lora_bytes_reserved"] == registry.bytes_reserved
    assert all(isinstance(value, int | float) for value in stats.values())
    assert all(key.startswith("lora_") for key in stats)


def test_hit_rate_is_zero_before_anything_happens() -> None:
    registry, _ = make_registry(1)
    assert registry.stats.hit_rate == 0.0
    assert registry.stats.adapter_token_fraction == 0.0


# -- the engine hook -------------------------------------------------------------------------


def scheduled_step(lora_ids: list[int]) -> Any:
    """One scheduler step holding a sequence per entry of ``lora_ids``."""
    scheduler = Scheduler(SchedulerConfig(num_blocks=64, block_size=16, max_num_seqs=8))
    for index, lora_id in enumerate(lora_ids):
        scheduler.add_request(f"r{index}", [10, 11, 12], lora_id=lora_id)
    return scheduler.schedule()


def test_build_context_returns_none_for_a_base_only_step() -> None:
    registry, _ = make_registry(2)
    assert registry.build_context(scheduled_step([NO_LORA, NO_LORA])) is None
    assert registry.stats.contexts == 1
    assert registry.stats.tokens == 6
    assert registry.stats.adapter_tokens == 0


def test_build_context_maps_every_token_to_a_live_slot() -> None:
    registry, _ = make_registry(2)
    first = registry.register_adapter(make_adapter("a"))
    second = registry.register_adapter(make_adapter("b", seed=1))
    context = registry.build_context(scheduled_step([first, NO_LORA, second]))
    assert isinstance(context, LoRABatch)
    context.validate()
    slots = context.token_lora_slot.tolist()
    assert slots[:3] == [registry.slot_for(first)] * 3
    assert slots[3:6] == [NO_LORA] * 3
    assert slots[6:] == [registry.slot_for(second)] * 3
    assert registry.stats.adapter_tokens == 6


def test_build_context_reflects_a_slot_that_moved() -> None:
    """The point of ids-versus-slots: a re-activated adapter may land elsewhere."""
    registry, _ = make_registry(1)
    first = registry.register_adapter(make_adapter("a"))
    second = registry.register_adapter(make_adapter("b", seed=1))
    registry.activate([first])
    context = registry.build_context(scheduled_step([second]))
    assert isinstance(context, LoRABatch)
    assert registry.slot_for(first) is None
    assert context.token_lora_slot.tolist() == [registry.slot_for(second)] * 3


def test_build_context_refuses_an_unregistered_adapter_id() -> None:
    registry, _ = make_registry(2)
    with pytest.raises(UnknownAdapterError):
        registry.build_context(scheduled_step([99]))


# -- configuration and end-to-end wiring --------------------------------------------------------


def test_lora_options_rejects_an_idle_configuration() -> None:
    with pytest.raises(ValueError, match="adapters or adapters_dir"):
        LoRAOptions()
    with pytest.raises(ValueError, match="target_modules"):
        LoRAOptions(adapters_dir="x", target_modules=[])
    options = LoRAOptions(adapters_dir="x", max_loras=3)
    assert options.max_loras == 3
    assert "q_proj" in options.target_modules


def test_lora_options_reads_the_engine_configs_untyped_block() -> None:
    from turboserve.engine.core.types import EngineConfig

    config = EngineConfig(model="m", lora={"max_loras": 2, "adapters_dir": "adapters"})
    options = LoRAOptions.from_engine_config(config)
    assert options is not None and options.max_loras == 2
    assert LoRAOptions.from_engine_config(EngineConfig(model="m")) is None


def test_setup_lora_registers_pins_and_installs_on_a_real_model(
    tiny_qwen2_path: Path, adapters_dir: Path
) -> None:
    from turboserve.engine.lora.layers import iter_lora_linears
    from turboserve.engine.model import CausalLM

    model = CausalLM.from_pretrained(
        tiny_qwen2_path, dtype=torch.float32, device="cpu", local_files_only=True
    )
    registry = setup_lora(
        model,
        LoRAOptions(
            max_loras=2,
            max_lora_rank=4,
            adapters_dir=str(adapters_dir),
            pinned=["tenant-0"],
        ),
    )
    assert len(registry) == 3
    assert registry.is_pinned("tenant-0")
    assert len(dict(iter_lora_linears(model))) == 2 * 7
    report = registry.vram_report(num_adapters=3)
    assert report["merged_bytes"] == 3 * report["base_bytes"]
    assert 0.0 < report["saved_pct"] < 100.0
    registry.check_invariants()


def test_register_directory_refuses_a_limit_the_directory_cannot_meet(
    adapters_dir: Path,
) -> None:
    registry, _ = make_registry(4, max_rank=4)
    with pytest.raises(AdapterError, match="3 adapters, 9 were requested"):
        registry.register_directory(adapters_dir, limit=9)
    with pytest.raises(ValueError, match="limit must be positive"):
        registry.register_directory(adapters_dir, limit=0)
