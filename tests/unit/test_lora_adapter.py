"""Reading PEFT adapter directories: what is accepted, what is refused, and why.

The adapters under test are real ones, trained here by ``scripts/make_lora_adapters.py``
over the cached tiny random Qwen2 checkpoint (two optimiser steps -- enough to move the
``lora_B`` factors off their zero initialisation, which is all these tests need). Using a
handwritten fixture instead would test this module against an idea of PEFT's format rather
than against PEFT's actual output, and the format is exactly what this module exists to
track.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from turboserve.engine.lora.adapter import (
    PEFT_CONFIG_FILE,
    PEFT_PICKLE_FILE,
    PEFT_WEIGHTS_FILE,
    AdapterError,
    LoRAAdapter,
    LoRAWeights,
    adapter_scaling,
    discover_adapters,
    load_peft_adapter,
    merged_state_dict,
    peft_module_name,
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
    """Two real PEFT adapters over the tiny Qwen2 checkpoint, trained for two steps."""
    module = load_adapter_script()
    out = tmp_path_factory.mktemp("adapters")
    module.make_adapters(
        str(tiny_qwen2_path),
        out,
        count=2,
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


def test_the_script_writes_loadable_peft_directories(adapters_dir: Path) -> None:
    found = discover_adapters(adapters_dir)
    assert [path.name for path in found] == ["tenant-0", "tenant-1"]
    for path in found:
        assert (path / PEFT_CONFIG_FILE).is_file()
        assert (path / PEFT_WEIGHTS_FILE).is_file()
    manifest = json.loads((adapters_dir / "manifest.json").read_text())
    assert manifest["count"] == 2
    assert [entry["name"] for entry in manifest["adapters"]] == ["tenant-0", "tenant-1"]


def test_load_reads_rank_alpha_and_every_targeted_projection(adapters_dir: Path) -> None:
    adapter = load_peft_adapter(adapters_dir / "tenant-0")
    assert adapter.name == "tenant-0"
    assert adapter.rank == 4
    assert adapter.alpha == 64
    # Two layers times the seven wrapped projections.
    assert len(adapter.weights) == 14
    assert "model.layers.0.self_attn.q_proj" in adapter
    for weights in adapter:
        assert weights.rank == 4
        assert weights.a.shape[1] == weights.in_features
        assert weights.b.shape[0] == weights.out_features
        assert weights.scaling == pytest.approx(64 / 4)


def test_the_two_adapters_are_actually_different(adapters_dir: Path) -> None:
    """Distinct synthetic tasks must produce distinct factors, or the whole set is one adapter."""
    first = load_peft_adapter(adapters_dir / "tenant-0")
    second = load_peft_adapter(adapters_dir / "tenant-1")
    module = "model.layers.0.self_attn.q_proj"
    delta_a = first.weights[module].merged_delta()
    delta_b = second.weights[module].merged_delta()
    assert float(delta_a.abs().max()) > 0.0
    assert float((delta_a - delta_b).abs().max()) > 1e-6


def test_names_default_to_the_directory_and_can_be_overridden(adapters_dir: Path) -> None:
    assert load_peft_adapter(adapters_dir / "tenant-1").name == "tenant-1"
    assert load_peft_adapter(adapters_dir / "tenant-1", name="acme").name == "acme"


def test_dtype_is_applied_on_load(adapters_dir: Path) -> None:
    adapter = load_peft_adapter(adapters_dir / "tenant-0", dtype=torch.float16)
    assert adapter.dtype == torch.float16
    again = adapter.to(dtype=torch.float16)
    assert again is adapter, "a no-op cast must not copy every factor"


def test_validate_against_accepts_the_model_it_was_trained_on(
    adapters_dir: Path, tiny_qwen2_path: Path
) -> None:
    from turboserve.engine.lora.adapter import linear_shapes
    from turboserve.engine.model import CausalLM

    model = CausalLM.from_pretrained(
        tiny_qwen2_path, dtype=torch.float32, device="cpu", local_files_only=True
    )
    adapter = load_peft_adapter(adapters_dir / "tenant-0")
    adapter.validate_against(linear_shapes(model))


def test_validate_against_rejects_a_shape_mismatch(adapters_dir: Path) -> None:
    adapter = load_peft_adapter(adapters_dir / "tenant-0")
    shapes = dict.fromkeys(adapter.modules, (999, 999))
    with pytest.raises(AdapterError, match="module"):
        adapter.validate_against(shapes)


def test_validate_against_rejects_an_unknown_module(adapters_dir: Path) -> None:
    adapter = load_peft_adapter(adapters_dir / "tenant-0")
    with pytest.raises(AdapterError, match="does not have"):
        adapter.validate_against({})
    adapter.validate_against({}, strict=False)


def test_merged_state_dict_is_base_plus_scaled_low_rank(adapters_dir: Path) -> None:
    adapter = load_peft_adapter(adapters_dir / "tenant-0")
    module = "model.layers.1.mlp.down_proj"
    weights = adapter.weights[module]
    base = torch.zeros(weights.out_features, weights.in_features)
    merged = merged_state_dict(adapter, {module: base}, modules=[module])
    expected = weights.scaling * (weights.b.float() @ weights.a.float())
    torch.testing.assert_close(merged[module], expected)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            ("model.layers.0.self_attn.q_proj", "A"),
        ),
        (
            "base_model.model.model.layers.3.mlp.up_proj.lora_B.default.weight",
            ("model.layers.3.mlp.up_proj", "B"),
        ),
        ("base_model.model.lm_head.lora_A.weight", ("lm_head", "A")),
        ("some.other.tensor", None),
    ],
)
def test_peft_key_parsing(key: str, expected: tuple[str, str] | None) -> None:
    assert peft_module_name(key) == expected


def test_peft_key_parsing_refuses_unsupported_variants() -> None:
    with pytest.raises(AdapterError, match="DoRA"):
        peft_module_name("base_model.model.model.layers.0.mlp.up_proj.lora_magnitude_vector")
    with pytest.raises(AdapterError, match="embedding"):
        peft_module_name("base_model.model.model.embed_tokens.lora_embedding_A")


def test_scaling_formula() -> None:
    assert adapter_scaling(8, 16) == pytest.approx(2.0)
    assert adapter_scaling(16, 16, use_rslora=True) == pytest.approx(4.0)
    with pytest.raises(AdapterError):
        adapter_scaling(0, 16)


def _write_config(directory: Path, **overrides: object) -> Path:
    """A minimal adapter directory carrying only ``adapter_config.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {"peft_type": "LORA", "r": 4, "lora_alpha": 8}
    data.update(overrides)
    (directory / PEFT_CONFIG_FILE).write_text(json.dumps(data), encoding="utf-8")
    return directory


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"peft_type": "IA3"}, "not supported"),
        ({"use_dora": True}, "DoRA"),
        ({"modules_to_save": ["score"]}, "modules_to_save"),
        ({"bias": "all"}, "bias"),
        ({"fan_in_fan_out": True}, "fan_in_fan_out"),
        ({"target_parameters": ["w"]}, "parameter-targeted"),
    ],
)
def test_unsupported_configurations_are_refused(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    directory = _write_config(tmp_path / "adapter", **overrides)
    with pytest.raises(AdapterError, match=message):
        load_peft_adapter(directory)


def test_a_pickle_checkpoint_is_refused_with_a_way_out(tmp_path: Path) -> None:
    directory = _write_config(tmp_path / "adapter")
    (directory / PEFT_PICKLE_FILE).write_bytes(b"not actually a pickle")
    with pytest.raises(AdapterError, match="safe_serialization"):
        load_peft_adapter(directory)


def test_a_directory_without_weights_is_refused(tmp_path: Path) -> None:
    directory = _write_config(tmp_path / "adapter")
    with pytest.raises(AdapterError, match=PEFT_WEIGHTS_FILE):
        load_peft_adapter(directory)


def test_a_non_adapter_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(AdapterError, match="not a PEFT adapter"):
        load_peft_adapter(tmp_path / "empty")
    with pytest.raises(AdapterError, match="not a directory"):
        load_peft_adapter(tmp_path / "missing")


def test_discover_adapters_skips_directories_without_a_config(
    tmp_path: Path, adapters_dir: Path
) -> None:
    root = tmp_path / "mixed"
    root.mkdir()
    (root / "notes").mkdir()
    _write_config(root / "b-adapter")
    _write_config(root / "a-adapter")
    assert [path.name for path in discover_adapters(root)] == ["a-adapter", "b-adapter"]
    # A directory that is itself an adapter resolves to itself.
    assert discover_adapters(adapters_dir / "tenant-0") == [adapters_dir / "tenant-0"]


def test_factors_must_come_in_pairs() -> None:
    with pytest.raises(AdapterError, match="rank mismatch"):
        LoRAWeights(module="m", a=torch.zeros(4, 8), b=torch.zeros(8, 2), scaling=1.0)
    with pytest.raises(AdapterError, match="2-D"):
        LoRAWeights(module="m", a=torch.zeros(4), b=torch.zeros(8, 4), scaling=1.0)


def test_an_adapter_needs_a_name_and_some_weights() -> None:
    weights = {"m": LoRAWeights(module="m", a=torch.zeros(2, 4), b=torch.zeros(4, 2), scaling=1.0)}
    with pytest.raises(AdapterError, match="name"):
        LoRAAdapter(name="", weights=weights, rank=2, alpha=4.0, target_modules=())
    with pytest.raises(AdapterError, match="no LoRA factors"):
        LoRAAdapter(name="x", weights={}, rank=2, alpha=4.0, target_modules=())


def test_to_dict_is_json_safe(adapters_dir: Path) -> None:
    adapter = load_peft_adapter(adapters_dir / "tenant-0")
    payload = json.loads(json.dumps(adapter.to_dict()))
    assert payload["rank"] == 4
    assert payload["num_modules"] == 14
    assert payload["dtype"] == "float32"
