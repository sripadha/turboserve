"""Profile tests: the shipped sizes are what the profiles file declares, and typos are errors."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
import yaml

from turboserve.bench.profiles import (
    PROFILE_SCHEMA_VERSION,
    PROFILES_ENV_VAR,
    SCENARIO_NAMES,
    BenchProfile,
    ChaosProfile,
    ProfileError,
    SLOSpec,
    TokenRange,
    available_profiles,
    default_profiles_path,
    load_profile,
    load_profiles,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "configs" / "bench" / "profiles.yaml"


@pytest.fixture
def profiles() -> dict[str, BenchProfile]:
    return load_profiles(SHIPPED)


# -- the shipped file ---------------------------------------------------------------------


def test_the_shipped_file_defines_both_profiles(profiles: dict[str, BenchProfile]) -> None:
    assert set(profiles) == {"h100", "dev-2060"}
    for name, profile in profiles.items():
        assert profile.name == name
        assert profile.description
        assert profile.tenants
        for scenario in SCENARIO_NAMES:
            assert profile.scenario(scenario) is not None


def test_the_h100_profile_matches_the_specified_sizes(profiles: dict[str, BenchProfile]) -> None:
    h100 = profiles["h100"]
    naive = h100.scenarios.naive_vs_cb
    assert naive.model == "Qwen/Qwen2.5-7B-Instruct"
    assert naive.dtype == "bfloat16"
    assert naive.num_requests == 256
    assert naive.input_tokens.as_tuple() == (128, 1024)
    assert naive.output_tokens.as_tuple() == (64, 512)
    assert naive.concurrencies == [32, 64, 128]
    # Both production engines are arms of the published profile; each is driven through its
    # own --url, so a run measures whichever servers that host has.
    assert {"vllm", "sglang"} <= set(naive.backends)

    assert h100.scenarios.prefix_cache.shared_prefix_tokens == 1024
    assert h100.scenarios.prefix_cache.prefix_caching == [False, True]
    assert {"vllm", "sglang"} <= set(h100.scenarios.prefix_cache.backends)

    spec_decode = h100.scenarios.spec_decode
    assert spec_decode.speculative_tokens == [2, 4, 6]
    assert spec_decode.concurrencies == [1, 4, 16]
    assert [pair.name for pair in spec_decode.pairs] == [
        "target-3b-draft-1.5b",
        "target-7b-draft-0.5b",
        "target-7b-ngram",
    ]
    assert spec_decode.pairs[0].draft == "Qwen/Qwen2.5-1.5B-Instruct"
    assert spec_decode.pairs[-1].drafter == "ngram"
    assert spec_decode.pairs[-1].draft is None

    multi_lora = h100.scenarios.multi_lora
    assert multi_lora.adapter_counts == [10, 32, 100, 128]
    assert multi_lora.lora_rank == 16
    assert multi_lora.include_base_only

    chaos = h100.scenarios.chaos
    assert isinstance(chaos, ChaosProfile)
    assert chaos.num_workers == 3


def test_the_dev_profile_is_small_and_fp16(profiles: dict[str, BenchProfile]) -> None:
    dev = profiles["dev-2060"]
    assert dev.scenarios.naive_vs_cb.num_requests == 64
    assert dev.scenarios.naive_vs_cb.concurrencies == [32]
    assert dev.scenarios.prefix_cache.shared_prefix_tokens == 512
    assert dev.scenarios.multi_lora.adapter_counts == [16, 64]
    # Pre-Ampere consumer cards have no bf16 tensor cores, so no dev arm may ask for bf16.
    for scenario in SCENARIO_NAMES:
        assert dev.scenario(scenario).dtype == "float16"
    # Every dev model is smaller than the h100 counterpart.
    assert "0.5B" in dev.scenarios.naive_vs_cb.model


def test_no_shipped_profile_states_a_latency_objective(profiles: dict[str, BenchProfile]) -> None:
    """Objectives are supplied at run time; the repository states no target figure."""
    for profile in profiles.values():
        for scenario in SCENARIO_NAMES:
            assert profile.scenario(scenario).slo is None


def test_config_for_carries_the_whole_sized_workload(profiles: dict[str, BenchProfile]) -> None:
    block = profiles["h100"].config_for("prefix_cache", extra_note="unit")
    assert block["profile"] == "h100"
    assert block["scenario"] == "prefix_cache"
    assert block["seed"] == profiles["h100"].seed
    assert block["workload"]["shared_prefix_tokens"] == 1024
    assert block["extra_note"] == "unit"
    # JSON-serialisable, because it is embedded verbatim in the result file.
    import json

    json.dumps(block)


def test_unknown_scenario_names_are_rejected(profiles: dict[str, BenchProfile]) -> None:
    with pytest.raises(KeyError, match="unknown scenario"):
        profiles["h100"].scenario("does_not_exist")


# -- lookup -------------------------------------------------------------------------------


def test_load_profile_and_available_profiles(tmp_path: Path) -> None:
    assert load_profile("h100", path=SHIPPED).name == "h100"
    assert available_profiles(SHIPPED) == ["dev-2060", "h100"]
    with pytest.raises(KeyError, match="available: dev-2060, h100"):
        load_profile("a6000", path=SHIPPED)


def test_default_path_honours_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROFILES_ENV_VAR, str(SHIPPED))
    assert default_profiles_path() == SHIPPED
    monkeypatch.setenv(PROFILES_ENV_VAR, str(tmp_path / "nope.yaml"))
    with pytest.raises(ProfileError, match="does not name a readable file"):
        default_profiles_path()


def test_default_path_finds_the_repository_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROFILES_ENV_VAR, raising=False)
    monkeypatch.chdir(REPO_ROOT)
    assert default_profiles_path() == SHIPPED


# -- validation ---------------------------------------------------------------------------


def mutate(edit: Any) -> dict[str, Any]:
    """Load the shipped file, apply ``edit`` to the parsed mapping, and return it."""
    data = yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
    edit(data)
    return data


def write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_an_unknown_key_is_an_error(tmp_path: Path) -> None:
    data = mutate(lambda d: d["profiles"]["h100"]["scenarios"]["naive_vs_cb"].update(typo=1))
    with pytest.raises(ProfileError, match="does not match the profile schema"):
        load_profiles(write(tmp_path, data))


def test_a_future_schema_version_is_refused(tmp_path: Path) -> None:
    data = mutate(lambda d: d.update(schema_version=PROFILE_SCHEMA_VERSION + 1))
    with pytest.raises(ProfileError, match="cannot be read by this build"):
        load_profiles(write(tmp_path, data))


def test_a_prefix_longer_than_the_prompt_is_an_error(tmp_path: Path) -> None:
    data = mutate(
        lambda d: d["profiles"]["h100"]["scenarios"]["prefix_cache"].update(
            shared_prefix_tokens=99_999
        )
    )
    with pytest.raises(ProfileError, match="must be smaller than"):
        load_profiles(write(tmp_path, data))


def test_an_ngram_pair_may_not_name_a_draft_model(tmp_path: Path) -> None:
    def edit(data: Any) -> None:
        pairs = data["profiles"]["h100"]["scenarios"]["spec_decode"]["pairs"]
        pairs[-1]["draft"] = "Qwen/Qwen2.5-0.5B-Instruct"

    with pytest.raises(ProfileError, match="must not name a draft model"):
        load_profiles(write(tmp_path, mutate(edit)))


def test_a_model_pair_needs_a_draft_model(tmp_path: Path) -> None:
    def edit(data: Any) -> None:
        data["profiles"]["h100"]["scenarios"]["spec_decode"]["pairs"][0]["draft"] = None

    with pytest.raises(ProfileError, match="requires a draft model"):
        load_profiles(write(tmp_path, mutate(edit)))


def test_duplicate_pair_names_are_rejected(tmp_path: Path) -> None:
    def edit(data: Any) -> None:
        pairs = data["profiles"]["h100"]["scenarios"]["spec_decode"]["pairs"]
        pairs[1]["name"] = pairs[0]["name"]

    with pytest.raises(ProfileError, match="unique names"):
        load_profiles(write(tmp_path, mutate(edit)))


def test_a_zero_in_a_sweep_is_rejected(tmp_path: Path) -> None:
    def edit(data: Any) -> None:
        data["profiles"]["h100"]["scenarios"]["multi_lora"]["adapter_counts"] = [0, 10]

    with pytest.raises(ProfileError, match="adapter_counts"):
        load_profiles(write(tmp_path, mutate(edit)))


def test_a_fault_interval_longer_than_the_window_is_rejected(tmp_path: Path) -> None:
    def edit(data: Any) -> None:
        data["profiles"]["h100"]["scenarios"]["chaos"]["fault_interval_s"] = 999.0

    with pytest.raises(ProfileError, match="would inject no fault"):
        load_profiles(write(tmp_path, mutate(edit)))


def test_duplicate_tenants_are_rejected(tmp_path: Path) -> None:
    data = mutate(lambda d: d["profiles"]["h100"].update(tenants=["a", "a"]))
    with pytest.raises(ProfileError, match="tenants must be unique"):
        load_profiles(write(tmp_path, data))


def test_a_missing_scenario_is_an_error(tmp_path: Path) -> None:
    data = mutate(lambda d: d["profiles"]["h100"]["scenarios"].pop("chaos"))
    with pytest.raises(ProfileError, match="does not match the profile schema"):
        load_profiles(write(tmp_path, data))


def test_unreadable_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="cannot read"):
        load_profiles(tmp_path / "absent.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("profiles: [\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="not valid YAML"):
        load_profiles(broken)
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("7\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="mapping at the top level"):
        load_profiles(scalar)


# -- small pieces -------------------------------------------------------------------------


def test_token_range_samples_inside_its_bounds_and_is_ordered() -> None:
    token_range = TokenRange(min=4, max=9)
    rng = random.Random(0)
    draws = [token_range.sample(rng) for _ in range(50)]
    assert all(4 <= value <= 9 for value in draws)
    assert token_range.as_tuple() == (4, 9)
    with pytest.raises(ValueError, match="exceeds max"):
        TokenRange(min=9, max=4)


def test_slo_spec_converts_and_reports_emptiness() -> None:
    assert SLOSpec().is_empty
    spec = SLOSpec(ttft_ms=250.0)
    assert not spec.is_empty
    assert spec.to_slo().ttft_ms == 250.0
    assert spec.to_slo().e2e_ms is None
    with pytest.raises(ValueError, match="greater than 0"):
        SLOSpec(ttft_ms=0.0)
