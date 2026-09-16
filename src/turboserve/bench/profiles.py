"""Machine-sized workload definitions for the benchmark scenarios.

A *profile* answers one question -- "how big is this run on this hardware?" -- and nothing
else. It names the models, the number of requests, the prompt and completion lengths, the
concurrencies and the scenario-specific knobs (shared prefix length, speculative ``k``,
adapter count). Every scenario takes ``--profile h100|dev-2060`` and reads its sizes from
here, so the same scenario code produces a publishable run on a rented H100 and a seconds-
long smoke run on a small consumer GPU without a branch in the scenario.

Two deliberate omissions:

* **No latency or throughput objective.** Goodput needs a service-level objective, but an
  objective shipped in the repository reads as a claim about what the system achieves.
  Objectives are passed at run time (``--slo-ttft-ms`` and friends) and recorded in the
  result file, so a table can always be traced to the objective it was scored against. The
  schema has a place for one (:class:`SLOSpec`) because a team running this in their own
  fleet will want it pinned in YAML; the profiles shipped here leave it unset.
* **No paths to weights.** Models are named by their Hugging Face id and resolved by the
  scenario, because the measurement host downloads them and a development checkout may not
  have them at all.

The file is validated strictly (``extra="forbid"``): a typo in a profile is a loud error
before a GPU-hour is spent, not a silently ignored key.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from turboserve.bench.records import SLO
from turboserve.config import DeviceName, DTypeName

if TYPE_CHECKING:  # pragma: no cover - typing only
    import random

logger = logging.getLogger(__name__)

__all__ = [
    "PROFILES_ENV_VAR",
    "PROFILES_RELATIVE_PATH",
    "PROFILE_SCHEMA_VERSION",
    "SCENARIO_NAMES",
    "SLOSpec",
    "BenchProfile",
    "ChaosProfile",
    "MultiLoRAProfile",
    "NaiveVsCBProfile",
    "PrefixCacheProfile",
    "ProfileError",
    "ProfileFile",
    "ScenarioProfile",
    "ScenarioSet",
    "SpecDecodePair",
    "SpecDecodeProfile",
    "TokenRange",
    "available_profiles",
    "default_profiles_path",
    "load_profile",
    "load_profiles",
]

#: Overrides the search for ``configs/bench/profiles.yaml`` -- set it on the measurement
#: host when the repository is not the working directory.
PROFILES_ENV_VAR = "TURBOSERVE_BENCH_PROFILES"

#: Where the shipped profiles live, relative to the repository root.
PROFILES_RELATIVE_PATH = Path("configs/bench/profiles.yaml")

#: Bumped only when a key is removed or changes meaning; adding an optional key does not.
PROFILE_SCHEMA_VERSION = 1

#: The scenarios a profile must size. Kept as a tuple so the CLI can list them without
#: importing the scenario modules (which import torch).
SCENARIO_NAMES: tuple[str, ...] = (
    "naive_vs_cb",
    "prefix_cache",
    "spec_decode",
    "multi_lora",
    "chaos",
)


class ProfileError(ValueError):
    """The profiles file is missing, unparsable, or does not match the schema."""


class _Strict(BaseModel):
    """Base for every profile model: unknown keys are errors and instances are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TokenRange(_Strict):
    """An inclusive token-length range the prompt builder samples from.

    Ranges rather than a single length because a benchmark of one prompt length measures
    one point of a curve: real traffic mixes short and long requests, and a scheduler that
    only looks good on uniform input is not interesting.
    """

    min: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> TokenRange:
        if self.min > self.max:
            raise ValueError(f"token range min ({self.min}) exceeds max ({self.max})")
        return self

    def sample(self, rng: random.Random) -> int:
        """Draw one length uniformly from the range using the caller's seeded RNG."""
        return rng.randint(self.min, self.max)

    def as_tuple(self) -> tuple[int, int]:
        """``(min, max)``, the form the prompt builder takes."""
        return (self.min, self.max)


class SLOSpec(_Strict):
    """Latency objectives to score goodput against; every field is optional.

    Left unset in the shipped profiles -- see the module docstring for why.
    """

    ttft_ms: float | None = Field(default=None, gt=0.0)
    tpot_ms: float | None = Field(default=None, gt=0.0)
    e2e_ms: float | None = Field(default=None, gt=0.0)

    def to_slo(self) -> SLO:
        """Convert to the :class:`~turboserve.bench.records.SLO` the summariser takes."""
        return SLO(ttft_ms=self.ttft_ms, tpot_ms=self.tpot_ms, e2e_ms=self.e2e_ms)

    @property
    def is_empty(self) -> bool:
        """Whether no objective is asserted at all."""
        return self.ttft_ms is None and self.tpot_ms is None and self.e2e_ms is None


class ScenarioProfile(_Strict):
    """Fields every count-driven scenario needs: how much work, and how big each unit is."""

    dtype: DTypeName = "auto"
    num_requests: int = Field(ge=1)
    input_tokens: TokenRange
    output_tokens: TokenRange
    backends: list[str] = Field(min_length=1)
    slo: SLOSpec | None = None


class NaiveVsCBProfile(ScenarioProfile):
    """Sizes for the sequential / static-batch / continuous-batching comparison."""

    model: str = Field(min_length=1)
    concurrencies: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _positive_concurrencies(self) -> NaiveVsCBProfile:
        _require_positive(self.concurrencies, "concurrencies")
        return self


class PrefixCacheProfile(ScenarioProfile):
    """Sizes for the shared-system-prompt experiment, run with the cache off and on."""

    model: str = Field(min_length=1)
    shared_prefix_tokens: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    prefix_caching: list[bool] = Field(min_length=1)

    @model_validator(mode="after")
    def _prefix_fits(self) -> PrefixCacheProfile:
        if self.shared_prefix_tokens >= self.input_tokens.min:
            raise ValueError(
                f"shared_prefix_tokens ({self.shared_prefix_tokens}) must be smaller than "
                f"input_tokens.min ({self.input_tokens.min}); otherwise no request has a "
                "unique suffix and the cache hit rate is trivially one"
            )
        if len(set(self.prefix_caching)) != len(self.prefix_caching):
            raise ValueError("prefix_caching lists the same arm twice")
        return self


class SpecDecodePair(_Strict):
    """One target/drafter pairing.

    ``drafter="model"`` needs a ``draft`` checkpoint; ``drafter="ngram"`` proposes from the
    prompt's own n-grams and must not name one, which is the difference the validator
    enforces so a mis-specified pair cannot silently run the wrong drafter.
    """

    name: str = Field(min_length=1)
    target: str = Field(min_length=1)
    draft: str | None = None
    drafter: Literal["model", "ngram"] = "model"

    @model_validator(mode="after")
    def _drafter_matches_draft(self) -> SpecDecodePair:
        if self.drafter == "model" and not self.draft:
            raise ValueError(f"pair {self.name!r}: drafter 'model' requires a draft model")
        if self.drafter == "ngram" and self.draft:
            raise ValueError(f"pair {self.name!r}: drafter 'ngram' must not name a draft model")
        return self


class SpecDecodeProfile(ScenarioProfile):
    """Sizes for speculative decoding: the pairings, the ``k`` sweep, the concurrencies."""

    pairs: list[SpecDecodePair] = Field(min_length=1)
    speculative_tokens: list[int] = Field(min_length=1)
    concurrencies: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _positive_sweeps(self) -> SpecDecodeProfile:
        _require_positive(self.speculative_tokens, "speculative_tokens")
        _require_positive(self.concurrencies, "concurrencies")
        names = [pair.name for pair in self.pairs]
        if len(set(names)) != len(names):
            raise ValueError("spec_decode pairs must have unique names")
        return self


class MultiLoRAProfile(ScenarioProfile):
    """Sizes for the multi-adapter experiment: how many adapters, of what rank."""

    model: str = Field(min_length=1)
    adapter_counts: list[int] = Field(min_length=1)
    lora_rank: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    include_base_only: bool = True
    """Whether to also run with no adapter, the reference point a p95 loss is measured against."""

    @model_validator(mode="after")
    def _positive_counts(self) -> MultiLoRAProfile:
        _require_positive(self.adapter_counts, "adapter_counts")
        return self


class ChaosProfile(_Strict):
    """Sizes for the chaos run, which is duration-driven rather than count-driven.

    Fault injection is measured over a window of steady offered load: a fixed request count
    would end early exactly when a fault slowed the fleet down, which is the opposite of
    what the experiment is for.
    """

    model: str = Field(min_length=1)
    dtype: DTypeName = "auto"
    input_tokens: TokenRange
    output_tokens: TokenRange
    backends: list[str] = Field(min_length=1)
    num_workers: int = Field(ge=1)
    rate_rps: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)
    fault_interval_s: float = Field(gt=0.0)
    slo: SLOSpec | None = None

    @model_validator(mode="after")
    def _fault_fits_window(self) -> ChaosProfile:
        if self.fault_interval_s > self.duration_s:
            raise ValueError(
                f"fault_interval_s ({self.fault_interval_s}) exceeds duration_s "
                f"({self.duration_s}); the run would inject no fault at all"
            )
        return self


class ScenarioSet(_Strict):
    """Every scenario a profile must size; a missing one is a validation error."""

    naive_vs_cb: NaiveVsCBProfile
    prefix_cache: PrefixCacheProfile
    spec_decode: SpecDecodeProfile
    multi_lora: MultiLoRAProfile
    chaos: ChaosProfile


class BenchProfile(_Strict):
    """One machine's worth of workload sizes, plus the seed that makes runs comparable."""

    name: str = ""
    description: str = ""
    device: DeviceName = "auto"
    seed: int = 0
    tenants: list[str] = Field(min_length=1)
    scenarios: ScenarioSet

    @model_validator(mode="after")
    def _unique_tenants(self) -> BenchProfile:
        if len(set(self.tenants)) != len(self.tenants):
            raise ValueError("tenants must be unique")
        return self

    def scenario(self, name: str) -> ScenarioProfile | ChaosProfile:
        """The sizes for one scenario, by the name the CLI subcommand uses."""
        table: dict[str, ScenarioProfile | ChaosProfile] = {
            "naive_vs_cb": self.scenarios.naive_vs_cb,
            "prefix_cache": self.scenarios.prefix_cache,
            "spec_decode": self.scenarios.spec_decode,
            "multi_lora": self.scenarios.multi_lora,
            "chaos": self.scenarios.chaos,
        }
        try:
            return table[name]
        except KeyError:
            raise KeyError(
                f"unknown scenario {name!r}; known: {', '.join(SCENARIO_NAMES)}"
            ) from None

    def config_for(self, scenario: str, **extra: Any) -> dict[str, Any]:
        """The ``config`` block a scenario embeds in its result file.

        Carries the whole sized workload, not a summary of it, so a result can be re-run
        from its own JSON without consulting the profiles file that has since changed.
        """
        block: dict[str, Any] = {
            "profile": self.name,
            "scenario": scenario,
            "device": self.device,
            "seed": self.seed,
            "tenants": list(self.tenants),
            "workload": self.scenario(scenario).model_dump(mode="json"),
        }
        block.update(extra)
        return block


class ProfileFile(_Strict):
    """The parsed ``configs/bench/profiles.yaml``."""

    schema_version: int
    profiles: dict[str, BenchProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def _known_schema(self) -> ProfileFile:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"profiles schema_version {self.schema_version} cannot be read by this "
                f"build (expected {PROFILE_SCHEMA_VERSION})"
            )
        return self

    def named(self) -> dict[str, BenchProfile]:
        """The profiles with :attr:`BenchProfile.name` filled in from the mapping key."""
        return {
            key: profile.model_copy(update={"name": key}) for key, profile in self.profiles.items()
        }


def _require_positive(values: list[int], field_name: str) -> None:
    """Raise unless every entry of a sweep list is at least one."""
    bad = [value for value in values if value < 1]
    if bad:
        raise ValueError(f"{field_name} must be >= 1, got {bad}")


def default_profiles_path() -> Path:
    """Locate ``configs/bench/profiles.yaml``.

    Searched in order: the ``TURBOSERVE_BENCH_PROFILES`` environment variable, then the
    current working directory and its parents, then the parents of this module. The last
    step matters for an editable install run from an unrelated directory; the first is the
    escape hatch for a deployment where the configs live outside the source tree.
    """
    override = os.environ.get(PROFILES_ENV_VAR)
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise ProfileError(f"{PROFILES_ENV_VAR}={override!r} does not name a readable file")
        return candidate
    roots = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / PROFILES_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    raise ProfileError(
        f"could not find {PROFILES_RELATIVE_PATH} from {Path.cwd()} or "
        f"{Path(__file__).resolve().parent}; set {PROFILES_ENV_VAR} to its path"
    )


def load_profiles(path: Path | str | None = None) -> dict[str, BenchProfile]:
    """Parse and validate every profile in the file, keyed by name."""
    resolved = Path(path) if path is not None else default_profiles_path()
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProfileError(f"cannot read {resolved}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ProfileError(f"{resolved} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"{resolved} must contain a mapping at the top level")
    try:
        parsed = ProfileFile.model_validate(raw)
    except ValidationError as exc:
        raise ProfileError(f"{resolved} does not match the profile schema:\n{exc}") from exc
    logger.debug("loaded %d bench profiles from %s", len(parsed.profiles), resolved)
    return parsed.named()


def load_profile(name: str, *, path: Path | str | None = None) -> BenchProfile:
    """One profile by name, with a message that lists the alternatives when it is absent."""
    profiles = load_profiles(path)
    try:
        return profiles[name]
    except KeyError:
        raise KeyError(
            f"unknown profile {name!r}; available: {', '.join(sorted(profiles))}"
        ) from None


def available_profiles(path: Path | str | None = None) -> list[str]:
    """Sorted profile names, for ``--profile`` help text and the CLI's error messages."""
    return sorted(load_profiles(path))
