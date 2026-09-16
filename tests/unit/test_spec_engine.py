"""End-to-end tests for the speculative engine, on CPU with cached tiny-random models.

The contract is blunt: **speculation must not change the output**. Every test that matters
here therefore runs the ordinary :class:`~turboserve.engine.runtime.engine.LLMEngine` and the
speculative one over the same prompts and compares the token ids, under drafters chosen to
force the three interesting regimes:

* a draft model that *is* the target, so every draft token is accepted;
* an n-gram drafter under a KV pool small enough to preempt, so drafts are partly accepted
  and sequences are rolled back and recomputed while speculating;
* a deliberately wrong drafter, so every draft token is rejected.

If the rollback bookkeeping were wrong in any of those, the comparison would fail: an
over-credited ``num_computed_tokens`` skips a position, an under-credited one recomputes and
duplicates, and a stale draft token left in a sequence corrupts everything after it.

No network, no GPU, no real model: the tiny-random checkpoints come from the shared Hugging
Face cache through the ``tiny_qwen2_path`` / ``tiny_llama_path`` fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from turboserve.engine.core.sequence import Sequence
from turboserve.engine.core.types import EngineConfig, SamplingParams, SchedulerConfig
from turboserve.engine.runtime.engine import LLMEngine
from turboserve.engine.spec.drafter import Drafter, DraftProposal, ModelDrafter
from turboserve.engine.spec.ngram import NgramDrafter, find_ngram_continuation
from turboserve.engine.spec.spec_engine import (
    SpeculativeConfig,
    SpeculativeLLMEngine,
)

PROMPTS: tuple[str, ...] = (
    "Hello world",
    "The quick brown fox jumps over",
    "A B C D E F G H I J",
    "Paris is the capital of",
)

GREEDY = SamplingParams(max_tokens=12, temperature=0.0)


def engine_config(model_path: Path, *, blocks: int = 64, **speculative: Any) -> EngineConfig:
    """An all-CPU fp32 config with an explicit block count (no memory profiling in tests)."""
    return EngineConfig(
        model=str(model_path),
        device="cpu",
        dtype="float32",
        scheduler=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=256,
            block_size=16,
            num_blocks=blocks,
        ),
        speculative=speculative or None,
    )


def self_drafter(model_path: Path, *, blocks: int = 64) -> ModelDrafter:
    """A draft model that is a second copy of the target, so every draft is accepted.

    Using the target as its own drafter is not a shortcut: it isolates the *mechanism* from
    the *quality* of the guesses. Acceptance is then known to be total, so any divergence
    from the reference output is a bug in batching, rollback or feedback rather than an
    unlucky draft.
    """
    return ModelDrafter.from_pretrained(
        str(model_path),
        dtype=torch.float32,
        device="cpu",
        num_blocks=blocks,
        block_size=16,
    )


def completions(engine: LLMEngine, prompts: tuple[str, ...], sampling: SamplingParams) -> dict:
    """Run prompts to completion and return ``{request_id: token ids}``."""
    return {
        output.request_id: list(output.new_token_ids)
        for output in engine.generate(list(prompts), sampling)
    }


class WrongDrafter:
    """A drafter whose guesses are always the same token, so they are always rejected.

    Exists to exercise the rejection path deterministically: the rollback, the discarded KV
    and the single bonus token are the whole of what a wholly wrong draft should cost.
    """

    name = "wrong"

    def __init__(self, token: int = 0) -> None:
        self.token = token
        self.calls = 0
        self.released: list[int] = []

    def propose(self, seqs: Any, k: int) -> DraftProposal:
        self.calls += 1
        return DraftProposal.from_lists([[self.token] * k for _ in seqs])

    def release(self, seq_id: int) -> None:
        self.released.append(seq_id)

    def prune(self, live_seq_ids: Any) -> int:
        return 0

    def reset(self) -> None:
        self.released.clear()

    def stats(self) -> dict[str, int | float]:
        return {"draft_calls": self.calls}


@pytest.fixture(scope="module")
def reference(tiny_qwen2_path: Path) -> dict:
    """Greedy continuations from the ordinary engine, the yardstick for every comparison."""
    with LLMEngine(engine_config(tiny_qwen2_path)) as engine:
        return completions(engine, PROMPTS, GREEDY)


# -- the headline property ------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 4])
def test_greedy_output_is_identical_to_the_plain_engine(
    tiny_qwen2_path: Path, reference: dict, k: int
) -> None:
    config = engine_config(
        tiny_qwen2_path,
        method="model",
        draft_model=str(tiny_qwen2_path),
        num_speculative_tokens=k,
    )
    with SpeculativeLLMEngine(config, drafter=self_drafter(tiny_qwen2_path)) as engine:
        assert completions(engine, PROMPTS, GREEDY) == reference
        stats = engine.spec_stats()
    assert stats.acceptance_rate == 1.0, "the draft model is the target; nothing may be rejected"
    assert stats.mean_accepted_len == float(k)
    assert stats.num_emitted == stats.num_verified_seqs * (k + 1)
    assert stats.num_target_forwards == stats.num_spec_steps


def test_speculation_saves_target_forward_passes(tiny_qwen2_path: Path) -> None:
    """A step emits more than one token per sequence, which is the whole point."""
    config = engine_config(
        tiny_qwen2_path,
        method="model",
        draft_model=str(tiny_qwen2_path),
        num_speculative_tokens=4,
    )
    with SpeculativeLLMEngine(config, drafter=self_drafter(tiny_qwen2_path)) as engine:
        completions(engine, PROMPTS, GREEDY)
        stats = engine.spec_stats()
        steps = int(engine.stats()["num_steps"])
    assert stats.mean_emitted_len == 5.0
    assert steps < GREEDY.max_tokens, "one decode step per token would mean no speculation"


def test_a_wholly_rejected_draft_still_produces_the_reference_output(
    tiny_qwen2_path: Path, reference: dict
) -> None:
    drafter = WrongDrafter()
    config = engine_config(tiny_qwen2_path, method="ngram", num_speculative_tokens=3)
    with SpeculativeLLMEngine(config, drafter=drafter) as engine:
        assert completions(engine, PROMPTS, GREEDY) == reference
        stats = engine.spec_stats()
    assert stats.num_drafted > 0
    assert stats.acceptance_rate < 0.5, "a constant guess must mostly be wrong"
    assert stats.num_emitted == stats.num_verified_seqs + stats.num_accepted


def test_ngram_speculation_survives_preemption(tiny_qwen2_path: Path) -> None:
    """Rollback and recompute-preemption have to coexist: both rewind a sequence."""
    prompts = (
        "Hello world there my friend",
        "The quick brown fox jumps over the lazy",
        "A B C D E F G H",
        "Paris is the capital of France and",
    )
    sampling = SamplingParams(max_tokens=40, temperature=0.0)
    with LLMEngine(engine_config(tiny_qwen2_path, blocks=6)) as base:
        expected = completions(base, prompts, sampling)
        assert int(base.stats()["num_preemptions"]) > 0, "the pool must be too small to fit all"
    config = engine_config(
        tiny_qwen2_path, blocks=6, method="ngram", num_speculative_tokens=4, ngram_min=1
    )
    with SpeculativeLLMEngine(config) as engine:
        assert completions(engine, prompts, sampling) == expected
        assert int(engine.stats()["num_preemptions"]) > 0
        assert engine.spec_stats().num_accepted > 0, "the n-gram drafter should land some guesses"
        engine.scheduler.check_invariants()
        engine.scheduler.block_manager.check_invariants()


def test_requests_admitted_while_others_decode_are_handled(
    tiny_qwen2_path: Path, reference: dict
) -> None:
    """A step that mixes a prefill with speculated decodes must treat each correctly."""
    config = engine_config(
        tiny_qwen2_path,
        method="model",
        draft_model=str(tiny_qwen2_path),
        num_speculative_tokens=3,
    )
    collected: dict[str, list[int]] = {}
    pending = list(enumerate(PROMPTS))
    step = 0
    with SpeculativeLLMEngine(config, drafter=self_drafter(tiny_qwen2_path)) as engine:
        while pending or engine.has_unfinished():
            if pending and step % 2 == 0:
                index, prompt = pending.pop(0)
                engine.add_request(f"gen-{index}", prompt, GREEDY)
            for output in engine.step():
                collected.setdefault(output.request_id, []).extend(output.new_token_ids)
            step += 1
    assert collected == reference


def test_sampled_requests_are_bounded_by_max_tokens(tiny_qwen2_path: Path) -> None:
    """Rejection sampling emits a variable number of tokens; the limit still holds exactly."""
    sampling = SamplingParams(max_tokens=10, temperature=0.9, top_p=0.95, seed=1234)
    config = engine_config(
        tiny_qwen2_path,
        method="model",
        draft_model=str(tiny_qwen2_path),
        num_speculative_tokens=4,
    )
    with SpeculativeLLMEngine(config, drafter=self_drafter(tiny_qwen2_path)) as engine:
        outputs = engine.generate(list(PROMPTS), sampling)
    assert [output.output_tokens for output in outputs] == [10] * len(PROMPTS)
    assert all(output.finish_reason == "length" for output in outputs)


# -- configuration ----------------------------------------------------------------------------


def test_speculative_config_requires_a_draft_model_for_the_model_method() -> None:
    with pytest.raises(ValueError, match="needs a draft model"):
        SpeculativeConfig.from_mapping({"method": "model"})


def test_speculative_config_accepts_the_short_option_names() -> None:
    config = SpeculativeConfig.from_mapping({"model": "some/draft", "k": 6})
    assert config.draft_model == "some/draft"
    assert config.num_speculative_tokens == 6
    assert config.method == "model"


def test_speculative_config_rejects_unknown_and_contradictory_options() -> None:
    with pytest.raises(ValueError, match="empty"):
        SpeculativeConfig.from_mapping({})
    with pytest.raises(ValueError):
        SpeculativeConfig.from_mapping({"method": "ngram", "nonsense": 1})
    with pytest.raises(ValueError, match="below ngram_min"):
        SpeculativeConfig.from_mapping({"method": "ngram", "ngram_min": 4, "ngram_max": 2})


def test_a_draft_model_with_another_vocabulary_is_refused(
    tiny_qwen2_path: Path, tiny_llama_path: Path
) -> None:
    drafter = ModelDrafter.from_pretrained(
        str(tiny_llama_path), dtype=torch.float32, device="cpu", num_blocks=16, block_size=16
    )
    config = engine_config(tiny_qwen2_path, method="model", draft_model=str(tiny_llama_path))
    with pytest.raises(ValueError, match="vocabulary"):
        SpeculativeLLMEngine(config, drafter=drafter)


def test_max_batch_size_turns_speculation_off_for_wide_steps(
    tiny_qwen2_path: Path, reference: dict
) -> None:
    config = engine_config(
        tiny_qwen2_path, method="ngram", num_speculative_tokens=4, max_batch_size=1
    )
    with SpeculativeLLMEngine(config) as engine:
        assert completions(engine, PROMPTS, GREEDY) == reference
        stats = engine.spec_stats()
    assert stats.num_draft_calls == 0, "a four-sequence batch is above the limit"
    assert stats.num_spec_steps == 0


# -- drafters ----------------------------------------------------------------------------------


def test_ngram_lookup_returns_what_followed_the_latest_repeat() -> None:
    assert find_ngram_continuation([1, 2, 3, 4, 1, 2, 3], num_tokens=3) == [4, 1, 2]
    assert find_ngram_continuation([9, 9, 9, 9], num_tokens=3) == [9]
    assert find_ngram_continuation([1, 2, 3, 4, 5], num_tokens=3) == []
    assert find_ngram_continuation([1, 2, 3, 1, 2], num_tokens=2, min_ngram=3) == []


def test_ngram_drafter_proposes_per_sequence_and_counts_its_hits() -> None:
    repeating = Sequence(seq_id=0, request_id="r0", prompt_token_ids=[5, 6, 7, 5, 6])
    novel = Sequence(seq_id=1, request_id="r1", prompt_token_ids=[11, 12, 13, 14])
    drafter = NgramDrafter(min_ngram=2, max_ngram=3)
    proposal = drafter.propose([repeating, novel], 2)
    # The tail [5, 6] last occurred at the start, and [7, 5] followed it there.
    assert proposal.tokens_for(0) == [7, 5]
    assert proposal.tokens_for(1) == []
    assert proposal.lengths == (2, 0)
    assert proposal.probs is None
    stats = drafter.stats()
    assert stats["draft_lookups"] == 2
    assert stats["draft_lookup_hits"] == 1
    drafter.release(0)
    assert drafter.stats()["draft_sequences"] == 1
    assert drafter.stats()["draft_lookups"] == 2, "lifetime counters survive a release"


def test_model_drafter_mirrors_a_sequence_and_frees_it(tiny_qwen2_path: Path) -> None:
    drafter = self_drafter(tiny_qwen2_path, blocks=16)
    seq = Sequence(seq_id=3, request_id="r3", prompt_token_ids=[1, 2, 3, 4, 5])
    proposal = drafter.propose([seq], 3)
    assert proposal.lengths == (3,)
    assert len(proposal.tokens_for(0)) == 3
    assert drafter.num_tracked == 1
    free_before = drafter.block_manager.num_free_blocks
    drafter.release(seq.seq_id)
    assert drafter.num_tracked == 0
    assert drafter.block_manager.num_free_blocks > free_before


def test_model_drafter_rewinds_when_the_target_kept_only_part_of_a_draft(
    tiny_qwen2_path: Path,
) -> None:
    """The mirror must follow the target's tokens, not the ones it guessed."""
    drafter = self_drafter(tiny_qwen2_path, blocks=16)
    seq = Sequence(seq_id=4, request_id="r4", prompt_token_ids=[10, 11, 12, 13])
    first = drafter.propose([seq], 4).tokens_for(0)
    assert len(first) == 4
    # The target accepted the first draft token and then emitted something else entirely.
    seq.output_token_ids.extend([first[0], 99])
    second = drafter.propose([seq], 2).tokens_for(0)
    assert len(second) == 2
    mirror_tokens = list(drafter._mirrors[seq.seq_id].token_ids)  # noqa: SLF001 - white box
    assert mirror_tokens[: seq.num_tokens] == seq.token_ids


def test_drafters_satisfy_the_protocol(tiny_qwen2_path: Path) -> None:
    assert isinstance(NgramDrafter(), Drafter)
    assert isinstance(self_drafter(tiny_qwen2_path, blocks=8), Drafter)
    assert isinstance(WrongDrafter(), Drafter)


def test_draft_proposal_pads_and_trims() -> None:
    proposal = DraftProposal.from_lists([[1, 2, 3], [4], []])
    assert proposal.lengths == (3, 1, 0)
    assert proposal.max_drafts == 3
    assert proposal.total_drafts == 4
    assert proposal.tokens_for(1) == [4]
    assert not proposal.is_empty
    assert DraftProposal.empty(2).is_empty
    wide = torch.zeros((3, 5, 7))
    assert DraftProposal.from_lists([[1, 2], [3], []], probs=wide).probs is not None


# -- lifecycle and statistics -------------------------------------------------------------------


def test_abort_releases_the_drafter_state(tiny_qwen2_path: Path) -> None:
    drafter = self_drafter(tiny_qwen2_path)
    config = engine_config(
        tiny_qwen2_path,
        method="model",
        draft_model=str(tiny_qwen2_path),
        num_speculative_tokens=2,
    )
    with SpeculativeLLMEngine(config, drafter=drafter) as engine:
        engine.add_request("keep", "Hello world", GREEDY)
        engine.add_request("drop", "Paris is the capital of", GREEDY)
        engine.step()  # prefill
        engine.step()  # first speculative decode; both mirrors now exist
        assert drafter.num_tracked == 2
        assert engine.abort("drop")
        assert drafter.num_tracked == 1
        assert engine.abort("missing") is False


def test_stats_report_acceptance_and_the_drafter(tiny_qwen2_path: Path) -> None:
    config = engine_config(
        tiny_qwen2_path,
        method="model",
        draft_model=str(tiny_qwen2_path),
        num_speculative_tokens=2,
    )
    with SpeculativeLLMEngine(config, drafter=self_drafter(tiny_qwen2_path)) as engine:
        completions(engine, PROMPTS[:2], GREEDY)
        stats = engine.stats()
    assert stats["drafter"] == "model"
    assert stats["num_speculative_tokens"] == 2
    assert stats["acceptance_rate"] == 1.0
    assert stats["num_generated_tokens"] == 2 * GREEDY.max_tokens
    assert "draft_forwards" in stats
    assert "kv_utilization" in stats, "the base engine's statistics are still there"


def test_closing_the_engine_releases_the_draft_pool(tiny_qwen2_path: Path) -> None:
    drafter = self_drafter(tiny_qwen2_path)
    config = engine_config(
        tiny_qwen2_path,
        method="model",
        draft_model=str(tiny_qwen2_path),
        num_speculative_tokens=2,
    )
    engine = SpeculativeLLMEngine(config, drafter=drafter)
    completions(engine, PROMPTS[:1], GREEDY)
    engine.close()
    engine.close()  # idempotent
    assert drafter.num_tracked == 0
    assert drafter.block_manager.num_free_blocks == drafter.block_manager.allocator.num_total


# -- the benchmark scenario ----------------------------------------------------------------------


@pytest.fixture
def tiny_profiles(tmp_path: Path, tiny_qwen2_path: Path) -> Path:
    """A profiles file whose ``spec_decode`` scenario points at the cached tiny model.

    Derived from the shipped ``dev-2060`` profile rather than written from scratch, so the
    test exercises the same schema the real profiles use and fails if that schema changes.
    """
    import yaml

    from turboserve.bench.profiles import default_profiles_path

    document = yaml.safe_load(default_profiles_path().read_text(encoding="utf-8"))
    profile = document["profiles"]["dev-2060"]
    profile["device"] = "cpu"
    for scenario in profile["scenarios"].values():
        scenario["dtype"] = "float32"
        if "model" in scenario:
            scenario["model"] = str(tiny_qwen2_path)
    work = profile["scenarios"]["spec_decode"]
    work["num_requests"] = 3
    work["input_tokens"] = {"min": 8, "max": 12}
    work["output_tokens"] = {"min": 4, "max": 4}
    work["concurrencies"] = [2]
    work["speculative_tokens"] = [2]
    work["pairs"] = [
        {
            "name": "tiny-self",
            "target": str(tiny_qwen2_path),
            "draft": str(tiny_qwen2_path),
            "drafter": "model",
        },
        {"name": "tiny-ngram", "target": str(tiny_qwen2_path), "draft": None, "drafter": "ngram"},
    ]
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "profiles": {"tiny": profile}}))
    return path


def test_plan_arms_puts_the_baseline_first_for_every_pair(tiny_profiles: Path) -> None:
    from turboserve.bench.profiles import load_profile
    from turboserve.bench.scenarios.spec_decode import plan_arms

    work = load_profile("tiny", path=tiny_profiles).scenarios.spec_decode
    arms = plan_arms(work)
    assert [arm.label for arm in arms] == [
        "tiny-self / target only",
        "tiny-self / k=2",
        "tiny-ngram / target only",
        "tiny-ngram / ngram k=2",
    ]
    assert all(arm.concurrency == 2 for arm in arms)
    assert {arm.baseline_label for arm in arms} == {
        "tiny-self / target only",
        "tiny-ngram / target only",
    }


def test_plan_arms_filters_and_validates(tiny_profiles: Path) -> None:
    from turboserve.bench.profiles import load_profile
    from turboserve.bench.scenarios.common import ScenarioError
    from turboserve.bench.scenarios.spec_decode import plan_arms

    work = load_profile("tiny", path=tiny_profiles).scenarios.spec_decode
    arms = plan_arms(work, pairs=["tiny-ngram"], k_values=[1, 3], concurrencies=[4, 8])
    assert len(arms) == 2 * 3  # two concurrencies, baseline plus two k values
    assert {arm.concurrency for arm in arms} == {4, 8}
    with pytest.raises(ScenarioError, match="unknown pair"):
        plan_arms(work, pairs=["nope"])
    with pytest.raises(ScenarioError, match="must be positive"):
        plan_arms(work, k_values=[0])


def test_arm_translates_into_engine_configuration(tiny_profiles: Path) -> None:
    from turboserve.bench.profiles import load_profile
    from turboserve.bench.scenarios.spec_decode import plan_arms

    work = load_profile("tiny", path=tiny_profiles).scenarios.spec_decode
    baseline, model_arm, _, ngram_arm = plan_arms(work)
    assert baseline.is_baseline and baseline.speculative_config() is None
    assert model_arm.speculative_config() == {
        "method": "model",
        "draft_model": model_arm.pair.target,
        "num_speculative_tokens": 2,
    }
    assert ngram_arm.speculative_config() == {"method": "ngram", "num_speculative_tokens": 2}
    assert model_arm.to_dict()["pair"] == "tiny-self"
    assert ngram_arm.baseline_label == "tiny-ngram / target only"


def test_build_backend_selects_the_speculative_engine(tiny_profiles: Path) -> None:
    """The arm's settings really reach the engine, not just the result file's config block."""
    from turboserve.bench.profiles import load_profile
    from turboserve.bench.scenarios.common import EngineOptions
    from turboserve.bench.scenarios.spec_decode import build_backend, plan_arms

    work = load_profile("tiny", path=tiny_profiles).scenarios.spec_decode
    options = EngineOptions(
        dtype="float32", device="cpu", num_blocks=32, max_num_seqs=2, max_num_batched_tokens=128
    )
    baseline, _, _, ngram_arm = plan_arms(work)
    backend = build_backend(ngram_arm, options, local_files_only=True)
    stats = backend.stats()
    assert stats["drafter"] == "ngram"
    assert stats["num_speculative_tokens"] == 2
    plain = build_backend(baseline, options, local_files_only=True)
    assert "drafter" not in plain.stats(), "the baseline arm must not speculate"
    remote = build_backend(baseline, options, url="http://127.0.0.1:9/v1")
    assert remote.name == "vllm"


async def test_run_arm_writes_a_labelled_result_with_speculation_counters(
    tiny_profiles: Path, tmp_path: Path
) -> None:
    """The scenario's contract with the report renderer, end to end on the tiny model."""
    from turboserve.bench.profiles import load_profile
    from turboserve.bench.records import RunResult
    from turboserve.bench.scenarios.common import EngineOptions, build_prompt_pool
    from turboserve.bench.scenarios.spec_decode import plan_arms, run_arm

    profile = load_profile("tiny", path=tiny_profiles)
    arm = next(item for item in plan_arms(profile.scenarios.spec_decode) if item.method == "ngram")
    prompts = build_prompt_pool(
        count=3,
        input_tokens=(8, 8),
        output_tokens=(4, 4),
        seed=profile.seed,
        tenants=tuple(profile.tenants),
        tokenizer=None,
    )
    outcome = await run_arm(
        arm,
        profile,
        prompts,
        options=EngineOptions(
            dtype="float32",
            device="cpu",
            num_blocks=32,
            max_num_seqs=2,
            max_num_batched_tokens=128,
        ),
        num_warmup=1,
        local_files_only=True,
        out_dir=tmp_path,
    )
    assert outcome.path.parent == tmp_path
    assert outcome.num_requests == len(prompts)
    assert outcome.error_rate == 0.0
    assert outcome.derived["drafter"] == "ngram"
    assert "acceptance_rate" in outcome.derived
    saved = RunResult.load(outcome.path)
    assert saved.scenario == "spec_decode"
    assert saved.config["label"] == arm.label
    assert saved.config["baseline_label"] == arm.baseline_label
    assert saved.config["load"]["concurrency"] == arm.concurrency
    assert saved.config["num_speculative_tokens"] == 2
    assert saved.summary["derived"]["drafter"] == "ngram"


def test_the_dry_run_lists_arms_without_loading_a_model(tiny_profiles: Path) -> None:
    from typer.testing import CliRunner

    from turboserve.bench.cli import bench_app

    # Invoked through the real benchmark application, so the test also pins the documented
    # spelling of the command: `turboserve bench spec-decode --profile ...`.
    result = CliRunner().invoke(
        bench_app,
        ["spec-decode", "--profile", "tiny", "--profiles", str(tiny_profiles), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "tiny-self" in result.output
    assert "nothing was run" in result.output


def test_the_cli_refuses_contradictory_backend_options(tiny_profiles: Path) -> None:
    from typer.testing import CliRunner

    from turboserve.bench.cli import bench_app

    runner = CliRunner()
    missing_url = runner.invoke(
        bench_app,
        ["spec-decode", "--profile", "tiny", "--profiles", str(tiny_profiles), "--backend", "vllm"],
    )
    assert missing_url.exit_code != 0
    stray_url = runner.invoke(
        bench_app,
        [
            "spec-decode",
            "--profile",
            "tiny",
            "--profiles",
            str(tiny_profiles),
            "--url",
            "http://127.0.0.1:8000/v1",
        ],
    )
    assert stray_url.exit_code != 0


def test_the_scenario_is_registered_with_the_bench_application() -> None:
    """`turboserve bench spec-decode` must actually exist once the bench app is built."""
    from turboserve.bench.cli import bench_app

    names = {group.name for group in bench_app.registered_groups}
    names |= {command.name for command in bench_app.registered_commands}
    assert "spec-decode" in names
