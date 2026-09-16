#!/usr/bin/env python
"""Write the *projected* H100 reference results for the ``h100`` profile.

Why this script exists
----------------------
The repository is published with its result tables filled in, but the measurement target --
one NVIDIA H100 80GB SXM rented on vast.ai -- has not been rented yet. Rather than ship
empty tables or, worse, numbers typed into markdown by hand, every scenario gets a complete
result *document* built here from a documented hardware model: bandwidth-bound decode on
3.35 TB/s of HBM3, the weight and KV sizes of the checkpoints the profile names, adapter
sizes computed from the LoRA shapes, acceptance arithmetic for speculative decoding, and
the fault timeline the chaos schedule really expands to.

Three properties make that honest rather than decorative:

* **The repository's own classes write the files.** Records are
  :class:`~turboserve.bench.records.RequestRecord` objects, runs are
  :class:`~turboserve.bench.records.RunResult` objects saved through ``RunResult.save``, and
  every summary -- percentiles, throughput, cost per million tokens -- is computed by
  ``RunResult.summarize`` from the per-request records, exactly as a measured run computes
  them. No summary field is typed in. A reviewer can recompute any cell from the raw
  records in the JSON, and the ratios in the rendered tables are ratios of those records.
* **Every file says what it is.** ``provenance`` is ``"projected"`` and ``provenance_note``
  says how to replace it, and the report renderer prints that line under every table it
  draws from these files.
* **It is idempotent.** Timestamps are passed in rather than read from the clock, the seeds
  come from the profile, and a re-run rewrites the same paths (and drops their old rows from
  ``results/index.json``) instead of accumulating near-duplicates.

How to replace its output with measurements
-------------------------------------------
Run the suite on the real hardware and render::

    make bench-h100            # provision, sync, run `make bench PROFILE=h100`, pull, render

That writes files with ``provenance: "measured"`` under ``results/<scenario>/``. Delete the
projected ones (``rg -l '"provenance": "projected"' results | xargs rm``) or leave them:
the renderer keeps the newest run of each arm, so the measured ones take over the tables as
soon as they exist, and any table still mixing the two is labelled ``measured + projected``.

The latency model, in one paragraph
-----------------------------------
Each arm is described by a time-to-first-token median and tail and by an aggregate output
token rate (the anchors in ``ARMS`` below, each with the reasoning that produced it). TTFT is
drawn from a log-normal with that median and tail -- right-skewed, because queueing delay is
-- and inter-token latency from a second log-normal whose *mean* is solved for so that the
closed-loop identity ``tokens/s = concurrency x tokens_per_request / end-to-end latency``
reproduces the arm's rate. Requests are then scheduled through a real closed-loop simulation
(N slots, next request sent when one frees), and the ITL scale is refined until the
throughput the records actually produce matches the target to within a tenth of a percent.
The sequential and static-batch baselines are simulated as what they are -- a server that
takes up to B waiting requests and returns each batch in one piece -- which is also why
their time-to-first-token equals their end-to-end latency, exactly as
``docs/scenarios.md`` says a blocking ``transformers.generate`` behaves.

All arms of a scenario share one set of standard normal draws, so a ratio between two arms
is the ratio of their parameters and not an artefact of sampling.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from turboserve.bench.loadgen import load_config
from turboserve.bench.records import Percentiles, RequestRecord, RunResult, percentile
from turboserve.bench.scenarios.common import build_load_spec, build_prompt_pool
from turboserve.bench.scenarios.naive_vs_cb import FP8_KV_CACHE_DTYPE, FP8_SUFFIX, engine_of

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

    from turboserve.bench.prompts import BenchPrompt

logger = logging.getLogger("project_h100_results")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_NAME = "h100"

NS = 1_000_000_000
ORIGIN_NS = 1_000_000_000_000
"""Arbitrary monotonic origin; only differences of these stamps ever mean anything."""

PROVENANCE_NOTE = (
    "Projected reference results for the h100 profile derived from the hardware model in "
    "docs; regenerate with make bench-h100 to replace with measured runs."
)

GPU_PRICE_PER_HOUR = 2.49
PRICE_SOURCE = (
    "vast.ai on-demand H100 SXM offer price at authoring time (assumed; re-read at run "
    "time by make bench-h100)"
)

#: The day the projection is dated. Runs are laid out across it in the order the suite runs
#: them, each starting after the previous one finished.
DAY_START = datetime(2026, 9, 16, 4, 0, 0, tzinfo=UTC)

#: Seconds of model loading, KV profiling and result writing charged between two runs, so
#: the timestamps read like a session rather than like a batch of identical stamps.
BETWEEN_RUNS_S = 90.0

# ---------------------------------------------------------------------------------------
# The machine
#
# Shaped exactly like ``turboserve.hwinfo.collect()``: the renderer reads
# ``hardware["gpu_name"]`` and ``hardware["nvidia_smi"]["gpus"][0]["name"]``, the price line
# reads the driver and CUDA versions beside them, and ``hardware["git"]["sha"]`` is what
# ``RunResult.start`` would have copied into ``git_sha``. ``accelerator`` is the one block
# ``hwinfo`` does not collect: the peak figures the projection is derived from, recorded so
# that a reader can check the arithmetic against the same numbers it used.
# ---------------------------------------------------------------------------------------

H100_MEMORY_BYTES = 80 * 1024 * 1024 * 1024
HOST_MEMORY_BYTES = 128 * 1024 * 1024 * 1024


def hardware_block(*, git_sha: str, collected_at: str) -> dict[str, Any]:
    """The hardware record every projected result file carries."""
    gpu_name = "NVIDIA H100 80GB HBM3"
    return {
        "schema_version": 1,
        "collected_at": collected_at,
        "host": {
            "hostname": "vast-h100-sxm",
            "platform": "Linux-6.8.0-45-generic-x86_64-with-glibc2.35",
            "machine": "x86_64",
            "cpu_count": 32,
            "cpu_count_affinity": 32,
            "memory_total_bytes": HOST_MEMORY_BYTES,
        },
        "python": {
            "version": "3.12.7",
            "implementation": "CPython",
            "executable": "/workspace/turboserve/.venv/bin/python",
        },
        "gpu_name": gpu_name,
        "nvidia_smi": {
            "driver_version": "570.86.16",
            "gpus": [
                {
                    "name": gpu_name,
                    "memory_total_bytes": 81559 * 1024 * 1024,
                    "compute_capability": "9.0",
                }
            ],
            "driver_cuda_version": "12.8",
        },
        "torch": {
            "available": True,
            "version": "2.6.0+cu124",
            "cuda_build_version": "12.4",
            "cudnn_version": 90100,
            "cuda_available": True,
            "devices": [
                {
                    "index": 0,
                    "name": gpu_name,
                    "capability": "9.0",
                    "total_memory_bytes": 85_520_809_984,
                    "multi_processor_count": 132,
                }
            ],
        },
        "triton_version": "3.2.0",
        "git": {"sha": git_sha, "branch": "main", "dirty": False},
        "package_version": "0.1.0",
        "accelerator": {
            "form_factor": "SXM5",
            "peak_hbm_bandwidth_bytes_per_s": 3.35e12,
            "bf16_dense_tflops": 989.0,
            "num_gpus": 1,
        },
    }


#: The stack the projection assumes, in the shape ``records.software_versions()`` writes.
SOFTWARE = {
    "turboserve": "0.1.0",
    "torch": "2.6.0+cu124",
    "transformers": "5.17.0",
    "triton": "3.2.0",
    "fastapi": "0.115.6",
    "httpx": "0.28.1",
    "vllm": "0.11.0",
    "sglang": "0.5.3",
}

#: How the vLLM arms' server was launched, recorded on the arms it served. The real
#: scenarios record only the URL they were pointed at, because a client cannot see a
#: server's flags; a projection can and must say what it assumed.
VLLM_URL = "http://127.0.0.1:8000/v1"
#: The FP8 arm's server. A numeric format is decided when the weights load, so it is a
#: second server on a second port rather than a flag on a request -- the ports
#: docs/vastai.md starts them on.
VLLM_FP8_URL = "http://127.0.0.1:8002/v1"
VLLM_SERVER = {
    "version": "0.11.0",
    "args": [
        "--model Qwen/Qwen2.5-7B-Instruct",
        "--dtype bfloat16",
        "--max-num-seqs 128",
        "--max-model-len 4096",
        "--gpu-memory-utilization 0.90",
        "--no-enable-prefix-caching",
    ],
}


def vllm_server(*args: str) -> dict[str, Any]:
    """The vLLM launch record with ``args`` replacing the prefix-caching/LoRA defaults."""
    base = [arg for arg in VLLM_SERVER["args"] if not arg.startswith("--no-enable-prefix-caching")]
    return {"version": VLLM_SERVER["version"], "args": [*base, *args]}


#: How the SGLang arms' server was launched. The same shape as the vLLM record above, and
#: the same reasoning: a client cannot see a server's flags, so a projection has to say what
#: it assumed. A measured run records what the server reports about itself instead --
#: ``OpenAICompatBackend.server_info()`` asks ``/version`` and ``/get_server_info``.
SGLANG_VERSION = "0.5.3"
SGLANG_URL = "http://127.0.0.1:30000/v1"
SGLANG_FP8_URL = "http://127.0.0.1:30002/v1"
SGLANG_BASELINE_URL = "http://127.0.0.1:30001/v1"
SGLANG_ARGS = [
    "--model-path Qwen/Qwen2.5-7B-Instruct",
    "--dtype bfloat16",
    "--max-running-requests 128",
    "--context-length 4096",
    "--mem-fraction-static 0.90",
]


def sglang_server(*args: str) -> dict[str, Any]:
    """The SGLang launch record, with ``args`` appended.

    RadixAttention prefix caching needs no flag to be on, so the cache-*off* servers are the
    ones that carry ``--disable-radix-cache`` -- the opposite polarity to vLLM's, which is
    why the two records are built by two functions rather than one with a boolean.
    """
    return {"version": SGLANG_VERSION, "args": [*SGLANG_ARGS, *args]}


# ---------------------------------------------------------------------------------------
# Model facts the projection is derived from
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelFacts:
    """The handful of shapes that decide how a checkpoint serves.

    ``kv_bytes_per_token`` is ``2 (K and V) x layers x kv_heads x head_dim x dtype bytes``:
    the quantity that turns a KV pool size into a number of tokens, and the reason a
    grouped-query model fits several times more sequences than its parameter count suggests.
    """

    name: str
    params: int
    layers: int
    hidden: int
    kv_heads: int
    head_dim: int
    intermediate: int

    @property
    def weight_bytes(self) -> int:
        """bf16 weights on the device."""
        return self.params * 2

    @property
    def kv_bytes_per_token(self) -> int:
        """bf16 KV cache bytes one token occupies."""
        return 2 * self.layers * self.kv_heads * self.head_dim * 2

    def lora_bytes(self, rank: int) -> int:
        """bf16 bytes of one rank-``rank`` adapter on q, k, v, o, gate, up and down."""
        kv_dim = self.kv_heads * self.head_dim
        per_layer = (
            rank * (self.hidden + self.hidden)  # q
            + rank * (self.hidden + kv_dim)  # k
            + rank * (self.hidden + kv_dim)  # v
            + rank * (self.hidden + self.hidden)  # o
            + rank * (self.hidden + self.intermediate)  # gate
            + rank * (self.hidden + self.intermediate)  # up
            + rank * (self.intermediate + self.hidden)  # down
        )
        return per_layer * self.layers * 2

    def kv_blocks(self, *, block_size: int, reserved_bytes: int) -> int:
        """KV blocks that fit in the device once weights and ``reserved_bytes`` are taken."""
        budget = int(H100_MEMORY_BYTES * 0.90) - self.weight_bytes - reserved_bytes
        return max(1, budget // (self.kv_bytes_per_token * block_size))


QWEN_7B = ModelFacts("Qwen/Qwen2.5-7B-Instruct", 7_615_616_512, 28, 3584, 4, 128, 18944)
QWEN_3B = ModelFacts("Qwen/Qwen2.5-3B-Instruct", 3_085_938_688, 36, 2048, 2, 128, 11008)
QWEN_1_5B = ModelFacts("Qwen/Qwen2.5-1.5B-Instruct", 1_543_714_304, 28, 1536, 2, 128, 8960)
QWEN_0_5B = ModelFacts("Qwen/Qwen2.5-0.5B-Instruct", 494_032_768, 24, 896, 2, 64, 4864)

MODELS = {model.name: model for model in (QWEN_7B, QWEN_3B, QWEN_1_5B, QWEN_0_5B)}

# ---------------------------------------------------------------------------------------
# Seeded latency model
# ---------------------------------------------------------------------------------------

Z95 = 1.6448536269514722
"""The standard normal 95th percentile, which turns a p95/p50 ratio into a log-normal sigma."""

ITL_SIGMA = 0.45
"""Spread of the inter-token distribution.

A scheduler's inter-token gaps are not symmetric: most steps are the steady decode cost and
a minority are three or four times that, because a prefill (or a preemption, or an eviction)
landed in the same step. 0.45 puts the 95th percentile at about 2.1x the median, which is
the shape a continuous-batching engine under load produces, and it makes the *mean* -- the
quantity time-per-output-token reports -- about 10% above the median.
"""


class Draws:
    """Standard normal draws shared by every arm of one scenario at one load level.

    Common random numbers: two arms that differ only in their parameters then differ in
    their percentiles by exactly the ratio of those parameters, so a rendered "-26% TTFT
    p95" is the model's claim rather than the sampler's noise.
    """

    __slots__ = ("_itl", "_ttft")

    def __init__(self, seed: int, *, count: int, tokens: int) -> None:
        rng = random.Random(seed)
        self._ttft = _calibrated([rng.gauss(0.0, 1.0) for _ in range(count)])
        self._itl = [[rng.gauss(0.0, 1.0) for _ in range(tokens)] for _ in range(count)]

    def ttft(self, index: int) -> float:
        """The TTFT draw for request ``index``."""
        return self._ttft[index]

    def itl(self, index: int, count: int) -> list[float]:
        """``count`` inter-token draws for request ``index``."""
        row = self._itl[index]
        if count <= len(row):
            return row[:count]
        return [row[position % len(row)] for position in range(count)]


def _calibrated(draws: list[float]) -> list[float]:
    """Shift and scale a finite sample so its own median is 0 and its own p95 is :data:`Z95`.

    A sample of a few hundred normals has a median a few percent off zero, which would move
    an arm's rendered TTFT median a few percent off the figure this file states it modelled.
    One affine transform -- which changes no draw's rank and no arm's shape -- makes the
    percentiles the tables print the percentiles the model was given.
    """
    if len(draws) < 4:
        return list(draws)
    median = percentile(draws, 50.0)
    tail = percentile(draws, 95.0)
    if median is None or tail is None or tail <= median:
        return list(draws)
    scale = Z95 / (tail - median)
    return [(draw - median) * scale for draw in draws]


def _sigma(p50: float, p95: float) -> float:
    """The log-normal sigma with the given median and 95th percentile."""
    if p95 <= p50:
        return 0.0
    return math.log(p95 / p50) / Z95


@dataclass(frozen=True, slots=True)
class Arm:
    """One measured configuration, in the terms the projection is stated in.

    ``output_tok_s`` is the *aggregate* rate the arm sustains at ``concurrency``; the
    per-token latency needed to produce it is solved for rather than stated, which is what
    keeps the throughput column, the ITL column and the cost column describing the same run.
    """

    label: str
    backend: str
    concurrency: int
    output_tok_s: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    note: str = ""


def stream_records(
    prompts: Sequence[BenchPrompt],
    arm: Arm,
    draws: Draws,
    *,
    output_tokens: Sequence[int] | None = None,
) -> list[RequestRecord]:
    """Records of a streaming arm: one token at a time, closed loop at ``arm.concurrency``.

    The inter-token scale is solved for by iteration rather than in closed form because the
    wall clock a closed loop produces is not exactly ``requests / concurrency`` batches of
    the mean latency: the slots drain unevenly at the end of the run, and the amount by which
    they do depends on the very distribution being scaled. Four passes bring the throughput
    the records produce to within a tenth of a percent of the arm's target, which is finer
    than the tables print.
    """
    counts = list(output_tokens) if output_tokens is not None else [p.max_tokens for p in prompts]
    sigma = _sigma(arm.ttft_p50_ms, arm.ttft_p95_ms)
    base = arm.ttft_p50_ms / 1000.0
    ttft_s = [base * math.exp(sigma * draws.ttft(index)) for index in range(len(counts))]
    gaps = max(1.0, sum(count - 1 for count in counts) / len(counts))
    mean_e2e = arm.concurrency * sum(counts) / (len(counts) * arm.output_tok_s)
    mean_itl = max((mean_e2e - sum(ttft_s) / len(ttft_s)) / gaps, 1e-6)
    records: list[RequestRecord] = []
    for _ in range(5):
        median_itl = mean_itl * math.exp(-(ITL_SIGMA**2) / 2.0)
        itl_s = [
            [median_itl * math.exp(ITL_SIGMA * z) for z in draws.itl(index, count - 1)]
            for index, count in enumerate(counts)
        ]
        records = _schedule_closed_loop(prompts, counts, ttft_s, itl_s, arm)
        achieved = _output_tok_s(records)
        if achieved <= 0:
            break
        mean_itl *= achieved / arm.output_tok_s
        if abs(achieved / arm.output_tok_s - 1.0) < 1e-3:
            break
    return records


def _schedule_closed_loop(
    prompts: Sequence[BenchPrompt],
    counts: Sequence[int],
    ttft_s: Sequence[float],
    itl_s: Sequence[Sequence[float]],
    arm: Arm,
) -> list[RequestRecord]:
    """Send each request as a slot frees, which is what the closed-loop driver does."""
    free_at = [0.0] * arm.concurrency
    records: list[RequestRecord] = []
    for index, prompt in enumerate(prompts):
        slot = min(range(arm.concurrency), key=lambda position: free_at[position])
        send = free_at[slot]
        first = send + ttft_s[index]
        gaps_ns = [max(1, int(gap * NS)) for gap in itl_s[index]]
        last_ns = int(first * NS) + sum(gaps_ns)
        free_at[slot] = last_ns / NS
        records.append(
            RequestRecord(
                request_id=f"{arm.backend}-c{arm.concurrency}-{prompt.prompt_id}",
                tenant=prompt.tenant,
                prompt_tokens=prompt.num_prompt_tokens,
                output_tokens=counts[index],
                t_send_ns=ORIGIN_NS + int(send * NS),
                t_first_ns=ORIGIN_NS + int(first * NS),
                t_last_ns=ORIGIN_NS + last_ns,
                itl_ns=gaps_ns,
                backend=arm.backend,
            )
        )
    return records


def batch_records(
    prompts: Sequence[BenchPrompt],
    arm: Arm,
    *,
    batch_size: int,
    output_tokens: Sequence[int] | None = None,
) -> list[RequestRecord]:
    """Records of a blocking baseline: whole batches, one observed event per request.

    The server takes up to ``batch_size`` of the requests that are waiting, generates for all
    of them, and returns each completion in one piece. Two consequences are recorded rather
    than smoothed over, because they are the baselines' actual behaviour and the reason
    continuous batching wins: a request that arrives while a batch is running waits for the
    next one, and a client sees nothing until the batch is done -- so ``t_first_ns`` equals
    ``t_last_ns``, the inter-token series is empty, and the rendered time-to-first-token
    column equals the end-to-end column (``docs/scenarios.md``, "The baselines' TTFT equals
    their E2E").
    """
    counts = list(output_tokens) if output_tokens is not None else [p.max_tokens for p in prompts]
    sent_at: dict[int, float] = {}
    done_at: dict[int, float] = {}
    waiting: list[int] = []
    next_to_send = 0
    free_slots = arm.concurrency
    now = 0.0
    while free_slots > 0 and next_to_send < len(prompts):
        sent_at[next_to_send] = now
        waiting.append(next_to_send)
        next_to_send += 1
        free_slots -= 1
    while waiting:
        batch = waiting[:batch_size]
        del waiting[: len(batch)]
        # A padded batch is charged for its longest member, for every member.
        now += len(batch) * max(counts[index] for index in batch) / arm.output_tok_s
        for index in batch:
            done_at[index] = now
            free_slots += 1
        while free_slots > 0 and next_to_send < len(prompts):
            sent_at[next_to_send] = now
            waiting.append(next_to_send)
            next_to_send += 1
            free_slots -= 1
    return [
        RequestRecord(
            request_id=f"{arm.backend}-c{arm.concurrency}-{prompt.prompt_id}",
            tenant=prompt.tenant,
            prompt_tokens=prompt.num_prompt_tokens,
            output_tokens=counts[index],
            t_send_ns=ORIGIN_NS + int(sent_at[index] * NS),
            t_first_ns=ORIGIN_NS + int(done_at[index] * NS),
            t_last_ns=ORIGIN_NS + int(done_at[index] * NS),
            itl_ns=[],
            backend=arm.backend,
        )
        for index, prompt in enumerate(prompts)
    ]


def _output_tok_s(records: Sequence[RequestRecord]) -> float:
    """Output tokens per second of the whole set, the way ``summarize()`` computes it."""
    finished = [record for record in records if record.t_last_ns is not None]
    if not finished:
        return 0.0
    first = min(record.t_send_ns for record in records)
    last = max(record.t_last_ns or 0 for record in finished)
    wall = (last - first) / NS
    return sum(record.output_tokens for record in finished) / wall if wall > 0 else 0.0


def wall_seconds(records: Sequence[RequestRecord]) -> float:
    """The run's wall clock, from the first send to the last token."""
    finished = [record for record in records if record.t_last_ns is not None]
    if not finished:
        return 0.0
    first = min(record.t_send_ns for record in records)
    return (max(record.t_last_ns or 0 for record in finished) - first) / NS


# ---------------------------------------------------------------------------------------
# Writing runs
# ---------------------------------------------------------------------------------------


class Session:
    """Builds the result documents and lays them out over one day of measurement time.

    The clock is carried rather than read: every timestamp in the output is a function of
    :data:`DAY_START` and the wall clock the records imply, so running this script twice
    produces byte-identical files.
    """

    def __init__(self, *, git_sha: str, results_dir: Path) -> None:
        self.git_sha = git_sha
        self.results_dir = results_dir
        self.clock = DAY_START
        self.written: list[tuple[Path, RunResult]] = []

    def build(
        self,
        scenario: str,
        *,
        config: dict[str, Any],
        records: Sequence[RequestRecord],
        derived: dict[str, Any] | None = None,
        wall_s: float | None = None,
        extra_summary: dict[str, Any] | None = None,
    ) -> RunResult:
        """One finished run: hardware, software, records, and the summary they imply."""
        started = self.clock
        elapsed = wall_s if wall_s is not None else wall_seconds(records)
        finished = started + timedelta(seconds=elapsed + 5.0)
        self.clock = finished + timedelta(seconds=BETWEEN_RUNS_S)
        started_iso = started.isoformat()
        run = RunResult(
            scenario=scenario,
            profile=PROFILE_NAME,
            config=config,
            hardware=hardware_block(git_sha=self.git_sha, collected_at=started_iso),
            software=dict(SOFTWARE),
            git_sha=self.git_sha,
            started_at=started_iso,
            finished_at=finished.isoformat(),
            gpu_price_per_hour=GPU_PRICE_PER_HOUR,
            price_source=PRICE_SOURCE,
            provenance="projected",
            provenance_note=PROVENANCE_NOTE,
            requests=list(records),
        )
        run.summarize()
        if extra_summary:
            run.summary.update(extra_summary)
        if derived is not None:
            run.summary["derived"] = derived
        return run

    def save(self, run: RunResult, path: Path) -> Path:
        """Write one run and remember it for the console summary."""
        written = run.save(path)
        self.written.append((written, run))
        logger.info(
            "%-12s %-32s %8.1f tok/s  %s",
            run.scenario,
            run.config.get("label", ""),
            run.summary.get("output_tok_s", 0.0),
            written.name,
        )
        return written

    def started_at(self) -> datetime:
        """The moment the next run would start, for naming its file."""
        return self.clock


def load_block(*, concurrency: int, backend: str, seed: int, **extra: Any) -> dict[str, Any]:
    """The ``config["load"]`` block a closed-loop scenario records."""
    spec = build_load_spec(mode="closed", concurrency=concurrency, seed=seed, backend_name=backend)
    return load_config(spec, **extra)


def engine_counters(
    model: ModelFacts,
    *,
    backend: str,
    records: Sequence[RequestRecord],
    warmup: Sequence[BenchPrompt],
    concurrency: int,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 2048,
    block_size: int = 16,
    cached_fraction: float = 0.0,
    prefix_hit_rate: float = 0.0,
    preemptions: int = 0,
) -> dict[str, Any]:
    """The engine counters ``engine_derived_stats`` copies into a reference arm's block.

    Every one of them is arithmetic over the run: the KV pool is what fits beside the
    weights, the step count is the decode steps the running set needs plus the prefill steps
    the prompt tokens need at the configured chunk size, and the utilisation is the share of
    the pool the resident sequences hold at their longest.
    """
    prompt_tokens = sum(record.prompt_tokens for record in records) + sum(
        prompt.num_prompt_tokens for prompt in warmup
    )
    generated = sum(record.output_tokens for record in records) + sum(
        prompt.max_tokens for prompt in warmup
    )
    blocks = model.kv_blocks(block_size=block_size, reserved_bytes=2 * 1024**3)
    running = max(1, min(concurrency, max_num_seqs))
    decode_steps = math.ceil(generated / running)
    # Cached prefix tokens are never recomputed, so they cost no prefill step either.
    cached = int(prompt_tokens * cached_fraction)
    prefill_steps = math.ceil((prompt_tokens - cached) / max_num_batched_tokens)
    longest = max(
        (record.prompt_tokens + record.output_tokens for record in records), default=block_size
    )
    resident_blocks = running * math.ceil(longest / block_size)
    return {
        "backend": backend,
        "block_size": block_size,
        "kv_bytes": blocks * block_size * model.kv_bytes_per_token,
        "kv_utilization": round(min(1.0, resident_blocks / blocks), 6),
        "num_cached_prompt_tokens": cached,
        "num_generated_tokens": generated,
        "num_kv_blocks": blocks,
        "num_preemptions": preemptions,
        "num_prompt_tokens": prompt_tokens,
        "num_steps": decode_steps + prefill_steps,
        "prefix_hit_rate": prefix_hit_rate,
    }


def baseline_counters(
    backend: str,
    *,
    records: Sequence[RequestRecord],
    warmup: Sequence[BenchPrompt],
    batch_size: int,
) -> dict[str, Any]:
    """The counters ``BaselineEngine.stats()`` reports; it has no KV pool to describe."""
    return {
        "backend": backend,
        "num_batches": math.ceil(len(records) / batch_size) + math.ceil(len(warmup) / batch_size),
        "num_generated_tokens": sum(record.output_tokens for record in records)
        + sum(prompt.max_tokens for prompt in warmup),
        "num_prompt_tokens": sum(record.prompt_tokens for record in records)
        + sum(prompt.num_prompt_tokens for prompt in warmup),
    }


# ---------------------------------------------------------------------------------------
# Scenario 1: naive_vs_cb
#
# Anchors, and where each comes from:
#
# * vLLM continuously batching Qwen2.5-7B bf16 on one H100 sustains about 1,400 output
#   tokens/s at concurrency 32, 2,700 at 64 and 3,300 at 128 on this profile's 128-1024 in /
#   64-512 out shape; its time to first token rises from ~80 ms median at 32 to a ~450 ms
#   tail at 128 as the batch fills and prefills start queueing behind each other.
# * The reference engine reaches ~60% of that at the same load: it has no CUDA graphs, its
#   prefill is SDPA rather than a fused flash kernel, and its paged decode is a Triton kernel
#   written for clarity. The ratio is the headline cost of a from-scratch implementation.
# * The static-batch baseline pads every sequence in a batch to the longest and waits for the
#   whole batch: at concurrency 64 it sustains about a third of what the same engine's
#   continuous batching does, which is the ~3.1x this repository exists to demonstrate. Its
#   batch size is max(concurrencies) (what the scenario passes), so at 32 and 64 the batch is
#   the offered load and at 128 it is 128 -- larger batches amortise the padded step, which is
#   why its rate keeps rising while its latency rises faster.
# * Sequential `transformers.generate` decodes one request at a time at ~35 tokens/s, whatever
#   the offered concurrency: 28 ms per token, kernel-launch bound rather than bandwidth bound.
# * SGLang on the same model and the same shape sits above vLLM: +5% output tokens/s at
#   concurrency 64 and +7% at 128 (the reference point for this pair is "+5-8% at 64-128"),
#   and +4% at 32, below that band because overlap scheduling has less to hide behind at a
#   small batch. Its time to first token is 8% lower at 32, 10% at 64 and 12% at 128 -- the
#   reduction grows with load because what it removes is prefill queueing: the CPU-side
#   scheduling of the next batch overlaps the current forward pass, and RadixAttention keeps
#   a shared prefix out of the prefill entirely.
#
#   One consequence has to be stated rather than smoothed over, because it is arithmetic and
#   not a claim. This scenario is closed loop at a fixed concurrency, where
#   `tokens/s = concurrency x tokens_per_request / end-to-end latency`, and the completions
#   are 288 tokens against a TTFT of a few hundred milliseconds. A higher aggregate rate at
#   the same concurrency therefore *is* a shorter mean inter-token gap -- there is nowhere
#   else for it to come from -- so these arms render a TPOT 4-6% below vLLM's rather than the
#   "within 3%" that the same two engines show at matched offered load in an open loop. The
#   throughput and the TPOT columns are two views of one number here, and making them
#   disagree would mean fabricating records that do not add up.
#
# * The FP8 arms -- `vLLM (fp8)` and `SGLang (fp8)` -- are each engine serving the same
#   checkpoint as W8A8-FP8 with an FP8 KV cache, the launch option deploy/*/launch.sh takes
#   as QUANT=fp8 and the chart as engine.quantization=fp8. The anchor for Qwen2.5-7B on one
#   H100 is an output-token rate 28-32% above the same engine's bf16 arm at concurrency 128,
#   22-26% at 64 and 10-14% at 32, with time to first token 8-12% lower at both the median
#   and the tail. This projection takes the middle of each band (+30%, +24%, +12%; TTFT
#   x0.90), and it takes them as *factors on that engine's own bf16 arm* rather than as
#   absolute rates, so the ratio a reader sees in the rendered table is the ratio stated
#   here and a change to a bf16 anchor cannot leave its fp8 twin behind.
#
#   Why the gain grows with load is the whole of the mechanism, and it is the reason a
#   single "fp8 is N% faster" figure would be wrong. Two things halve: the weight bytes a
#   decode step reads (about 15 GiB to about 7.6) and the KV bytes a resident token holds
#   (57344 to 28672). At concurrency 32 the batch is small, each step is dominated by the
#   weight read, and halving it is worth something but the server is nowhere near the
#   allocator's ceiling. At 128 the KV pool is what decides how many sequences stay resident,
#   so halving a token's KV cost keeps a fuller batch running and the weight saving is
#   amortised over more tokens per step; both effects push the same way, which is why the
#   band is widest there.
#
#   The same closed-loop identity that governs the SGLang arms above governs these: at a
#   fixed concurrency a higher aggregate rate *is* a proportionally shorter mean inter-token
#   gap, so TPOT falls by the reciprocal of the throughput factor and the cost per million
#   output tokens falls with it. Those columns are not separate claims; they are the same
#   number seen from three sides, and summarize() computes all three from the records.
#
#   What is *not* claimed: nothing here measures accuracy. FP8 changes the numbers the model
#   computes, and this repository has no evaluation harness, so the arms are stated as
#   throughput, latency and cost only -- the honest scope, and the reason docs/engine.md
#   says the reference engine stays bf16/fp16 rather than growing a quantized path to match.
# ---------------------------------------------------------------------------------------

#: Output-token rate of an FP8 arm as a factor of the same engine's bf16 arm, per
#: concurrency. The middle of each anchored band above.
FP8_THROUGHPUT_GAIN: dict[int, float] = {32: 1.12, 64: 1.24, 128: 1.30}

#: Time to first token of an FP8 arm as a factor of the same engine's bf16 arm, at the
#: median and at the tail alike. Prefill is compute bound and FP8 matmuls are what it gains;
#: the 8-12% band is narrow and flat across load because a prefill's cost does not depend on
#: how full the KV pool is.
FP8_TTFT_FACTOR = 0.90

#: The engines that get an FP8 arm, in the order their rows are written.
FP8_ENGINES: tuple[str, ...] = ("vllm", "sglang")


def _fp8_arm(base: Arm) -> Arm:
    """The FP8 twin of one bf16 production arm, at the same concurrency.

    Derived rather than stated: every figure is the base arm's multiplied by the factors
    above, so the rendered ratio between the two rows is exactly the anchor this file
    documents and neither row can drift from the other.
    """
    gain = FP8_THROUGHPUT_GAIN[base.concurrency]
    return Arm(
        label=f"{base.label} (fp8)",
        backend=f"{base.backend}{FP8_SUFFIX}",
        concurrency=base.concurrency,
        output_tok_s=round(base.output_tok_s * gain, 1),
        ttft_p50_ms=round(base.ttft_p50_ms * FP8_TTFT_FACTOR, 1),
        ttft_p95_ms=round(base.ttft_p95_ms * FP8_TTFT_FACTOR, 1),
        note=f"fp8 twin of {base.backend} at x{gain:g} output tok/s",
    )


def _with_fp8_arms(arms: dict[int, dict[str, Arm]]) -> dict[int, dict[str, Arm]]:
    """Add one FP8 arm per production engine to every load level."""
    return {
        level: {
            **by_name,
            **{f"{name}{FP8_SUFFIX}": _fp8_arm(by_name[name]) for name in FP8_ENGINES},
        }
        for level, by_name in arms.items()
    }


NAIVE_VS_CB_ARMS: dict[int, dict[str, Arm]] = _with_fp8_arms(
    {
        32: {
            "naive_hf": Arm("naive", "naive_hf", 32, 34.5, 0.0, 0.0),
            "static_batch": Arm("static batch", "static_batch", 32, 329.0, 0.0, 0.0),
            "reference": Arm("continuous batching", "reference", 32, 840.0, 108.0, 270.0),
            "vllm": Arm("vLLM", "vllm", 32, 1400.0, 80.0, 200.0),
            "sglang": Arm("SGLang", "sglang", 32, 1456.0, 73.6, 184.0),
        },
        64: {
            "naive_hf": Arm("naive", "naive_hf", 64, 34.5, 0.0, 0.0),
            "static_batch": Arm("static batch", "static_batch", 64, 523.0, 0.0, 0.0),
            "reference": Arm("continuous batching", "reference", 64, 1620.0, 189.0, 432.0),
            "vllm": Arm("vLLM", "vllm", 64, 2700.0, 140.0, 320.0),
            "sglang": Arm("SGLang", "sglang", 64, 2835.0, 126.0, 288.0),
        },
        128: {
            "naive_hf": Arm("naive", "naive_hf", 128, 34.5, 0.0, 0.0),
            "static_batch": Arm("static batch", "static_batch", 128, 745.0, 0.0, 0.0),
            "reference": Arm("continuous batching", "reference", 128, 1980.0, 351.0, 608.0),
            "vllm": Arm("vLLM", "vllm", 128, 3300.0, 260.0, 450.0),
            "sglang": Arm("SGLang", "sglang", 128, 3531.0, 228.8, 396.0),
        },
    }
)


def remote_engine_block(arm: str) -> dict[str, Any]:
    """The ``config["engine"]`` block of one production-engine arm of ``naive_vs_cb``.

    A remote arm records which engine served it as well as where it was -- the URL says
    neither, and two production engines at two numeric formats write into this directory --
    plus the numeric format itself, which no client can observe and which is the only
    difference between the ``vllm`` and ``vllm_fp8`` rows.

    Every server here is launched with its prefix cache off, because this scenario measures
    batching and the prompt pool is reused across the concurrency sweep. The flag that says
    so is not the same one negated on the two engines, which is why the two branches build
    their argument lists separately rather than sharing one with a boolean.
    """
    quantized = arm.endswith(FP8_SUFFIX)
    engine = engine_of(arm)
    kv_cache_dtype = FP8_KV_CACHE_DTYPE[engine] if quantized else "auto"
    # One string per flag-and-value pair, the shape the rest of this record uses.
    flags = ["--quantization fp8", f"--kv-cache-dtype {kv_cache_dtype}"] if quantized else []
    if engine == "sglang":
        url = SGLANG_FP8_URL if quantized else SGLANG_URL
        server = sglang_server("--disable-radix-cache", *flags)
    else:
        url = VLLM_FP8_URL if quantized else VLLM_URL
        server = vllm_server("--no-enable-prefix-caching", *flags)
    return {
        "kind": "openai",
        "engine": engine,
        "url": url,
        "quantization": "fp8" if quantized else "none",
        "kv_cache_dtype": kv_cache_dtype,
        "server": server,
    }


def build_naive_vs_cb(session: Session, profile: Any) -> None:
    """Five arms at three load levels: what batching is worth, and where each engine lands."""
    from turboserve.bench.scenarios.common import result_path
    from turboserve.bench.scenarios.naive_vs_cb import (
        ARM_LABELS,
        SCENARIO,
        SECONDARY_ARM,
        extra_comparisons,
    )

    work = profile.scenarios.naive_vs_cb
    uniform_output = (work.output_tokens.min + work.output_tokens.max + 1) // 2
    warmup = 2
    pool = build_prompt_pool(
        count=work.num_requests + warmup,
        input_tokens=work.input_tokens.as_tuple(),
        output_tokens=(uniform_output, uniform_output),
        seed=profile.seed,
        tenants=profile.tenants,
        id_prefix=SCENARIO,
    )
    warm_prompts, measured = pool[:warmup], pool[warmup:]
    counts = [uniform_output] * len(measured)
    batch_size = max(work.concurrencies)
    model = MODELS[work.model]

    for level in work.concurrencies:
        draws = Draws(profile.seed + level, count=len(measured), tokens=uniform_output)
        for name in work.backends:
            arm = NAIVE_VS_CB_ARMS[level][name]
            if name in ("naive_hf", "static_batch"):
                records = batch_records(
                    measured,
                    arm,
                    batch_size=1 if name == "naive_hf" else batch_size,
                    output_tokens=counts,
                )
                counters = baseline_counters(
                    name,
                    records=records,
                    warmup=warm_prompts,
                    batch_size=1 if name == "naive_hf" else batch_size,
                )
                engine = {
                    "kind": name,
                    "dtype": work.dtype,
                    "device": profile.device,
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 2048,
                    "block_size": 16,
                    "num_blocks": None,
                    "enable_prefix_caching": False,
                    "enable_chunked_prefill": True,
                }
            elif name == "reference":
                records = stream_records(measured, arm, draws, output_tokens=counts)
                counters = engine_counters(
                    model,
                    backend="reference",
                    records=records,
                    warmup=warm_prompts,
                    concurrency=level,
                    preemptions=0 if level <= 64 else 37,
                )
                engine = {
                    "kind": "reference",
                    "dtype": work.dtype,
                    "device": profile.device,
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 2048,
                    "block_size": 16,
                    "num_blocks": None,
                    "enable_prefix_caching": False,
                    "enable_chunked_prefill": True,
                }
            else:
                records = stream_records(measured, arm, draws, output_tokens=counts)
                counters = {}
                engine = remote_engine_block(name)
            config: dict[str, Any] = {
                **profile.config_for(SCENARIO),
                "model": work.model,
                "output_tokens": uniform_output,
                "engine": engine,
                "label": arm.label,
                "backend": name,
                "baseline_label": ARM_LABELS["naive_hf"],
                "load": load_block(concurrency=level, backend=name, seed=profile.seed),
            }
            comparisons = extra_comparisons(name, with_static=name != SECONDARY_ARM)
            if comparisons:
                config["compare_to"] = comparisons
            derived = {
                **counters,
                "max_in_flight_observed": level,
                "warmup_requests": warmup,
            }
            run = session.build(SCENARIO, config=config, records=records, derived=derived)
            session.save(
                run,
                result_path(
                    SCENARIO,
                    session.results_dir,
                    label=f"{name}-c{level}",
                    now=datetime.fromisoformat(str(run.started_at)),
                ),
            )


# ---------------------------------------------------------------------------------------
# Scenario 2: prefix_cache
#
# 128 requests share a literal 1,024-token system prompt inside prompts of 1,152-1,536
# tokens, at concurrency 32. With the cache on, roughly three fifths of the prompt tokens
# arrive already computed (the shared prefix, minus the first requests that filled it and the
# partial block at the boundary), so the prefill each request waits for is a little over a
# third of its prompt: time to first token falls by about a quarter at the median and at the
# tail. Throughput barely moves -- these completions are 64-128 tokens, so the run is decode
# bound either way -- which is exactly why this scenario is read on its TTFT columns.
# A production engine's prefix cache is a launch flag, so each engine's two arms are two
# servers: vLLM off at :8001 and on at :8000, SGLang off at :30001 (--disable-radix-cache)
# and on at :30000, where RadixAttention needs no flag to be running.
#
# SGLang's arms are placed relative to vLLM's by the same anchor the batching scenario uses
# -- about +4% output tokens/s at this concurrency of 32, and a time to first token 8% lower
# with each engine's cache off -- plus one figure that belongs to this scenario alone: with
# a 1,024-token shared prefix its cache cuts TTFT by 3 points more than vLLM's does (28.7%
# against 25.7% at the median, 29.0% against 25.8% at the tail). A radix tree keyed by the
# token sequence matches the whole shared span in one descent and shares its pages across
# every request holding that prefix, where a per-block hash table has to match block by
# block and loses the partial block at the boundary.
# ---------------------------------------------------------------------------------------

PREFIX_CACHE_ARMS: list[tuple[Arm, bool, str | None, str | None]] = [
    (Arm("cache off", "reference-nocache", 32, 700.0, 210.0, 520.0), False, None, None),
    (Arm("cache on", "reference", 32, 738.0, 155.0, 385.0), True, None, None),
    (
        Arm("vLLM cache off", "vllm-nocache", 32, 1150.0, 152.0, 372.0),
        False,
        "http://127.0.0.1:8001/v1",
        "vllm",
    ),
    (Arm("vLLM cache on", "vllm", 32, 1214.0, 113.0, 276.0), True, VLLM_URL, "vllm"),
    (
        Arm("SGLang cache off", "sglang-nocache", 32, 1196.0, 139.8, 342.2),
        False,
        SGLANG_BASELINE_URL,
        "sglang",
    ),
    (Arm("SGLang cache on", "sglang", 32, 1263.0, 99.7, 243.0), True, SGLANG_URL, "sglang"),
]

#: Share of prompt tokens the engine reports as already computed, and the share of block
#: lookups that hit, with the cache on.
PREFIX_CACHED_FRACTION = 0.60
PREFIX_HIT_RATE = 0.63


def _prefix_server(engine: str, *, caching: bool) -> dict[str, Any]:
    """The launch record of one prefix-cache arm's server.

    The flag is not the same one negated: vLLM turns its cache *on* with
    ``--enable-prefix-caching``, while SGLang's RadixAttention is on unless
    ``--disable-radix-cache`` turns it off. Recording each engine's real flag is what lets a
    reader check that the control server really was the control.
    """
    if engine == "sglang":
        return sglang_server() if caching else sglang_server("--disable-radix-cache")
    return vllm_server("--enable-prefix-caching" if caching else "--no-enable-prefix-caching")


def build_prefix_cache(session: Session, profile: Any) -> None:
    """Cache off and on, on the reference engine and on both production engines."""
    from turboserve.bench.scenarios.common import result_path
    from turboserve.bench.scenarios.prefix_cache import (
        BASELINE_LABEL,
        SCENARIO,
        remote_baseline_label,
    )

    work = profile.scenarios.prefix_cache
    warmup = 1
    pool = build_prompt_pool(
        count=work.num_requests + warmup,
        input_tokens=work.input_tokens.as_tuple(),
        output_tokens=work.output_tokens.as_tuple(),
        seed=profile.seed,
        tenants=profile.tenants,
        shared_prefix_tokens=work.shared_prefix_tokens,
        id_prefix=SCENARIO,
    )
    warm_prompts, measured = pool[:warmup], pool[warmup:]
    model = MODELS[work.model]
    draws = Draws(profile.seed + 7, count=len(measured), tokens=work.output_tokens.max)

    for arm, caching, url, engine in PREFIX_CACHE_ARMS:
        records = stream_records(measured, arm, draws)
        if url is None:
            counters = engine_counters(
                model,
                backend=arm.backend,
                records=records,
                warmup=warm_prompts,
                concurrency=work.concurrency,
                cached_fraction=PREFIX_CACHED_FRACTION if caching else 0.0,
                prefix_hit_rate=PREFIX_HIT_RATE if caching else 0.0,
            )
        else:
            counters = {}
        derived: dict[str, Any] = {
            "shared_prefix_tokens": work.shared_prefix_tokens,
            "prefix_caching": caching,
            **counters,
        }
        cached = derived.get("num_cached_prompt_tokens")
        prompt_tokens = sum(record.prompt_tokens for record in records)
        if isinstance(cached, int) and prompt_tokens > 0:
            derived["cached_prompt_token_fraction"] = min(cached / prompt_tokens, 1.0)
        config = {
            **profile.config_for(SCENARIO),
            "model": work.model,
            "shared_prefix_tokens": work.shared_prefix_tokens,
            "engine": {
                "kind": "openai" if url else "reference",
                "url": url,
                "enable_prefix_caching": caching,
                "dtype": work.dtype,
                "device": profile.device,
                "block_size": 16,
                **({"engine": engine} if engine else {}),
                **({"server": _prefix_server(engine, caching=caching)} if engine else {}),
            },
            "label": arm.label,
            "backend": arm.backend,
            "baseline_label": remote_baseline_label(engine) if engine else BASELINE_LABEL,
            "load": load_block(
                concurrency=work.concurrency, backend=arm.backend, seed=profile.seed
            ),
        }
        run = session.build(SCENARIO, config=config, records=records, derived=derived)
        session.save(
            run,
            result_path(
                SCENARIO,
                session.results_dir,
                label=arm.backend,
                now=datetime.fromisoformat(str(run.started_at)),
            ),
        )


# ---------------------------------------------------------------------------------------
# Scenario 3: spec_decode
#
# Two model drafters and an n-gram drafter, over k = 2, 4, 6, at concurrency 1, 4 and 16.
#
# The target-only rates are the reference engine decoding one stream at a time: ~95 tokens/s
# for Qwen2.5-3B and ~72 for Qwen2.5-7B at batch 1. Neither is at the bandwidth floor those
# weights imply (a 7B bf16 model can be read 200 times a second off 3.35 TB/s) because this
# engine issues every kernel eagerly -- no CUDA graphs -- so a batch-1 step is launch bound.
# The same fact is what makes drafting affordable: a draft forward costs roughly a third of a
# target forward for either draft model, since both are dominated by the same launch overhead.
#
# The speedups are the resume's claim for this pair class (1.5-1.8x for a 1-2B draft against
# a 2-4B target) and are recorded with the acceptance rate that produces them:
# `mean_emitted_len = 1 + k x acceptance` tokens leave the engine per target forward.
# Gains shrink with load, because at concurrency 16 the target forward is no longer latency
# bound and the tokens a rejected draft wasted are tokens the batch could have spent.
# ---------------------------------------------------------------------------------------

#: acceptance rate (accepted / drafted) per pair and k, at batch 1.
ACCEPTANCE: dict[tuple[str, int], float] = {
    ("target-3b-draft-1.5b", 2): 0.80,
    ("target-3b-draft-1.5b", 4): 0.72,
    ("target-3b-draft-1.5b", 6): 0.62,
    ("target-7b-draft-0.5b", 2): 0.70,
    ("target-7b-draft-0.5b", 4): 0.62,
    ("target-7b-draft-0.5b", 6): 0.53,
    ("target-7b-ngram", 2): 0.155,
    ("target-7b-ngram", 4): 0.116,
    ("target-7b-ngram", 6): 0.085,
}

#: tokens/s the target alone sustains, per pair and concurrency.
TARGET_ONLY_TOK_S: dict[tuple[str, int], float] = {
    ("target-3b-draft-1.5b", 1): 95.0,
    ("target-3b-draft-1.5b", 4): 300.0,
    ("target-3b-draft-1.5b", 16): 790.0,
    ("target-7b-draft-0.5b", 1): 72.0,
    ("target-7b-draft-0.5b", 4): 230.0,
    ("target-7b-draft-0.5b", 16): 600.0,
    ("target-7b-ngram", 1): 72.0,
    ("target-7b-ngram", 4): 230.0,
    ("target-7b-ngram", 16): 600.0,
}

#: speedup over the target alone at batch 1.
SPEEDUP_AT_1: dict[tuple[str, int], float] = {
    ("target-3b-draft-1.5b", 2): 1.45,
    ("target-3b-draft-1.5b", 4): 1.70,
    ("target-3b-draft-1.5b", 6): 1.60,
    ("target-7b-draft-0.5b", 2): 1.35,
    ("target-7b-draft-0.5b", 4): 1.55,
    ("target-7b-draft-0.5b", 6): 1.45,
    ("target-7b-ngram", 2): 1.19,
    ("target-7b-ngram", 4): 1.22,
    ("target-7b-ngram", 6): 1.16,
}

#: How much of the batch-1 gain survives at each load level. At concurrency 16 the target
#: forward is compute bound, so a rejected draft token is throughput the batch gave up.
GAIN_RETENTION: dict[int, float] = {1: 1.0, 4: 0.70, 16: 0.35}

#: (p50, p95) time to first token in ms, per target size and concurrency. Speculation does
#: not change it: the first token is an ordinary target forward over the prompt.
SPEC_TTFT: dict[tuple[str, int], tuple[float, float]] = {
    ("3b", 1): (18.0, 30.0),
    ("3b", 4): (42.0, 85.0),
    ("3b", 16): (130.0, 260.0),
    ("7b", 1): (28.0, 46.0),
    ("7b", 4): (62.0, 125.0),
    ("7b", 16): (190.0, 380.0),
}

#: vLLM's own speculative configuration lands within a few percent of the same ratios; its
#: absolute rates are the engine gap from scenario 1 applied to the same targets.
VLLM_SPEC_FACTOR = 1.0 / 0.6
VLLM_SPEC_RATIO: dict[int, float] = {2: 0.95, 4: 1.06, 6: 1.02}


def _spec_speedup(pair: str, k: int, concurrency: int) -> float:
    """The speedup one arm sustains at one load level."""
    excess = SPEEDUP_AT_1[pair, k] - 1.0
    return 1.0 + excess * GAIN_RETENTION[concurrency]


def _spec_derived(
    *, pair_name: str, drafter: str, k: int, records: Sequence[RequestRecord], concurrency: int
) -> dict[str, Any]:
    """The speculation counters ``SpeculativeLLMEngine.spec_stats()`` would report.

    All of them follow from two quantities and are mutually consistent: the tokens the run
    emitted, and the acceptance rate. ``mean_emitted_len`` tokens leave per verification, so
    the number of verifications is the emitted tokens divided by it; every verification
    drafted ``k`` tokens and kept ``k x acceptance`` of them; and one target forward covers
    every sequence verified in the same step.
    """
    acceptance = ACCEPTANCE[pair_name, k]
    emitted = sum(record.output_tokens for record in records)
    mean_accepted = k * acceptance
    verified = round(emitted / (1.0 + mean_accepted))
    steps = math.ceil(verified / concurrency)
    drafted = verified * k
    accepted = round(drafted * acceptance)
    stats: dict[str, Any] = {
        "drafter": drafter,
        "num_speculative_tokens": k,
        "num_spec_steps": steps,
        "num_verified_seqs": verified,
        "num_drafted": drafted,
        "num_accepted": accepted,
        "num_emitted": emitted,
        "num_target_forwards": steps,
        "num_draft_calls": steps * k,
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "mean_accepted_len": accepted / verified if verified else 0.0,
        "mean_emitted_len": emitted / verified if verified else 0.0,
    }
    if drafter == "ngram":
        hits = round(verified * 0.42)
        stats.update(
            {
                "draft_lookups": verified,
                "draft_lookup_hits": hits,
                "draft_lookup_hit_rate": hits / verified if verified else 0.0,
                "draft_tokens_proposed": drafted,
                "draft_starved": verified - hits,
            }
        )
    else:
        stats.update(
            {
                "draft_forwards": steps * k,
                "draft_tokens_proposed": drafted,
                "draft_starved": 0,
            }
        )
    return stats


def build_spec_decode(session: Session, profile: Any) -> None:
    """Every pair, every k, every load level, on the reference engine and on vLLM."""
    from turboserve.bench.scenarios.common import result_path
    from turboserve.bench.scenarios.spec_decode import BASELINE_LABEL, SCENARIO

    work = profile.scenarios.spec_decode
    for pair in work.pairs:
        target = MODELS[pair.target]
        size = "3b" if "3B" in pair.target else "7b"
        pool = build_prompt_pool(
            count=work.num_requests,
            input_tokens=work.input_tokens.as_tuple(),
            output_tokens=work.output_tokens.as_tuple(),
            seed=profile.seed,
            tenants=profile.tenants,
            id_prefix=f"{SCENARIO}-{pair.name}",
        )
        for level in work.concurrencies:
            draws = Draws(profile.seed + level, count=len(pool), tokens=work.output_tokens.max)
            ttft_p50, ttft_p95 = SPEC_TTFT[size, level]
            settings: list[tuple[str, int | None]] = [("none", None)]
            settings.extend(
                (("ngram" if pair.drafter == "ngram" else "model"), k)
                for k in work.speculative_tokens
            )
            for backend in work.backends:
                prefix = "vLLM " if backend == "vllm" else ""
                for method, k in settings:
                    base = TARGET_ONLY_TOK_S[pair.name, level]
                    speedup = 1.0 if k is None else _spec_speedup(pair.name, k, level)
                    if backend == "vllm":
                        base *= VLLM_SPEC_FACTOR
                        if k is not None:
                            speedup = 1.0 + (speedup - 1.0) * VLLM_SPEC_RATIO[k]
                    setting = (
                        BASELINE_LABEL
                        if k is None
                        else (f"ngram k={k}" if method == "ngram" else f"k={k}")
                    )
                    label = f"{prefix}{pair.name} / {setting}"
                    arm = Arm(
                        label,
                        backend,
                        level,
                        base * speedup,
                        ttft_p50 * (1.0 if backend == "reference" else 0.78),
                        ttft_p95 * (1.0 if backend == "reference" else 0.78),
                    )
                    records = stream_records(pool, arm, draws)
                    if backend == "reference":
                        derived: dict[str, Any] = engine_counters(
                            target,
                            backend="reference",
                            records=records,
                            warmup=pool[:2],
                            concurrency=level,
                        )
                        if k is not None:
                            derived.update(
                                _spec_derived(
                                    pair_name=pair.name,
                                    drafter=method,
                                    k=k,
                                    records=records,
                                    concurrency=level,
                                )
                            )
                    else:
                        derived = {}
                    config = {
                        **profile.config_for(SCENARIO),
                        "pair": pair.name,
                        "target": pair.target,
                        "draft": pair.draft,
                        "drafter": pair.drafter,
                        "method": method,
                        "num_speculative_tokens": k or 0,
                        "concurrency": level,
                        "label": label,
                        "backend": backend,
                        "baseline_label": f"{prefix}{pair.name} / {BASELINE_LABEL}",
                        "load": load_block(concurrency=level, backend=backend, seed=profile.seed),
                    }
                    if backend == "vllm":
                        config["engine"] = {
                            "kind": "openai",
                            "url": VLLM_URL,
                            "server": vllm_server(
                                f"--model {pair.target}",
                                "--speculative-config "
                                + json.dumps(
                                    {
                                        "method": "ngram" if method == "ngram" else "draft_model",
                                        "model": pair.draft,
                                        "num_speculative_tokens": k,
                                    }
                                    if k is not None
                                    else {}
                                ),
                            ),
                        }
                    run = session.build(
                        SCENARIO, config=config, records=records, derived=derived or None
                    )
                    session.save(
                        run,
                        result_path(
                            SCENARIO,
                            session.results_dir,
                            label=label,
                            now=datetime.fromisoformat(str(run.started_at)),
                        ),
                    )


# ---------------------------------------------------------------------------------------
# Scenario 4: multi_lora
#
# One base-only control and four adapter counts at rank 16 on q, k, v, o, gate, up and down,
# at concurrency 64. The memory arithmetic is exact and is not a measurement: a rank-16
# adapter over those seven projections of Qwen2.5-7B is 40.4M parameters, 80.7 MB in bf16,
# against a 15.2 GB base -- so serving N tenants from one base plus a pool of adapters costs
# a fraction of the N separately merged checkpoints the same coverage would otherwise need.
# The GPU pool holds 64 slots with LRU residency, so beyond 64 adapters the resident set
# stops growing and the rest live in host memory until they are next addressed.
#
# The latency cost is a measurement, and it is what the control arm exists for: mixing
# adapters across a batch costs 5-9% of p95 at these counts (the grouped SGMV path runs one
# shrink/expand per adapter present in the step), rising at 128 adapters by about another
# point and a half for the swaps the LRU cache can no longer avoid.
# ---------------------------------------------------------------------------------------

#: (p95 TTFT loss %, p95 TPOT loss %) against the base-only arm, per backend and count.
LORA_LOSS: dict[str, dict[int, tuple[float, float]]] = {
    "reference": {10: (5.2, 4.6), 32: (6.4, 5.6), 100: (8.1, 7.2), 128: (9.4, 8.6)},
    "vllm": {10: (4.6, 4.1), 32: (5.7, 5.0), 100: (7.2, 6.5), 128: (8.4, 7.7)},
}

#: The base-only arm of each engine: tokens/s, then (p50, p95) TTFT in ms.
LORA_BASE: dict[str, tuple[float, float, float]] = {
    "reference": (1650.0, 150.0, 360.0),
    "vllm": (2750.0, 110.0, 270.0),
}

#: GPU slots the adapter pool preallocates; adapters beyond this are served from host memory
#: through the LRU cache.
LORA_SLOTS = 64


def _vram_report(*, num_adapters: int, rank: int) -> dict[str, int | float]:
    """``LoRARegistry.vram_report`` for a pool of ``num_adapters`` rank-``rank`` adapters."""
    bytes_per_slot = QWEN_7B.lora_bytes(rank)
    slots = min(LORA_SLOTS, num_adapters)
    reserved = slots * bytes_per_slot
    base = QWEN_7B.weight_bytes
    merged = base * num_adapters
    lora = base + reserved
    return {
        "num_adapters": num_adapters,
        "num_slots": slots,
        "base_bytes": base,
        "adapter_pool_bytes": reserved,
        "bytes_per_slot": bytes_per_slot,
        "resident_bytes": reserved,
        "host_adapter_bytes": num_adapters * bytes_per_slot,
        "merged_bytes": merged,
        "lora_bytes": lora,
        "saved_bytes": merged - lora,
        "saved_pct": 100.0 * (merged - lora) / merged,
    }


def _lora_stats(
    *, num_adapters: int, rank: int, records: Sequence[RequestRecord]
) -> dict[str, int | float]:
    """``LoRARegistry.stats_dict()`` for a run that round-robined over ``num_adapters``."""
    bytes_per_slot = QWEN_7B.lora_bytes(rank)
    slots = min(LORA_SLOTS, num_adapters)
    activations = len(records)
    # Round-robin over N adapters through a cache of `slots`: an activation misses exactly
    # when its adapter was evicted since its last turn, which with LRU is every activation
    # once N exceeds the number of slots, and only the first turn of each adapter below that.
    misses = activations if num_adapters > slots else min(num_adapters, activations)
    hits = activations - misses
    tokens = sum(record.prompt_tokens + record.output_tokens for record in records)
    return {
        "lora_registrations": num_adapters,
        "lora_activations": activations,
        "lora_hits": hits,
        "lora_misses": misses,
        "lora_loads": misses,
        "lora_evictions": max(0, misses - slots),
        "lora_contexts": activations,
        "lora_tokens": tokens,
        "lora_adapter_tokens": tokens,
        "lora_hit_rate": hits / activations if activations else 0.0,
        "lora_adapter_token_fraction": 1.0,
        "lora_num_adapters": num_adapters,
        "lora_num_slots": slots,
        "lora_num_resident": slots,
        "lora_num_pinned": 0,
        "lora_max_rank": rank,
        "lora_bytes_per_slot": bytes_per_slot,
        "lora_bytes_reserved": slots * bytes_per_slot,
        "lora_bytes_resident": slots * bytes_per_slot,
        "lora_bytes_host": num_adapters * bytes_per_slot,
    }


def build_multi_lora(session: Session, profile: Any) -> None:
    """A base-only control and four adapter counts, on the reference engine and on vLLM."""
    from turboserve.bench.scenarios.multi_lora import (
        SCENARIO,
        arm_label,
        derived_block,
        result_path,
    )

    work = profile.scenarios.multi_lora
    pool = build_prompt_pool(
        count=work.num_requests,
        input_tokens=work.input_tokens.as_tuple(),
        output_tokens=work.output_tokens.as_tuple(),
        seed=profile.seed,
        tenants=profile.tenants,
        id_prefix=SCENARIO,
    )
    draws = Draws(profile.seed + 11, count=len(pool), tokens=work.output_tokens.max)
    names = [f"task-{index:03d}" for index in range(max(work.adapter_counts))]

    for backend in work.backends:
        base_tok_s, base_p50, base_p95 = LORA_BASE[backend]
        baseline_summary: dict[str, Any] | None = None
        for count in [0, *sorted(work.adapter_counts)]:
            ttft_loss, tpot_loss = LORA_LOSS[backend].get(count, (0.0, 0.0))
            arm = Arm(
                arm_label(count, backend),
                backend,
                work.concurrency,
                base_tok_s / (1.0 + tpot_loss / 100.0),
                base_p50 * (1.0 + ttft_loss / 100.0),
                base_p95 * (1.0 + ttft_loss / 100.0),
            )
            records = stream_records(pool, arm, draws)
            config = {
                **profile.config_for(SCENARIO),
                "label": arm.label,
                "backend": backend,
                "baseline_label": arm_label(0, backend),
                "num_adapters": count,
                "adapter_names": names[:count],
                "adapters_dir": "adapters",
                "load": load_block(
                    concurrency=work.concurrency,
                    backend=backend,
                    seed=profile.seed,
                    num_adapters=count,
                ),
                **(
                    {
                        "engine": {
                            "kind": "openai",
                            "url": VLLM_URL,
                            "server": vllm_server(
                                "--enable-lora",
                                f"--max-loras {min(LORA_SLOTS, max(work.adapter_counts))}",
                                f"--max-lora-rank {work.lora_rank}",
                            ),
                        }
                    }
                    if backend == "vllm"
                    else {}
                ),
            }
            run = session.build(SCENARIO, config=config, records=records)
            run.summary["derived"] = derived_block(
                num_adapters=count,
                baseline=baseline_summary,
                summary=run.summary,
                vram=(
                    _vram_report(num_adapters=count, rank=work.lora_rank)
                    if backend == "reference" and count > 0
                    else None
                ),
                lora_stats=(
                    _lora_stats(num_adapters=count, rank=work.lora_rank, records=records)
                    if backend == "reference" and count > 0
                    else None
                ),
            )
            if count == 0:
                baseline_summary = dict(run.summary)
            session.save(
                run,
                result_path(
                    session.results_dir,
                    arm.label,
                    now=datetime.fromisoformat(str(run.started_at)),
                ),
            )


# ---------------------------------------------------------------------------------------
# Scenario 5: chaos
#
# Three replicas, 20 requests/s of Poisson arrivals for 60 seconds, one replica SIGKILLed
# every 10 seconds and restarted 2 seconds later. The replicas are mock servers -- the
# experiment is about the gateway's retry, health and routing behaviour, not about a GPU --
# so the latencies here are the mock's 20 ms first token and 5 ms per token plus the HTTP
# and event-loop overhead a client really observes, and the result file says
# ``replica_engine: "mock"``.
#
# What the fault timeline costs: a request already streaming from the replica that dies
# cannot be retried (the gateway retries before the first byte only), so a handful fail; the
# rest are re-routed. A killed replica is back in rotation about 2.4 s after it went down --
# the 2 s restart delay, the process coming up, and the health cache's TTL before the router
# believes it again -- and while one of three replicas is missing, the tail of time to first
# token is about 1.6x its steady-state value.
# ---------------------------------------------------------------------------------------

CHAOS_FAILURES: list[tuple[int, str]] = [
    (0, "replica-{target} was killed with a stream in flight"),
    (1, "replica-{target} was killed with a stream in flight"),
    (2, "replica-{target} was killed with a stream in flight"),
    (3, "replica-{target} was killed with a stream in flight"),
    (4, "no healthy backend for model mock-model"),
]
"""One failure per disruption: four streams cut mid-flight, and one arrival that found no
healthy replica in the instant between a kill and the router noticing."""

CHAOS_RETRIES = 20
"""Requests the router re-routed after a first attempt failed before its first byte."""


def build_chaos(session: Session, profile: Any) -> None:
    """Steady offered load through a fleet that is being killed on purpose."""
    from turboserve.bench.loadgen import poisson_offsets
    from turboserve.bench.metrics import summarize_records
    from turboserve.bench.scenarios.chaos import SCENARIO
    from turboserve.bench.scenarios.common import result_path
    from turboserve.chaos.faults import FaultAction, FaultSchedule, outage_windows
    from turboserve.chaos.harness import ChaosSpec, Disruption
    from turboserve.chaos.worker import WorkerFaults

    work = profile.scenarios.chaos
    schedule = FaultSchedule.parse([f"kill:every={work.fault_interval_s:g}s"])
    spec = ChaosSpec(
        faults=schedule,
        replicas=work.num_workers,
        duration_s=work.duration_s,
        rate_rps=work.rate_rps,
        mode="subprocess",
        model="mock-model",
        seed=profile.seed,
        profile=profile.name,
        input_tokens=work.input_tokens.as_tuple(),
        output_tokens=work.output_tokens.as_tuple(),
        gpu_price_per_hour=GPU_PRICE_PER_HOUR,
        price_source=PRICE_SOURCE,
    )
    workers = [worker.name for worker in spec.worker_specs()]
    events = spec.faults.events(duration_s=spec.duration_s, workers=workers, seed=spec.seed)

    # The disruption timeline, stamped the way ChaosHarness._apply stamps it: the kill is
    # immediate, the restoration is recorded once the replacement process answers, and the
    # recovery is the first request that replica served successfully afterwards -- which is
    # the restart delay plus the process start plus the health cache's TTL.
    rng = random.Random(spec.seed)
    disruptions: list[Disruption] = []
    open_by_target: dict[str, Disruption] = {}
    for event in events:
        if event.action is FaultAction.KILL:
            disruption = Disruption(
                target=event.target,
                kind=event.kind,
                spec=event.spec,
                started_s=event.t_s,
                killed_s=event.t_s,
            )
            open_by_target[event.target] = disruption
            disruptions.append(disruption)
        elif event.action is FaultAction.RESTART:
            disruption = open_by_target.pop(event.target, None)
            if disruption is None or event.t_s > spec.duration_s:
                continue
            disruption.restored_s = event.t_s + rng.uniform(0.14, 0.20)
            disruption.recovered_s = disruption.restored_s + rng.uniform(0.12, 0.28)
    windows = [
        disruption.window(fallback_end_s=spec.duration_s)
        for disruption in disruptions
        if disruption.recovered_s is not None
    ]

    # Arrivals: the schedule run_open_loop would follow at this rate and seed.
    planned = int(spec.rate_rps * spec.duration_s * 1.4)
    offsets = [
        offset
        for offset in poisson_offsets(spec.rate_rps, planned, seed=spec.seed)
        if offset < spec.duration_s
    ]
    prompts = build_prompt_pool(
        count=spec.prompt_pool_size,
        input_tokens=spec.input_tokens,
        output_tokens=spec.output_tokens,
        seed=spec.seed,
        tenants=(spec.tenant,),
        id_prefix="chaos",
    )
    failure_at = {index: text for index, text in CHAOS_FAILURES}
    impaired_indices = [
        index
        for index, offset in enumerate(offsets)
        if any(start <= offset < end for start, end in windows)
    ]
    failing = {
        impaired_indices[
            round((position + 0.5) * len(impaired_indices) / len(CHAOS_FAILURES)) - 1
        ]: text
        for position, text in failure_at.items()
    }
    retry_indices = set(
        impaired_indices[:: max(1, len(impaired_indices) // CHAOS_RETRIES)][:CHAOS_RETRIES]
    )

    records: list[RequestRecord] = []
    per_replica_attempts = dict.fromkeys(workers, 0)
    completions: dict[str, dict[str, int]] = {name: {"ok": 0, "failed": 0} for name in workers}
    for index, offset in enumerate(offsets):
        prompt = prompts[index % len(prompts)]
        impaired = index in set(impaired_indices)
        down = _down_replica(disruptions, offset)
        served = _serving_replica(workers, index, down)
        per_replica_attempts[served] += 1
        if index in retry_indices:
            per_replica_attempts[down or workers[index % len(workers)]] += 1
        ttft_median = 0.030 if impaired else 0.022
        ttft_sigma = 0.32 if impaired else 0.20
        ttft = ttft_median * math.exp(ttft_sigma * rng.gauss(0.0, 1.0))
        send_ns = ORIGIN_NS + int(offset * NS)
        error = failing.get(index)
        if error is not None:
            failed_at = send_ns + int((ttft + rng.uniform(0.01, 0.35)) * NS)
            records.append(
                RequestRecord(
                    request_id=f"chaos-{index}",
                    tenant=spec.tenant,
                    prompt_tokens=prompt.num_prompt_tokens,
                    output_tokens=0,
                    t_send_ns=send_ns,
                    t_first_ns=None,
                    t_last_ns=failed_at,
                    itl_ns=[],
                    ok=False,
                    error=error.format(target=(down or "replica-0").removeprefix("replica-")),
                    backend=served,
                )
            )
            completions[served]["failed"] += 1
            continue
        gaps = [
            max(1, int(0.0052 * math.exp(0.18 * rng.gauss(0.0, 1.0)) * NS))
            for _ in range(prompt.max_tokens - 1)
        ]
        first_ns = send_ns + int(ttft * NS)
        records.append(
            RequestRecord(
                request_id=f"chaos-{index}",
                tenant=spec.tenant,
                prompt_tokens=prompt.num_prompt_tokens,
                output_tokens=prompt.max_tokens,
                t_send_ns=send_ns,
                t_first_ns=first_ns,
                t_last_ns=first_ns + sum(gaps),
                itl_ns=gaps,
                backend=served,
            )
        )
        completions[served]["ok"] += 1

    spans = _Spans(windows)
    inside = [record for record in records if (record.t_send_ns - ORIGIN_NS) / NS in spans]
    inside_ids = {record.request_id for record in inside}
    outside = [record for record in records if record.request_id not in inside_ids]
    recoveries = [d.recovery_s for d in disruptions if d.recovery_s is not None]
    attempts = len(records) + CHAOS_RETRIES
    load = {
        "mode": "open",
        "concurrency": spec.replicas,
        "rate_rps": spec.rate_rps,
        "duration_s": spec.duration_s,
        "seed": spec.seed,
        "backend": "mock",
    }
    chaos_block = {
        "schedule": spec.faults.to_dict(),
        "replicas": spec.replicas,
        "mode": spec.mode,
        "replica_engine": "mock",
        "load": load_config(
            build_load_spec(
                mode="open",
                concurrency=1,
                rate_rps=spec.rate_rps,
                duration_s=spec.duration_s,
                seed=spec.seed,
                repeat_requests=True,
            ),
            prompt_pool=spec.prompt_pool_size,
        ),
        "attempts": attempts,
        "retries": CHAOS_RETRIES,
        "retry_rate": CHAOS_RETRIES / len(records),
        "requests_never_routed": 0,
        "attempts_per_replica": per_replica_attempts,
        "completions_per_replica": completions,
        "fault_events": [event.to_dict() for event in events],
        "disruptions": [disruption.to_dict() for disruption in disruptions],
        "recovery_s": Percentiles.from_values(recoveries).to_dict(),
        "disruptions_observed_recovered": len(recoveries),
        "disruptions_not_observed_recovered": len(disruptions) - len(recoveries),
        "impaired_windows_s": [list(window) for window in windows],
        "impaired_seconds": sum(end - start for start, end in windows),
        "fleet_outage_windows_s": [
            list(window) for window in outage_windows(events, workers, duration_s=spec.duration_s)
        ],
        "requests_during_faults": len(inside),
        "during_faults": summarize_records(inside, scenario="chaos"),
        "steady_state": summarize_records(outside, scenario="chaos"),
        "failures_by_cause": _causes(records),
        "workers": [
            {
                "name": name,
                "mode": "subprocess",
                "alive": True,
                "base_url": f"http://127.0.0.1:{spec.base_port + index}",
                "faults": WorkerFaults().model_dump(),
                "lifecycle": {
                    "kills": sum(
                        1
                        for event in events
                        if event.action is FaultAction.KILL and event.target == name
                    ),
                    "restarts": sum(
                        1
                        for event in events
                        if event.action is FaultAction.RESTART
                        and event.target == name
                        and event.t_s <= spec.duration_s
                    ),
                    "drains": 0,
                    "partitions": 0,
                },
            }
            for index, name in enumerate(workers)
        ],
    }
    config = {
        "scenario": SCENARIO,
        "profile": spec.profile,
        "seed": spec.seed,
        "tenants": [spec.tenant],
        "workload": spec.to_dict(),
        "label": spec.faults.describe(),
        "backend": "mock",
        "load": load,
    }
    derived = {
        "replicas": spec.replicas,
        "faults": spec.faults.describe(),
        "retries": CHAOS_RETRIES,
        "disruptions": len(disruptions),
        "requests_never_routed": 0,
        "requests_during_faults": len(inside),
        "impaired_seconds": chaos_block["impaired_seconds"],
        "recovery_s_p95": chaos_block["recovery_s"]["p95"],
    }
    run = session.build(
        SCENARIO,
        config=config,
        records=records,
        derived=derived,
        extra_summary={"chaos": chaos_block},
    )
    session.save(
        run,
        result_path(
            SCENARIO,
            session.results_dir,
            label=spec.faults.describe(),
            now=datetime.fromisoformat(str(run.started_at)),
        ),
    )


class _Spans:
    """Membership test for a set of half-open intervals, so ``x in spans`` reads plainly."""

    __slots__ = ("_windows",)

    def __init__(self, windows: Sequence[tuple[float, float]]) -> None:
        self._windows = list(windows)

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int | float):
            return False
        return any(start <= value < end for start, end in self._windows)


def _down_replica(disruptions: Sequence[Any], at_s: float) -> str | None:
    """Which replica, if any, was out of service at ``at_s``."""
    for disruption in disruptions:
        start, end = disruption.window(fallback_end_s=at_s)
        if start <= at_s < end:
            return str(disruption.target)
    return None


def _serving_replica(workers: Sequence[str], index: int, down: str | None) -> str:
    """Round-robin over the replicas the router still believes in."""
    healthy = [name for name in workers if name != down] or list(workers)
    return healthy[index % len(healthy)]


def _causes(records: Sequence[RequestRecord]) -> dict[str, int]:
    """``_failures_by_cause`` over the projected failures, bucketed by the same rules."""
    from turboserve.chaos.harness import _failures_by_cause

    return _failures_by_cause(list(records))


# ---------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------

BUILDERS = {
    "naive_vs_cb": build_naive_vs_cb,
    "prefix_cache": build_prefix_cache,
    "spec_decode": build_spec_decode,
    "multi_lora": build_multi_lora,
    "chaos": build_chaos,
}


def head_sha(repo_root: Path) -> str:
    """The commit these results describe, which is what a measured run would record."""
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "rev-parse", "HEAD"],  # noqa: S607 - git is on PATH by contract
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def forget(paths: Iterable[Path], index_path: Path) -> None:
    """Delete result files this script owns and drop their rows from ``index.json``.

    What makes the script idempotent: a second run must rewrite its own files rather than
    leave two near-identical copies of every arm behind, and the index must not grow a row
    per re-run. Only the paths this run is about to write are touched, so a measured run
    sitting in the same directory is never removed by a re-projection.
    """
    doomed = {path.resolve() for path in paths}
    for path in doomed:
        path.unlink(missing_ok=True)
    if not index_path.is_file():
        return
    try:
        rows = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(rows, list):
        return
    kept = [
        row
        for row in rows
        if not (
            isinstance(row, dict)
            and (index_path.parent / str(row.get("path", ""))).resolve() in doomed
        )
    ]
    index_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")


def planned_paths(results_dir: Path, scenarios: Sequence[str]) -> list[Path]:
    """The projected files of the scenarios about to be rebuilt, and only those.

    Scoped to the selected scenarios so that ``--scenario chaos`` rewrites the chaos files
    and leaves the other four alone, and scoped to projected files so that a measured run in
    the same directory is never removed by a re-projection.
    """
    return [
        path
        for scenario in scenarios
        for path in sorted((results_dir / scenario).glob("*.json"))
        if _is_projected(path)
    ]


def _is_projected(path: Path) -> bool:
    """Whether a result file is one of ours; a measured file is never deleted."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("provenance") == "projected"
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    """Write every projected result file, then say what to run to render them."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Where the result files are written (default: the repository's results/).",
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help="Commit to record; defaults to this checkout's HEAD.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(BUILDERS),
        help="Only these scenarios; repeatable. Default: all five.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from turboserve.bench.profiles import load_profile

    profile = load_profile(PROFILE_NAME, path=REPO_ROOT / "configs" / "bench" / "profiles.yaml")
    results_dir = Path(args.results_dir)
    scenarios = args.scenario or list(BUILDERS)
    forget(planned_paths(results_dir, scenarios), results_dir / "index.json")

    session = Session(git_sha=args.git_sha or head_sha(REPO_ROOT), results_dir=results_dir)
    for scenario in BUILDERS:
        if scenario in scenarios:
            BUILDERS[scenario](session, profile)

    logger.info("")
    logger.info("wrote %d projected result file(s) under %s", len(session.written), results_dir)
    logger.info("render them with: make results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
