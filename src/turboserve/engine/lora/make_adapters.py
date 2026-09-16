"""Train N small LoRA adapters on N distinct synthetic tasks.

The multi-adapter experiments need many adapters that are *different from each other* and
*cheap to produce*. Downloading a pile of community adapters would make the benchmark
depend on the Hub and on whatever those adapters happen to target; training them here makes
the whole thing reproducible from a seed, and lets the same script produce a handful of
three-step adapters over a tiny random model for the unit tests and a full set over
Qwen2.5 for a real run.

The tasks are deliberately trivial string transformations with a per-tenant marker
("rewrite in upper case", "answer as <<tenant-7>>", ...). What matters for a serving
benchmark is that each adapter moves the base model's logits in its own direction -- so
that a batch mixing adapters is doing genuinely different work per row and a bug that
serves the wrong slot is visible -- not that any of them is a useful model. The written
adapters are ordinary PEFT directories: ``adapter_config.json`` plus
``adapter_model.safetensors``, loadable by :mod:`turboserve.engine.lora.adapter`, by PEFT
itself, and by vLLM's ``--enable-lora``.

Usage::

    turboserve lora make-adapters --model Qwen/Qwen2.5-0.5B-Instruct \\
        --n 16 --rank 8 --steps 30 --out adapters/

``scripts/make_lora_adapters.py`` is a one-line wrapper around the same typer app, kept so
the trainer can be run from a checkout without installing the package.

Nothing here runs at import time, so the module can also be imported by a test to call
:func:`make_adapters` directly.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import torch
import typer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

logger = logging.getLogger("turboserve.make_lora_adapters")

__all__ = [
    "DEFAULT_TARGET_MODULES",
    "AdapterSpec",
    "SyntheticTask",
    "app",
    "build_tasks",
    "lora_app",
    "main",
    "make_adapters",
    "task_examples",
]

#: Projections the engine's LoRA layer wraps, and therefore the ones worth training.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Word pool the synthetic sentences are drawn from. Short, common tokens so that a tiny
#: tokenizer with a small effective vocabulary still produces varied sequences.
_WORDS: tuple[str, ...] = (
    "river",
    "signal",
    "copper",
    "garden",
    "window",
    "silent",
    "market",
    "paper",
    "bridge",
    "yellow",
    "matter",
    "listen",
    "planet",
    "orange",
    "letter",
    "summer",
    "candle",
    "forest",
    "island",
    "pocket",
    "ribbon",
    "shadow",
    "travel",
    "winter",
)

#: The transformations an adapter can be trained to perform. Each is a pure function of a
#: sentence, so an example is fully determined by (task, sentence) and the whole training
#: set is reproducible from the seed.
_TRANSFORMS: tuple[tuple[str, str, Callable[[str], str]], ...] = (
    ("upper", "Rewrite in upper case", str.upper),
    ("title", "Rewrite in title case", str.title),
    ("reverse", "Reverse the word order", lambda s: " ".join(reversed(s.split()))),
    ("exclaim", "Rewrite with an exclamation", lambda s: s + " !"),
    ("first", "Keep only the first word", lambda s: s.split()[0]),
    ("last", "Keep only the last word", lambda s: s.split()[-1]),
    ("double", "Repeat every word twice", lambda s: " ".join(w + " " + w for w in s.split())),
    ("sorted", "Sort the words alphabetically", lambda s: " ".join(sorted(s.split()))),
)


@dataclass(frozen=True, slots=True)
class SyntheticTask:
    """One adapter's task: a persona marker plus a deterministic transformation."""

    name: str
    """Adapter (and directory) name, e.g. ``tenant-3``."""

    marker: str
    """Per-tenant tag put in the prompt, so two adapters sharing a transform still differ."""

    transform: str
    """Key into :data:`_TRANSFORMS`."""

    instruction: str
    """Human-readable description, written into the manifest."""

    def render(self, sentence: str) -> tuple[str, str]:
        """``(prompt, completion)`` for one training sentence."""
        function = dict((key, fn) for key, _, fn in _TRANSFORMS)[self.transform]
        prompt = f"{self.marker} {self.instruction}: {sentence}\n"
        return prompt, function(sentence)

    def to_dict(self) -> dict[str, str]:
        """JSON-safe description for the manifest."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """What was written for one adapter, returned by :func:`make_adapters`."""

    name: str
    path: Path
    task: SyntheticTask
    final_loss: float
    num_steps: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe description for the manifest."""
        return {
            "name": self.name,
            "path": str(self.path),
            "task": self.task.to_dict(),
            "final_loss": self.final_loss,
            "num_steps": self.num_steps,
        }


def build_tasks(count: int, *, prefix: str = "tenant") -> list[SyntheticTask]:
    """``count`` distinct tasks, cycling the transforms and tagging each with its index."""
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    tasks: list[SyntheticTask] = []
    for index in range(count):
        key, instruction, _ = _TRANSFORMS[index % len(_TRANSFORMS)]
        tasks.append(
            SyntheticTask(
                name=f"{prefix}-{index}",
                marker=f"<<{prefix}-{index}>>",
                transform=key,
                instruction=instruction,
            )
        )
    return tasks


def task_examples(
    task: SyntheticTask, *, count: int, seed: int, words_per_sentence: int = 6
) -> list[tuple[str, str]]:
    """``count`` ``(prompt, completion)`` pairs for one task, reproducible from ``seed``."""
    rng = random.Random(f"{seed}:{task.name}")
    pairs: list[tuple[str, str]] = []
    for _ in range(count):
        sentence = " ".join(rng.choice(_WORDS) for _ in range(words_per_sentence))
        pairs.append(task.render(sentence))
    return pairs


def _resolve_device(name: str) -> torch.device:
    """``auto`` picks CUDA when there is one; anything else is taken literally."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _encode_batch(
    tokenizer: Any, pairs: Sequence[tuple[str, str]], *, max_length: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Tokenise prompt+completion pairs, masking the prompt out of the loss.

    Training on the prompt tokens as well would teach every adapter the same thing (the
    shared instruction template), which is the opposite of what a per-tenant adapter is
    for.
    """
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    for prompt, completion in pairs:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        ids = (prompt_ids + completion_ids)[:max_length]
        label = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
        if not ids:
            raise ValueError("the tokenizer produced an empty sequence for a training pair")
        input_ids.append(ids)
        labels.append(label)

    width = max(len(ids) for ids in input_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.eos_token_id or 0
    padded = torch.full((len(input_ids), width), int(pad_id), dtype=torch.long)
    label_tensor = torch.full((len(input_ids), width), -100, dtype=torch.long)
    mask = torch.zeros((len(input_ids), width), dtype=torch.long)
    for row, (ids, label) in enumerate(zip(input_ids, labels, strict=True)):
        padded[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        label_tensor[row, : len(label)] = torch.tensor(label, dtype=torch.long)
        mask[row, : len(ids)] = 1
    return {
        "input_ids": padded.to(device),
        "attention_mask": mask.to(device),
        "labels": label_tensor.to(device),
    }


def _train_one(
    base_model: Any,
    tokenizer: Any,
    task: SyntheticTask,
    *,
    rank: int,
    alpha: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    target_modules: Sequence[str],
    device: torch.device,
) -> tuple[Any, float]:
    """Attach a fresh LoRA to ``base_model``, train it on ``task``, return ``(peft model, loss)``.

    The base model is reused across adapters and never updated: ``get_peft_model`` freezes
    it and only the injected factors carry gradients, so training the 64th adapter costs
    the same as the first and no copy of the base weights is made.
    """
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed + hash(task.name) % 10_000)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    model = get_peft_model(base_model, config, adapter_name=task.name)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    loss_value = float("nan")
    for step in range(steps):
        pairs = task_examples(task, count=batch_size, seed=seed + step)
        batch = _encode_batch(tokenizer, pairs, max_length=max_length, device=device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach())
        logger.debug("%s step %d/%d loss %.4f", task.name, step + 1, steps, loss_value)
    model.eval()
    return model, loss_value


def make_adapters(
    model_path: str,
    out_dir: Path | str,
    *,
    count: int = 16,
    rank: int = 8,
    alpha: int | None = None,
    steps: int = 30,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    max_length: int = 64,
    seed: int = 20260916,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    prefix: str = "tenant",
    local_files_only: bool = False,
) -> list[AdapterSpec]:
    """Train ``count`` adapters over ``model_path`` and write them under ``out_dir``.

    Returns one :class:`AdapterSpec` per adapter and writes ``out_dir/manifest.json``
    describing the whole set, which is what a benchmark result embeds so that the adapters
    a run used can be regenerated exactly.

    Training is fp32 by default even on a GPU: LoRA factors are initialised near zero and
    a handful of fp16 steps without a gradient scaler routinely produce zero gradients,
    which yields adapters that load fine and do nothing.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_device = _resolve_device(device)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=local_files_only, use_fast=True
    )
    base: Any = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, local_files_only=local_files_only
    )
    base = base.to(torch_device)
    base.config.use_cache = False

    tasks = build_tasks(count, prefix=prefix)
    specs: list[AdapterSpec] = []
    for task in tasks:
        peft_model, loss = _train_one(
            base,
            tokenizer,
            task,
            rank=rank,
            alpha=alpha if alpha is not None else 2 * rank,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            seed=seed,
            target_modules=target_modules,
            device=torch_device,
        )
        adapter_dir = destination / task.name
        peft_model.save_pretrained(
            str(destination), selected_adapters=[task.name], safe_serialization=True
        )
        # ``get_peft_model`` wraps the base in place; unloading returns the untouched base
        # so the next adapter starts from the same weights rather than from a stack of
        # previously injected (frozen, but present) modules.
        base = peft_model.unload()
        base.config.use_cache = False
        specs.append(
            AdapterSpec(
                name=task.name, path=adapter_dir, task=task, final_loss=loss, num_steps=steps
            )
        )
        logger.info("wrote %s (final loss %.4f)", adapter_dir, loss)

    manifest = {
        "base_model": model_path,
        "count": count,
        "rank": rank,
        "alpha": alpha if alpha is not None else 2 * rank,
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "target_modules": list(target_modules),
        "device": str(torch_device),
        "dtype": str(dtype).removeprefix("torch."),
        "adapters": [spec.to_dict() for spec in specs],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return specs


lora_app = typer.Typer(
    name="lora",
    help="Multi-tenant LoRA tooling: train the synthetic adapter set the benchmarks use.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

#: Alias kept so ``scripts/make_lora_adapters.py`` (and anything that loaded that script by
#: path before the trainer moved into the package) still finds a runnable typer app.
app = lora_app


@lora_app.command("make-adapters")
def main(  # noqa: PLR0913 - a CLI is a flat list of options
    model: Annotated[
        str, typer.Option("--model", help="Base model repo id or local path.")
    ] = "Qwen/Qwen2.5-0.5B-Instruct",
    n: Annotated[int, typer.Option("--n", min=1, help="How many adapters to train.")] = 16,
    rank: Annotated[int, typer.Option("--rank", min=1, help="LoRA rank.")] = 8,
    steps: Annotated[int, typer.Option("--steps", min=1, help="Optimiser steps per adapter.")] = 30,
    out: Annotated[
        Path, typer.Option("--out", help="Directory to write the adapter directories into.")
    ] = Path("adapters"),
    alpha: Annotated[
        int | None, typer.Option("--alpha", min=1, help="LoRA alpha; default is 2 x rank.")
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 4,
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=0.0)] = 1e-3,
    max_length: Annotated[int, typer.Option("--max-length", min=8)] = 64,
    seed: Annotated[int, typer.Option("--seed")] = 20260916,
    device: Annotated[str, typer.Option("--device", help="'auto', 'cpu' or 'cuda'.")] = "auto",
    prefix: Annotated[
        str, typer.Option("--prefix", help="Adapter name prefix; directories are <prefix>-<i>.")
    ] = "tenant",
    target_modules: Annotated[
        list[str] | None,
        typer.Option("--target-module", help="Repeatable; defaults to the engine's seven."),
    ] = None,
    local_files_only: Annotated[bool, typer.Option("--local-files-only/--allow-download")] = False,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Train the adapters and write a manifest describing the set."""
    logging.basicConfig(level=log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    specs = make_adapters(
        model,
        out,
        count=n,
        rank=rank,
        alpha=alpha,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
        device=device,
        target_modules=tuple(target_modules) if target_modules else DEFAULT_TARGET_MODULES,
        prefix=prefix,
        local_files_only=local_files_only,
    )
    typer.echo(f"wrote {len(specs)} adapters to {Path(out).resolve()}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    app()
