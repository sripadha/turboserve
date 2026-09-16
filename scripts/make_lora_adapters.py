#!/usr/bin/env python
"""Run the synthetic LoRA adapter trainer from a checkout.

The implementation lives in :mod:`turboserve.engine.lora.make_adapters` so that it can be
imported by the engine's tests and exposed as ``turboserve lora make-adapters``. This
wrapper exists for the one case the installed entry point does not cover: running the
trainer straight out of a clone, e.g. on a freshly provisioned GPU instance.

    python scripts/make_lora_adapters.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --n 16 --rank 8 --steps 30 --out adapters/
"""

from __future__ import annotations

from turboserve.engine.lora.make_adapters import lora_app

if __name__ == "__main__":
    lora_app()
