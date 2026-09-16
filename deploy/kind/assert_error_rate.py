#!/usr/bin/env python3
"""Assert that a load-generator run file meets the chaos error-rate objective.

This is the gate of the kind end-to-end job. The load generator writes a result file whose
schema is :class:`turboserve.bench.records.RunResult`; this script reads one such file (or
stdin), prints the handful of numbers a human wants to see in CI output, and exits non-zero
when the run did not meet the objective.

It is deliberately dependency-free stdlib -- it runs on whatever Python the CI runner has,
outside the project's virtual environment, so it cannot import turboserve to do the
arithmetic and re-derives it from the raw records instead.

Two failure modes are checked, not one:

* the error rate exceeded the threshold, and
* the run is too small to mean anything.

The second check matters more than it looks. A run in which the gateway was never reached
has an error rate of 0.0 and would sail through a threshold-only assertion, turning a
completely broken deployment into a green build. ``--min-requests`` is what stops that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load(source: str) -> dict[str, Any]:
    """Read the run file from a path, or from stdin when ``source`` is ``-``."""
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object at the top level, got {type(data).__name__}")
    return data


def counts(run: dict[str, Any]) -> tuple[int, int]:
    """Return ``(total, failed)`` for the run.

    The summary block is preferred because it is what the report renderer reads, but a run
    that was interrupted before ``finish()`` still has its raw records, so those are the
    fallback. Disagreement between the two is not possible: both count the same list.
    """
    summary = run.get("summary") or {}
    total = summary.get("num_requests")
    failed = summary.get("num_failed")
    if isinstance(total, int) and isinstance(failed, int):
        return total, failed
    records = run.get("requests") or []
    return len(records), sum(1 for record in records if not record.get("ok", True))


def percentile_of(run: dict[str, Any], metric: str, name: str) -> float | None:
    """Pull one percentile out of the summary, if the run computed it."""
    value = ((run.get("summary") or {}).get(metric) or {}).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="path to the run JSON, or - for stdin")
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.005,
        help="fail above this fraction of failed requests (default: %(default)s)",
    )
    parser.add_argument(
        "--min-requests",
        type=int,
        default=1,
        help="fail below this many requests, so an empty run cannot pass (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        run = load(args.result)
    except (OSError, ValueError) as exc:
        print(f"FAIL: could not read the run file: {exc}", file=sys.stderr)
        return 2

    total, failed = counts(run)
    error_rate = failed / total if total else 1.0
    ttft_p95 = percentile_of(run, "ttft_ms", "p95")
    e2e_p95 = percentile_of(run, "e2e_ms", "p95")
    req_s = (run.get("summary") or {}).get("req_s")

    print("chaos end-to-end result")
    print(f"  scenario        {run.get('scenario', 'unknown')}")
    print(f"  requests        {total}")
    print(f"  failed          {failed}")
    print(f"  error rate      {error_rate:.4%} (limit {args.max_error_rate:.4%})")
    if isinstance(req_s, (int, float)):
        print(f"  requests/s      {req_s:.2f}")
    if ttft_p95 is not None:
        print(f"  ttft p95 (ms)   {ttft_p95:.1f}")
    if e2e_p95 is not None:
        print(f"  e2e p95 (ms)    {e2e_p95:.1f}")

    failures: list[str] = []
    if total < args.min_requests:
        failures.append(
            f"only {total} requests were recorded, expected at least {args.min_requests}"
        )
    if error_rate > args.max_error_rate:
        failures.append(f"error rate {error_rate:.4%} exceeds {args.max_error_rate:.4%}")

    for message in failures:
        print(f"FAIL: {message}", file=sys.stderr)
    if failures:
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
