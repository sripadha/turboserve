"""Logging setup shared by the CLI, the gateway and the benchmark drivers.

Library code never configures logging; it only calls ``logging.getLogger(__name__)``.
Entry points call :func:`configure_logging` exactly once so that every process in the
repository emits the same line format, which the benchmark and chaos harnesses parse
when they collect worker output.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final, TextIO

LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S%z"

#: Loggers of third-party packages that are too chatty at ``INFO``/``DEBUG`` for our
#: benchmark output to stay readable.
_NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "matplotlib",
)


def resolve_level(level: int | str | None) -> int:
    """Translate a level name, a level number or ``None`` into a ``logging`` level.

    ``None`` falls back to ``$TURBOSERVE_LOG_LEVEL`` and finally to ``INFO`` so that a
    subprocess started by the chaos harness inherits the parent's verbosity.
    """
    if level is None:
        level = os.environ.get("TURBOSERVE_LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.strip().upper())
    if resolved is None:
        raise ValueError(f"unknown log level: {level!r}")
    return resolved


def configure_logging(
    level: int | str | None = None,
    *,
    stream: TextIO | None = None,
    force: bool = True,
) -> int:
    """Install a single stderr handler on the root logger and return the level used.

    ``force=True`` (the default) replaces handlers installed by an earlier call or by a
    dependency, which keeps repeated CLI invocations inside one pytest process from
    duplicating every line.
    """
    resolved = resolve_level(level)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logging.basicConfig(level=resolved, handlers=[handler], force=force)
    quiet = max(resolved, logging.WARNING)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(quiet)
    return resolved
