"""Pluggable gateway backends: the :class:`Backend` contract and a name registry.

A backend is chosen by name -- from ``configs/models.yaml``, from a CLI flag, or from a
Helm value -- so the gateway needs a mapping from that name to a class. The registry here
is deliberately tiny: a dict, a decorator that fills it, and a lookup that imports the
built-in modules on demand.

The import-on-demand matters because the backends have very different costs.
``LocalEngineBackend`` pulls in torch and the whole reference engine; ``MockBackend``
pulls in nothing. A gateway fronting a remote vLLM server, or the kind end-to-end test
running the mock, must not pay for torch, so no concrete backend is imported at package
import time -- each registers itself when its module is first loaded.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, TypeVar

from turboserve.gateway.backends.protocol import (
    AdapterNotFoundError,
    Backend,
    BackendError,
    BackendOverloadedError,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
    GenerateRequest,
    ModelNotFoundError,
    NonRetryableBackendError,
    RetryableBackendError,
    StreamInterruptedError,
    TokenEvent,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "BACKENDS",
    "AdapterNotFoundError",
    "Backend",
    "BackendError",
    "BackendOverloadedError",
    "BackendRequestError",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "GenerateRequest",
    "ModelNotFoundError",
    "NonRetryableBackendError",
    "RetryableBackendError",
    "StreamInterruptedError",
    "TokenEvent",
    "available_backends",
    "get_backend_cls",
    "load_builtin_backends",
    "register_backend",
    "unregister_backend",
]

#: Registered backend classes by name. Populated by :func:`register_backend` when a
#: backend module is imported; read through :func:`get_backend_cls`.
BACKENDS: dict[str, type[Backend]] = {}

#: Modules that register a built-in backend. They are imported lazily by
#: :func:`load_builtin_backends`; a module that is not present yet is skipped, which is
#: what lets the gateway run with only the backends a given deployment actually ships.
BUILTIN_BACKEND_MODULES: tuple[str, ...] = (
    "turboserve.gateway.backends.local_engine",
    "turboserve.gateway.backends.openai_compat",
    "turboserve.gateway.backends.mock",
)

#: The attributes :class:`Backend` requires. Checked at registration so a typo in a method
#: name is a loud ``TypeError`` at import time instead of an ``AttributeError`` in the
#: middle of a tenant's stream.
_REQUIRED_MEMBERS: tuple[str, ...] = ("generate", "health", "models", "close")

_BackendT = TypeVar("_BackendT", bound=type)

_builtins_loaded = False


def register_backend(name: str, *, override: bool = False) -> Callable[[_BackendT], _BackendT]:
    """Class decorator that registers a backend implementation under ``name``.

    Refuses to shadow an existing registration unless ``override=True``: two backends
    answering to one name is a configuration bug that would otherwise depend on import
    order, which is exactly the kind of thing that differs between the test process and
    the server process.
    """
    if not name:
        raise ValueError("backend name must not be empty")

    def decorator(cls: _BackendT) -> _BackendT:
        missing = [member for member in _REQUIRED_MEMBERS if not hasattr(cls, member)]
        if missing:
            raise TypeError(
                f"{cls.__name__} cannot be registered as backend {name!r}: "
                f"missing {', '.join(missing)}"
            )
        if name in BACKENDS and not override and BACKENDS[name] is not cls:
            raise ValueError(
                f"backend {name!r} is already registered to {BACKENDS[name].__name__}; "
                "pass override=True to replace it"
            )
        BACKENDS[name] = cls
        logger.debug("registered backend %r -> %s", name, cls.__name__)
        return cls

    return decorator


def unregister_backend(name: str) -> None:
    """Remove a registration. Provided so tests can register a fake and clean up."""
    BACKENDS.pop(name, None)


def load_builtin_backends() -> list[str]:
    """Import the built-in backend modules once and return every registered name.

    A missing module is skipped and logged at debug level: the tuple above names the
    backends this package *may* ship, and a trimmed deployment (or a partially built
    checkout) legitimately has fewer. Any other import error propagates -- a backend that
    exists but fails to import is a real fault and must not be hidden.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return available_backends()
    for module in BUILTIN_BACKEND_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != module:
                raise
            logger.debug("built-in backend module %s is not present", module)
    _builtins_loaded = True
    return available_backends()


def get_backend_cls(name: str) -> type[Backend]:
    """Look up a registered backend class by name.

    Loads the built-in backends on a miss, so callers never have to import a backend
    module for its side effect. Raises :class:`KeyError` with the known names when the
    name is still unknown.
    """
    cls = BACKENDS.get(name)
    if cls is not None:
        return cls
    load_builtin_backends()
    cls = BACKENDS.get(name)
    if cls is None:
        known = ", ".join(available_backends()) or "none"
        raise KeyError(f"unknown backend {name!r}; registered backends: {known}")
    return cls


def available_backends() -> list[str]:
    """Names currently registered, sorted."""
    return sorted(BACKENDS)
