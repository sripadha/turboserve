"""OpenAI-compatible multi-tenant HTTP gateway: auth, quotas, routing, metrics, usage.

The public names are re-exported here, but *lazily*: importing
:mod:`turboserve.gateway.app` pulls in FastAPI, and importing a backend may pull in torch or
httpx. Several consumers only ever want one leaf of this package -- the chaos harness wants
the mock backend, the canary controller wants nothing at all -- so the module-level
``__getattr__`` below defers each import until the name is actually touched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "Authenticator",
    "ChatTemplate",
    "ChatTemplateCache",
    "GatewayMetrics",
    "GatewayOptions",
    "GatewayState",
    "LimiterRegistry",
    "ModelsFile",
    "PriceTable",
    "RateLimitExceeded",
    "Router",
    "Tenant",
    "TenantRegistry",
    "TokenBucket",
    "UsageRecord",
    "UsageTracker",
    "create_app",
    "gateway_app",
    "hash_api_key",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from turboserve.gateway.app import (
        GatewayOptions,
        GatewayState,
        create_app,
        gateway_app,
    )
    from turboserve.gateway.auth import Authenticator, hash_api_key
    from turboserve.gateway.chat_template import ChatTemplate, ChatTemplateCache
    from turboserve.gateway.limits import LimiterRegistry, RateLimitExceeded, TokenBucket
    from turboserve.gateway.metrics import GatewayMetrics
    from turboserve.gateway.router import ModelsFile, Router
    from turboserve.gateway.tenants import Tenant, TenantRegistry
    from turboserve.gateway.usage import PriceTable, UsageRecord, UsageTracker

#: Which submodule each exported name lives in, used by ``__getattr__``.
_EXPORTS: dict[str, str] = {
    "Authenticator": "turboserve.gateway.auth",
    "ChatTemplate": "turboserve.gateway.chat_template",
    "ChatTemplateCache": "turboserve.gateway.chat_template",
    "GatewayMetrics": "turboserve.gateway.metrics",
    "GatewayOptions": "turboserve.gateway.app",
    "GatewayState": "turboserve.gateway.app",
    "LimiterRegistry": "turboserve.gateway.limits",
    "ModelsFile": "turboserve.gateway.router",
    "PriceTable": "turboserve.gateway.usage",
    "RateLimitExceeded": "turboserve.gateway.limits",
    "Router": "turboserve.gateway.router",
    "Tenant": "turboserve.gateway.tenants",
    "TenantRegistry": "turboserve.gateway.tenants",
    "TokenBucket": "turboserve.gateway.limits",
    "UsageRecord": "turboserve.gateway.usage",
    "UsageTracker": "turboserve.gateway.usage",
    "create_app": "turboserve.gateway.app",
    "gateway_app": "turboserve.gateway.app",
    "hash_api_key": "turboserve.gateway.auth",
}


def __getattr__(name: str) -> Any:
    """Import the submodule that owns ``name`` on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """Make the lazy names discoverable by ``dir()`` and by tab completion."""
    return sorted(__all__)
