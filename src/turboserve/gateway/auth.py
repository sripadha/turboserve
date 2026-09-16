"""API-key authentication and per-tenant authorisation.

The gateway is multi-tenant, so every request must answer two questions before it costs a
GPU cycle: *who is this* (authentication, 401 on failure) and *may they do this*
(authorisation, 403 on failure). Keeping the two apart matters operationally -- a spike of
401s means a rotated key was not rolled out, a spike of 403s means a model or adapter was
removed from a tenant's allow-list -- and the metric labels reflect that split.

Keys are compared by SHA-256 digest, never in clear text (see
:mod:`turboserve.gateway.tenants` for why the stored form is hashed). Hashing the presented
key first also means the dictionary lookup that follows cannot leak key material through
timing: the attacker controls the input to a one-way function, not the comparison operand.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from turboserve.gateway.tenants import DEFAULT_TENANT_ID, Tenant, TenantRegistry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "AuthError",
    "Authenticator",
    "Forbidden",
    "Principal",
    "Unauthenticated",
    "hash_api_key",
]

_BEARER: Final = "bearer"


def hash_api_key(key: str) -> str:
    """Return the lowercase SHA-256 hex digest of an API key.

    This is the single definition used by the gateway, by the ``turboserve gateway
    hash-key`` command and by the comment at the top of ``configs/tenants.yaml``; it must
    match ``printf %s '<key>' | sha256sum`` exactly, so the key is encoded as UTF-8 with no
    trailing newline.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class AuthError(Exception):
    """Base class for a refusal that maps directly onto an HTTP status.

    Carries the OpenAI-shaped error ``type``/``code`` as well as the status, so the route
    layer can render the body without a second mapping table.
    """

    status_code: int = 401
    error_type: str = "invalid_request_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return self.message


class Unauthenticated(AuthError):
    """No usable credential was presented: missing, malformed, or unknown key (401)."""

    status_code = 401
    error_type = "invalid_request_error"


class Forbidden(AuthError):
    """The caller is known but not allowed to do this (403).

    Disabled tenants, models outside the allow-list and unknown adapters all land here, and
    all three are deliberately *not* 401: retrying with a different key is not the fix, and
    a client that treats them as 401 will loop.
    """

    status_code = 403
    error_type = "permission_error"


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller: a tenant plus an opaque handle on the key used.

    ``key_id`` is the first twelve characters of the key's digest. It is enough to tell two
    of a tenant's keys apart in a log line during a rotation, and it is not enough to
    reconstruct the key.
    """

    tenant: Tenant
    key_id: str = ""
    anonymous: bool = False

    @property
    def tenant_id(self) -> str:
        """Shorthand for ``principal.tenant.tenant_id``, the universal metric label."""
        return self.tenant.tenant_id


class Authenticator:
    """Turns an ``Authorization`` header into a :class:`Principal`, then gates actions.

    The key index is built once from the registry, so authenticating a request is one hash
    and one dictionary lookup. Rotating keys means building a new ``Authenticator`` from a
    new :class:`~turboserve.gateway.tenants.TenantRegistry` and swapping it into the app
    state; nothing here mutates.
    """

    __slots__ = ("_anonymous_tenant_id", "_index", "_registry", "_require_auth")

    def __init__(
        self,
        registry: TenantRegistry,
        *,
        require_auth: bool = True,
        anonymous_tenant_id: str = DEFAULT_TENANT_ID,
    ) -> None:
        self._registry = registry
        self._index: Mapping[str, Tenant] = registry.key_index()
        self._require_auth = require_auth
        self._anonymous_tenant_id = anonymous_tenant_id
        if not require_auth and registry.get(anonymous_tenant_id) is None:
            raise ValueError(
                f"authentication is disabled but the anonymous tenant "
                f"{anonymous_tenant_id!r} is not in the registry"
            )
        if require_auth and not self._index:
            logger.warning(
                "authentication is enabled but no tenant has an api key configured; "
                "every request will be rejected with 401"
            )

    @property
    def registry(self) -> TenantRegistry:
        """The tenant directory this authenticator was built from."""
        return self._registry

    @property
    def require_auth(self) -> bool:
        """Whether a credential is demanded; ``False`` attributes traffic to one tenant."""
        return self._require_auth

    # -- authentication ---------------------------------------------------------------

    @staticmethod
    def extract_bearer(authorization: str | None) -> str | None:
        """Pull the credential out of an ``Authorization`` header value.

        The scheme is matched case-insensitively because HTTP says schemes are, and a bare
        value with no scheme is accepted as well: several OpenAI-compatible clients and
        ``curl`` recipes send the key alone, and rejecting them buys nothing.
        """
        if authorization is None:
            return None
        value = authorization.strip()
        if not value:
            return None
        scheme, _, rest = value.partition(" ")
        if scheme.lower() == _BEARER:
            token = rest.strip()
            return token or None
        if " " in value:  # some other scheme (Basic, Digest, ...) -- not ours
            return None
        return value

    def authenticate(self, authorization: str | None) -> Principal:
        """Resolve a credential to a principal, or raise :class:`AuthError`.

        With authentication disabled the header is ignored entirely and every request is
        attributed to the anonymous tenant; that mode exists for closed deployments (the
        kind end-to-end test, the chaos harness) where the only consumer is the load
        generator and a key would be ceremony.
        """
        if not self._require_auth:
            tenant = self._registry[self._anonymous_tenant_id]
            return Principal(tenant=tenant, key_id="", anonymous=True)

        key = self.extract_bearer(authorization)
        if key is None:
            raise Unauthenticated(
                "missing API key; send 'Authorization: Bearer <key>'",
                code="missing_api_key",
            )
        digest = hash_api_key(key)
        matched = self._index.get(digest)
        if matched is None:
            logger.info("rejected unknown api key %s...", digest[:12])
            raise Unauthenticated("invalid API key", code="invalid_api_key")
        if not matched.enabled:
            raise Forbidden(
                f"tenant {matched.tenant_id!r} is disabled",
                code="tenant_disabled",
            )
        return Principal(tenant=matched, key_id=digest[:12])

    # -- authorisation ----------------------------------------------------------------

    def authorize_model(self, principal: Principal, model: str) -> None:
        """Raise :class:`Forbidden` unless the tenant may address ``model``."""
        if not principal.tenant.allows_model(model):
            raise Forbidden(
                f"tenant {principal.tenant_id!r} is not allowed to use model {model!r}",
                code="model_not_allowed",
            )

    def resolve_adapter(self, principal: Principal, adapter: str | None) -> str | None:
        """Map a tenant-visible adapter name to the backend's name for it.

        Returns ``None`` when no adapter was requested. An unknown name is a 403 rather than
        a 404 on purpose: adapter names are tenant-private, and answering "no such adapter"
        to an unauthorised caller would let them enumerate another tenant's adapters.
        """
        if adapter is None:
            return None
        target = principal.tenant.adapter_target(adapter)
        if target is None:
            raise Forbidden(
                f"adapter {adapter!r} is not available to tenant {principal.tenant_id!r}",
                code="adapter_not_allowed",
            )
        return target

    def visible_models(self, principal: Principal, models: list[str]) -> list[str]:
        """Filter a served-model list down to what this tenant may see.

        ``GET /v1/models`` must not advertise models the caller would be refused for: the
        list is what OpenAI clients populate their model pickers from.
        """
        return [name for name in models if principal.tenant.allows_model(name)]
