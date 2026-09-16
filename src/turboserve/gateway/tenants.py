"""Tenant directory: who may call the gateway, with which models, under which quotas.

A *tenant* is the unit the whole system is labelled by: every metric, log line, usage
record and engine request carries a ``tenant_id``. This module holds the static half of
that -- the fleet data loaded from ``configs/tenants.yaml`` -- while the dynamic half
(token buckets, in-flight counters) lives in :mod:`turboserve.gateway.limits`.

Two decisions are worth stating, because they shape every call site:

* **Keys are stored hashed, never in clear text.** ``tenants.yaml`` is a config file that
  ends up in git, in a ConfigMap and in a Helm values file; a leaked copy must not be a
  leaked credential. Only the SHA-256 of each key is stored, and
  :mod:`turboserve.gateway.auth` hashes the presented key before looking it up.
* **An empty ``allowed_models`` means "every model this gateway serves".** The common case
  for a small deployment is one pool and no per-tenant restriction, and making that the
  default keeps the file short; a tenant that must be fenced in lists patterns explicitly.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TENANT_ID",
    "EXAMPLE_API_KEYS",
    "EXAMPLE_KEY_DIGESTS",
    "Tenant",
    "TenantConfigError",
    "TenantRegistry",
]

#: Tenant used when the gateway runs with authentication disabled (the kind end-to-end
#: test, the chaos harness and ``--no-require-auth`` local runs all land here).
DEFAULT_TENANT_ID = "default"

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

#: The throwaway keys shipped with ``configs/tenants.yaml`` so that a fresh clone can issue
#: a request without minting anything. Their plaintext is printed in that file's comments
#: and in ``docs/gateway.md``, so they are public: a deployment that copies the example file
#: and forgets to replace the digests is accepting credentials anyone can read off GitHub.
#: :meth:`TenantRegistry.tenants_with_example_keys` finds them, ``gateway config-check``
#: prints them, and the gateway logs a warning at startup when authentication is required.
EXAMPLE_API_KEYS: tuple[str, ...] = (
    "sk-turboserve-acme-dev",
    "sk-turboserve-acme-dev-rotating",
    "sk-turboserve-globex-dev",
    "sk-turboserve-labs-dev",
    "sk-turboserve-suspended-dev",
)

#: SHA-256 digests of :data:`EXAMPLE_API_KEYS`, in the form ``tenants.yaml`` stores.
EXAMPLE_KEY_DIGESTS: frozenset[str] = frozenset(
    hashlib.sha256(key.encode("utf-8")).hexdigest() for key in EXAMPLE_API_KEYS
)


class TenantConfigError(ValueError):
    """``tenants.yaml`` is missing, unreadable, or does not describe tenants."""


class Tenant(BaseModel):
    """One API consumer and everything the gateway enforces on its behalf.

    ``rpm``/``tpm`` are *per-minute* budgets turned into token buckets by
    :class:`~turboserve.gateway.limits.TenantLimiter`; ``None`` means unlimited, which is
    what the development default uses so that a fresh checkout is not rate limited into
    uselessness. ``priority`` and ``weight`` are passed through to the engine's scheduler:
    ``priority`` decides who is preempted first when the KV pool runs out -- it does not
    reorder admission -- and ``weight`` is the tenant's share under the
    ``tenant_fair`` scheduler policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    tenant_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("tenant_id", "id"),
        serialization_alias="id",
    )
    name: str = ""
    """Human-readable label for dashboards; never used for lookups."""

    keys_sha256: list[str] = Field(default_factory=list)
    """SHA-256 hex digests of this tenant's API keys.

    See :func:`turboserve.gateway.auth.hash_api_key` for how a key becomes a digest."""

    enabled: bool = True
    """A disabled tenant authenticates (the key is known) but is refused with 403."""

    rpm: int | None = Field(default=None, ge=1)
    tpm: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)
    allowed_models: list[str] = Field(default_factory=list)
    """``fnmatch`` patterns; empty means every model the gateway serves."""

    adapters: dict[str, str] = Field(default_factory=dict)
    """LoRA adapter name -> path or upstream model id the backend understands."""

    priority: int = 0
    weight: float = Field(default=1.0, gt=0.0)

    @field_validator("keys_sha256")
    @classmethod
    def _keys_are_hex_digests(cls, value: list[str]) -> list[str]:
        """Reject anything that is not a SHA-256 hex digest.

        This is the guard that catches the most likely and most damaging mistake in this
        file: pasting a *plain* API key where the digest belongs, which would both leak the
        key and silently never match a request.
        """
        cleaned: list[str] = []
        for entry in value:
            digest = entry.strip().lower()
            if not _SHA256_HEX.match(digest):
                raise ValueError(
                    "keys_sha256 entries must be 64-character sha256 hex digests; "
                    f"got {entry[:8]!r}... (hash the key, do not paste it)"
                )
            cleaned.append(digest)
        return cleaned

    def allows_model(self, model: str) -> bool:
        """Whether this tenant may address ``model`` (exact name or ``fnmatch`` pattern)."""
        if not self.allowed_models:
            return True
        return any(fnmatch.fnmatchcase(model, pattern) for pattern in self.allowed_models)

    def adapter_target(self, adapter: str) -> str | None:
        """Resolve a tenant-visible adapter name to what the backend expects, or ``None``.

        Adapters are namespaced per tenant on purpose: two tenants may both call their
        adapter ``support-bot`` and mean different weights.
        """
        return self.adapters.get(adapter)

    @property
    def model_patterns(self) -> tuple[str, ...]:
        """The configured patterns, or ``("*",)`` when the tenant is unrestricted."""
        return tuple(self.allowed_models) if self.allowed_models else ("*",)


#: The tenant a gateway with authentication disabled attributes every request to. Quotas
#: are unset, because the deployments that disable auth (kind e2e, chaos harness) are
#: closed systems measuring the engine, not the quota code.
_DEFAULT_TENANT = Tenant(tenant_id=DEFAULT_TENANT_ID, name="default (auth disabled)")


class TenantRegistry:
    """An immutable, id-indexed view of ``tenants.yaml``.

    Built once at startup and shared by the authenticator, the limiter registry and the
    ``/v1/models`` route. Reloading means building a new registry and swapping it in, which
    keeps every reader lock-free.
    """

    __slots__ = ("_tenants",)

    def __init__(self, tenants: Iterable[Tenant]) -> None:
        indexed: dict[str, Tenant] = {}
        for tenant in tenants:
            if tenant.tenant_id in indexed:
                raise TenantConfigError(f"duplicate tenant id {tenant.tenant_id!r}")
            indexed[tenant.tenant_id] = tenant
        self._tenants = indexed

    # -- construction ---------------------------------------------------------------

    @classmethod
    def default(cls) -> TenantRegistry:
        """Registry holding only the unrestricted ``default`` tenant."""
        return cls([_DEFAULT_TENANT])

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TenantRegistry:
        """Build from an already-parsed ``tenants.yaml`` document.

        Both shapes the file may take are accepted -- a list of tenant objects each with an
        ``id``, or a mapping from id to the rest of the fields -- because the mapping form
        reads better in a Helm values file while the list form reads better in git.
        """
        raw = data.get("tenants", data)
        if isinstance(raw, Mapping):
            entries = [{"id": key, **dict(value or {})} for key, value in raw.items()]
        elif isinstance(raw, list):
            entries = [dict(item) for item in raw]
        else:
            raise TenantConfigError(
                f"'tenants' must be a list or a mapping, got {type(raw).__name__}"
            )
        if not entries:
            raise TenantConfigError("no tenants defined")
        try:
            tenants = [Tenant.model_validate(entry) for entry in entries]
        except ValueError as exc:
            raise TenantConfigError(f"invalid tenant definition: {exc}") from exc
        return cls(tenants)

    @classmethod
    def from_yaml(cls, path: Path) -> TenantRegistry:
        """Load and validate a ``tenants.yaml``."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TenantConfigError(f"cannot read tenants file {path}: {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TenantConfigError(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise TenantConfigError(f"{path} must contain a YAML mapping")
        registry = cls.from_mapping(data)
        logger.info("loaded %d tenants from %s", len(registry), path)
        return registry

    # -- lookup ---------------------------------------------------------------------

    def get(self, tenant_id: str) -> Tenant | None:
        """Tenant by id, or ``None``."""
        return self._tenants.get(tenant_id)

    def __getitem__(self, tenant_id: str) -> Tenant:
        try:
            return self._tenants[tenant_id]
        except KeyError as exc:
            raise KeyError(f"unknown tenant {tenant_id!r}") from exc

    def __contains__(self, tenant_id: object) -> bool:
        return tenant_id in self._tenants

    def __iter__(self) -> Iterator[Tenant]:
        return iter(self._tenants.values())

    def __len__(self) -> int:
        return len(self._tenants)

    def __repr__(self) -> str:
        return f"TenantRegistry({', '.join(sorted(self._tenants))})"

    def ids(self) -> list[str]:
        """Tenant ids, sorted."""
        return sorted(self._tenants)

    def tenants_with_example_keys(self) -> list[str]:
        """Ids of tenants still holding one of the shipped example key digests, sorted.

        Empty for any configuration that has replaced them, which is what a real deployment
        must look like.
        """
        return sorted(
            tenant.tenant_id
            for tenant in self._tenants.values()
            if EXAMPLE_KEY_DIGESTS.intersection(tenant.keys_sha256)
        )

    def key_index(self) -> dict[str, Tenant]:
        """Map every configured key digest to its tenant.

        Built once by the authenticator. A digest shared by two tenants is a configuration
        error that would make authentication depend on file order, so it is rejected here
        rather than resolved arbitrarily at request time.
        """
        index: dict[str, Tenant] = {}
        for tenant in self._tenants.values():
            for digest in tenant.keys_sha256:
                owner = index.get(digest)
                if owner is not None and owner.tenant_id != tenant.tenant_id:
                    raise TenantConfigError(
                        f"api key digest {digest[:8]}... is claimed by both "
                        f"{owner.tenant_id!r} and {tenant.tenant_id!r}"
                    )
                index[digest] = tenant
        return index
