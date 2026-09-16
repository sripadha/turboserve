"""Authentication and tenant-directory behaviour: 401 vs 403, hashing, allow-lists."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from turboserve.gateway.auth import (
    Authenticator,
    Forbidden,
    Unauthenticated,
    hash_api_key,
)
from turboserve.gateway.tenants import (
    EXAMPLE_API_KEYS,
    EXAMPLE_KEY_DIGESTS,
    Tenant,
    TenantConfigError,
    TenantRegistry,
)

KEY = "sk-turboserve-test-key"
DIGEST = hashlib.sha256(KEY.encode()).hexdigest()


def make_tenant(**overrides: object) -> Tenant:
    """A tenant with one known key, overridable field by field."""
    data: dict[str, object] = {"tenant_id": "acme", "keys_sha256": [DIGEST]}
    data.update(overrides)
    return Tenant.model_validate(data)


def test_hash_api_key_matches_sha256sum() -> None:
    # The value operators paste into tenants.yaml comes from `printf %s | sha256sum`, which
    # hashes the bytes with no trailing newline; the two must not diverge.
    assert hash_api_key(KEY) == DIGEST
    assert hash_api_key(KEY) == hashlib.sha256(KEY.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("BEARER   abc  ", "abc"),
        ("abc", "abc"),
        ("Basic dXNlcjpwYXNz", None),
        ("Bearer ", None),
        ("   ", None),
        (None, None),
    ],
)
def test_extract_bearer(header: str | None, expected: str | None) -> None:
    assert Authenticator.extract_bearer(header) == expected


def test_missing_and_unknown_keys_are_401() -> None:
    auth = Authenticator(TenantRegistry([make_tenant()]))
    with pytest.raises(Unauthenticated) as missing:
        auth.authenticate(None)
    assert missing.value.status_code == 401
    assert missing.value.code == "missing_api_key"

    with pytest.raises(Unauthenticated) as unknown:
        auth.authenticate("Bearer not-a-real-key")
    assert unknown.value.code == "invalid_api_key"


def test_known_key_resolves_to_its_tenant() -> None:
    auth = Authenticator(TenantRegistry([make_tenant()]))
    principal = auth.authenticate(f"Bearer {KEY}")
    assert principal.tenant_id == "acme"
    # The key handle identifies the key without being usable as one.
    assert principal.key_id == DIGEST[:12]
    assert KEY not in principal.key_id
    assert principal.anonymous is False


def test_disabled_tenant_is_403_not_401() -> None:
    # 401 would tell the client to try another key; 403 correctly says the account is off.
    auth = Authenticator(TenantRegistry([make_tenant(enabled=False)]))
    with pytest.raises(Forbidden) as exc:
        auth.authenticate(f"Bearer {KEY}")
    assert exc.value.status_code == 403
    assert exc.value.code == "tenant_disabled"


def test_model_allow_list_is_403() -> None:
    auth = Authenticator(TenantRegistry([make_tenant(allowed_models=["Qwen/*-Instruct"])]))
    principal = auth.authenticate(f"Bearer {KEY}")
    auth.authorize_model(principal, "Qwen/Qwen2.5-7B-Instruct")
    with pytest.raises(Forbidden) as exc:
        auth.authorize_model(principal, "meta-llama/Llama-3-8B")
    assert exc.value.code == "model_not_allowed"


def test_empty_allow_list_permits_every_model() -> None:
    auth = Authenticator(TenantRegistry([make_tenant()]))
    principal = auth.authenticate(f"Bearer {KEY}")
    auth.authorize_model(principal, "anything-at-all")
    assert auth.visible_models(principal, ["a", "b"]) == ["a", "b"]


def test_visible_models_is_filtered_per_tenant() -> None:
    auth = Authenticator(TenantRegistry([make_tenant(allowed_models=["small-*"])]))
    principal = auth.authenticate(f"Bearer {KEY}")
    assert auth.visible_models(principal, ["small-a", "big-b", "small-c"]) == ["small-a", "small-c"]


def test_adapter_resolution_maps_names_and_refuses_unknown() -> None:
    auth = Authenticator(TenantRegistry([make_tenant(adapters={"support": "acme-support-r16"})]))
    principal = auth.authenticate(f"Bearer {KEY}")
    assert auth.resolve_adapter(principal, None) is None
    assert auth.resolve_adapter(principal, "support") == "acme-support-r16"
    with pytest.raises(Forbidden) as exc:
        auth.resolve_adapter(principal, "someone-elses-adapter")
    assert exc.value.code == "adapter_not_allowed"


def test_adapters_are_namespaced_per_tenant() -> None:
    other_key = "sk-other"
    tenants = TenantRegistry(
        [
            make_tenant(adapters={"support": "acme-support-r16"}),
            Tenant(
                tenant_id="globex",
                keys_sha256=[hash_api_key(other_key)],
                adapters={"support": "globex-support-r16"},
            ),
        ]
    )
    auth = Authenticator(tenants)
    acme = auth.authenticate(f"Bearer {KEY}")
    globex = auth.authenticate(f"Bearer {other_key}")
    assert auth.resolve_adapter(acme, "support") == "acme-support-r16"
    assert auth.resolve_adapter(globex, "support") == "globex-support-r16"


def test_auth_disabled_attributes_everything_to_the_anonymous_tenant() -> None:
    auth = Authenticator(TenantRegistry.default(), require_auth=False)
    principal = auth.authenticate(None)
    assert principal.tenant_id == "default"
    assert principal.anonymous is True
    # A bogus header is ignored rather than rejected in this mode.
    assert auth.authenticate("Bearer nonsense").tenant_id == "default"


def test_auth_disabled_requires_the_anonymous_tenant_to_exist() -> None:
    with pytest.raises(ValueError, match="anonymous tenant"):
        Authenticator(TenantRegistry([make_tenant()]), require_auth=False)


def test_plain_key_in_keys_sha256_is_rejected() -> None:
    # The mistake this guard exists for: pasting the key where the digest belongs.
    with pytest.raises(ValueError, match="sha256 hex digests"):
        Tenant(tenant_id="acme", keys_sha256=[KEY])


def test_duplicate_key_digest_across_tenants_is_a_config_error() -> None:
    tenants = TenantRegistry([make_tenant(), make_tenant(tenant_id="globex")])
    with pytest.raises(TenantConfigError, match="claimed by both"):
        tenants.key_index()


def test_duplicate_tenant_id_is_a_config_error() -> None:
    with pytest.raises(TenantConfigError, match="duplicate tenant id"):
        TenantRegistry([make_tenant(), make_tenant()])


def test_registry_reads_the_shipped_example_file() -> None:
    # configs/tenants.yaml is documentation as much as configuration: if it stops loading,
    # the quickstart in docs/gateway.md is wrong.
    path = Path(__file__).resolve().parents[2] / "configs" / "tenants.yaml"
    registry = TenantRegistry.from_yaml(path)
    assert "acme" in registry
    assert registry["suspended"].enabled is False
    auth = Authenticator(registry)
    assert auth.authenticate("Bearer sk-turboserve-acme-dev").tenant_id == "acme"
    with pytest.raises(Forbidden):
        auth.authenticate("Bearer sk-turboserve-suspended-dev")


def test_registry_accepts_the_mapping_form(tmp_path: Path) -> None:
    path = tmp_path / "tenants.yaml"
    path.write_text(
        f"tenants:\n  acme:\n    rpm: 10\n    keys_sha256: ['{DIGEST}']\n", encoding="utf-8"
    )
    registry = TenantRegistry.from_yaml(path)
    assert registry["acme"].rpm == 10


def test_registry_reports_a_bad_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"
    with pytest.raises(TenantConfigError, match="cannot read"):
        TenantRegistry.from_yaml(missing)
    empty = tmp_path / "empty.yaml"
    empty.write_text("tenants: []\n", encoding="utf-8")
    with pytest.raises(TenantConfigError, match="no tenants"):
        TenantRegistry.from_yaml(empty)


def test_the_shipped_example_keys_are_reported_as_example_keys() -> None:
    """A deployment that copies ``configs/tenants.yaml`` keeps public credentials.

    The plaintext of every digest in that file is written in its own comments, so the file
    is a convenience for a fresh clone and a hazard for anything else. ``config-check`` and
    the gateway's startup path warn about exactly this list.
    """
    path = Path(__file__).resolve().parents[2] / "configs" / "tenants.yaml"
    registry = TenantRegistry.from_yaml(path)
    assert registry.tenants_with_example_keys() == ["acme", "globex", "labs", "suspended"]
    # The digests really are the sha256 of the documented plaintexts, so the check cannot
    # rot into a list of hard-coded strings nobody can reproduce.
    assert {hash_api_key(key) for key in EXAMPLE_API_KEYS} == set(EXAMPLE_KEY_DIGESTS)


def test_a_configuration_with_its_own_keys_reports_nothing() -> None:
    registry = TenantRegistry([Tenant(id="real", keys_sha256=[hash_api_key("sk-minted-here")])])
    assert registry.tenants_with_example_keys() == []
