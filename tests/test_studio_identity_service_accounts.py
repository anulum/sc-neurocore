# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity service-account auth

"""Bearer token authentication and store payload validation for service accounts."""

from __future__ import annotations

from tests.studio_identity_support import *  # noqa: F403

def test_studio_identity_store_authenticates_service_account_token(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path, "admin-token")
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))

    result = authenticator.authenticate_authorization_header("Bearer admin-token")

    assert result.principal is not None
    assert result.principal.principal_id == "svc-admin"
    assert result.principal.roles == frozenset({"studio.admin", "studio.viewer"})
    assert result.failure_reason is None

def test_studio_identity_store_rejects_invalid_bearer_token(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path, "admin-token")
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))

    result = authenticator.authenticate_authorization_header("Bearer wrong-token")

    assert result.principal is None
    assert result.failure_reason == "invalid_identity_token"

def test_studio_identity_store_treats_missing_authorization_as_neutral(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path, "admin-token")
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))

    result = authenticator.authenticate_authorization_header(None)

    assert result.principal is None
    assert result.failure_reason is None

def test_studio_identity_store_rejects_malformed_authorization_header(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path, "admin-token")
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))

    result = authenticator.authenticate_authorization_header("Basic admin-token")

    assert result.principal is None
    assert result.failure_reason == "invalid_identity_token"

def test_studio_identity_store_rejects_disabled_service_account_token(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    token_hash = hashlib.sha256(b"admin-token").hexdigest()
    _write_payload(
        identity_path,
        {
            "schema_version": "sc-neurocore.studio.identity.v1",
            "service_accounts": [
                {
                    "active": False,
                    "principal_id": "svc-admin",
                    "roles": ["studio.admin"],
                    "token_sha256": token_hash,
                }
            ],
        },
    )
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))

    result = authenticator.authenticate_authorization_header("Bearer admin-token")

    assert result.principal is None
    assert result.failure_reason == "disabled_identity_token"

def test_studio_identity_store_rejects_expired_service_account_token(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(
        identity_path,
        "admin-token",
        expires_at_utc="2020-01-01T00:00:00Z",
    )
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))

    result = authenticator.authenticate_authorization_header("Bearer admin-token")

    assert result.principal is None
    assert result.failure_reason == "expired_identity_token"

def test_studio_identity_store_accepts_future_expiry_without_z_suffix(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(
        identity_path,
        "admin-token",
        expires_at_utc="2030-01-01T00:00:00+00:00",
    )
    authenticator = StudioIdentityAuthenticator(
        load_studio_identity_store(identity_path),
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    result = authenticator.authenticate_authorization_header("Bearer admin-token")

    assert result.principal is not None
    assert result.failure_reason is None

def test_studio_identity_store_rejects_malformed_service_account_role(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin", ""],
                        "token_sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="roles"):
        load_studio_identity_store(identity_path)

@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([], "JSON object"),
        (
            {"schema_version": "sc-neurocore.studio.identity.v1"},
            "service_accounts",
        ),
        (
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": ["not-an-object"],
            },
            "service account 0",
        ),
        (
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {"principal_id": "", "roles": ["studio.admin"], "token_sha256": "0" * 64}
                ],
            },
            "principal_id",
        ),
        (
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {"principal_id": "svc-admin", "roles": [], "token_sha256": "0" * 64}
                ],
            },
            "roles",
        ),
        (
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": "not-a-hash",
                    }
                ],
            },
            "token_sha256",
        ),
        (
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "active": "yes",
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": "0" * 64,
                    }
                ],
            },
            "active",
        ),
        (
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "expires_at_utc": 42,
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": "0" * 64,
                    }
                ],
            },
            "UTC timestamp",
        ),
        (
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "expires_at_utc": "not-a-date",
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": "0" * 64,
                    }
                ],
            },
            "ISO timestamp",
        ),
        (
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "expires_at_utc": "2030-01-01T00:00:00",
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": "0" * 64,
                    }
                ],
            },
            "timezone",
        ),
    ],
)
def test_studio_identity_store_rejects_malformed_payloads(
    tmp_path: Path,
    payload: object,
    match: str,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_payload(identity_path, payload)

    with pytest.raises(ValueError, match=match):
        load_studio_identity_store(identity_path)

def test_studio_identity_store_rejects_unreadable_identity_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be read"):
        load_studio_identity_store(tmp_path)

def test_studio_identity_store_rejects_invalid_json(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    identity_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        load_studio_identity_store(identity_path)

def test_studio_identity_store_rejects_unsupported_schema(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    identity_path.write_text(
        json.dumps({"schema_version": "legacy", "service_accounts": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema"):
        load_studio_identity_store(identity_path)
