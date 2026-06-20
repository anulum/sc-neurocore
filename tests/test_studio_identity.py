# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity contract tests

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.identity import (
    StudioIdentityAuthenticator,
    add_studio_browser_user_record,
    load_studio_identity_store,
    make_browser_user_password_verifier,
    rotate_studio_browser_user_password,
    update_studio_browser_user_record,
    update_studio_identity_record,
    verify_browser_user_password,
)


def _write_identity_file(path: Path, token: str, *, expires_at_utc: str | None = None) -> None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    payload: dict[str, object] = {
        "schema_version": "sc-neurocore.studio.identity.v1",
        "service_accounts": [
            {
                "principal_id": "svc-admin",
                "roles": ["studio.admin", "studio.viewer"],
                "token_sha256": token_hash,
            }
        ],
    }
    if expires_at_utc is not None:
        accounts = payload["service_accounts"]
        assert isinstance(accounts, list)
        account = accounts[0]
        assert isinstance(account, dict)
        account["expires_at_utc"] = expires_at_utc
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_browser_user_authentication_rejects_empty_expired_and_unknown_users(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_payload(
        identity_path,
        {
            "browser_users": [
                {
                    "active": True,
                    "expires_at_utc": "2020-01-01T00:00:00Z",
                    "password_pbkdf2_sha256": make_browser_user_password_verifier(
                        "operator-password"
                    ),
                    "principal_id": "human-operator",
                    "roles": ["studio.admin"],
                    "username": "operator",
                }
            ],
            "schema_version": "sc-neurocore.studio.identity.v1",
            "service_accounts": [],
        },
    )
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))

    empty = authenticator.authenticate_browser_user("", "operator-password")
    expired = authenticator.authenticate_browser_user("operator", "operator-password")
    unknown = authenticator.authenticate_browser_user("missing", "operator-password")

    assert empty.failure_reason == "invalid_browser_login"
    assert expired.failure_reason == "expired_browser_user"
    assert unknown.failure_reason == "invalid_browser_login"


@pytest.mark.parametrize(
    ("browser_user", "match"),
    [
        ("not-an-object", "browser user 0"),
        (
            {
                "password_pbkdf2_sha256": make_browser_user_password_verifier("pw"),
                "principal_id": "human-operator",
                "roles": ["studio.admin"],
                "username": "",
            },
            "username",
        ),
        (
            {
                "password_pbkdf2_sha256": make_browser_user_password_verifier("pw"),
                "principal_id": "",
                "roles": ["studio.admin"],
                "username": "operator",
            },
            "principal_id",
        ),
        (
            {
                "password_pbkdf2_sha256": make_browser_user_password_verifier("pw"),
                "principal_id": "human-operator",
                "roles": [],
                "username": "operator",
            },
            "roles",
        ),
        (
            {
                "password_pbkdf2_sha256": "not-a-verifier",
                "principal_id": "human-operator",
                "roles": ["studio.admin"],
                "username": "operator",
            },
            "password verifier",
        ),
        (
            {
                "active": "yes",
                "password_pbkdf2_sha256": make_browser_user_password_verifier("pw"),
                "principal_id": "human-operator",
                "roles": ["studio.admin"],
                "username": "operator",
            },
            "active",
        ),
    ],
)
def test_studio_identity_store_rejects_malformed_browser_users(
    tmp_path: Path,
    browser_user: object,
    match: str,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_payload(
        identity_path,
        {
            "browser_users": [browser_user],
            "schema_version": "sc-neurocore.studio.identity.v1",
            "service_accounts": [],
        },
    )

    with pytest.raises(ValueError, match=match):
        load_studio_identity_store(identity_path)


def test_studio_identity_store_rejects_non_list_browser_users(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_payload(
        identity_path,
        {
            "browser_users": "operator",
            "schema_version": "sc-neurocore.studio.identity.v1",
            "service_accounts": [],
        },
    )

    with pytest.raises(ValueError, match="browser_users"):
        load_studio_identity_store(identity_path)


def test_identity_mutations_report_missing_records(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path, "admin-token")

    with pytest.raises(KeyError, match="missing-service"):
        update_studio_identity_record(
            identity_path,
            active=True,
            expires_at_utc=None,
            principal_id="missing-service",
            roles=["studio.admin"],
        )
    with pytest.raises(KeyError, match="missing-browser"):
        update_studio_browser_user_record(
            identity_path,
            active=True,
            expires_at_utc=None,
            roles=["studio.admin"],
            username="missing-browser",
        )
    with pytest.raises(KeyError, match="missing-browser"):
        rotate_studio_browser_user_password(
            identity_path,
            password="rotated-password",
            username="missing-browser",
        )


def test_browser_user_mutations_preserve_non_target_users(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_payload(
        identity_path,
        {
            "browser_users": [
                {
                    "active": True,
                    "expires_at_utc": None,
                    "password_pbkdf2_sha256": make_browser_user_password_verifier(
                        "operator-password"
                    ),
                    "principal_id": "human-operator",
                    "roles": ["studio.admin"],
                    "username": "operator",
                },
                {
                    "active": True,
                    "expires_at_utc": None,
                    "password_pbkdf2_sha256": make_browser_user_password_verifier(
                        "analyst-password"
                    ),
                    "principal_id": "human-analyst",
                    "roles": ["studio.viewer"],
                    "username": "analyst",
                },
            ],
            "schema_version": "sc-neurocore.studio.identity.v1",
            "service_accounts": [],
        },
    )

    update_studio_browser_user_record(
        identity_path,
        active=True,
        expires_at_utc=None,
        roles=["studio.admin", "studio.viewer"],
        username="operator",
    )
    rotate_studio_browser_user_password(
        identity_path,
        password="rotated-password",
        username="operator",
    )
    store = load_studio_identity_store(identity_path)
    authenticator = StudioIdentityAuthenticator(store)

    assert authenticator.authenticate_browser_user("operator", "rotated-password").principal
    assert authenticator.authenticate_browser_user("analyst", "analyst-password").principal


def test_add_browser_user_rejects_whitespace_username(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path, "admin-token")

    with pytest.raises(ValueError, match="whitespace"):
        add_studio_browser_user_record(
            identity_path,
            password="operator-password",
            principal_id="human-operator",
            roles=["studio.viewer"],
            username="bad user",
        )


@pytest.mark.parametrize(
    "encoded_verifier",
    [
        "sha1$390000$" + ("0" * 32) + "$" + ("0" * 64),
        "pbkdf2_sha256$not-int$" + ("0" * 32) + "$" + ("0" * 64),
        "pbkdf2_sha256$1$" + ("0" * 32) + "$" + ("0" * 64),
        "pbkdf2_sha256$390000$00$" + ("0" * 64),
        "pbkdf2_sha256$390000$" + ("0" * 32) + "$not-a-hash",
    ],
)
def test_browser_password_verifier_rejects_malformed_encodings(
    encoded_verifier: str,
) -> None:
    assert not verify_browser_user_password("password", encoded_verifier)
