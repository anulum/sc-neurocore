# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity admin record contracts

"""Public-record hygiene, update preservation, and last-admin refusal contracts."""

from __future__ import annotations

from tests.studio_identity_admin_support import *  # noqa: F403

def test_identity_public_records_never_expose_token_hashes(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path)

    records = list_studio_identity_public_records(identity_path)

    assert [record.principal_id for record in records] == ["svc-admin", "svc-viewer"]
    assert records[0].to_public_dict() == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "svc-admin",
        "roles": ["studio.admin", "studio.viewer"],
    }
    assert "token_sha256" not in json.dumps([record.to_public_dict() for record in records])

def test_identity_record_update_preserves_token_hash_and_reloads_auth(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path)
    before = json.loads(identity_path.read_text(encoding="utf-8"))
    original_hash = before["service_accounts"][0]["token_sha256"]

    updated = update_studio_identity_record(
        identity_path,
        active=False,
        expires_at_utc="2030-01-01T00:00:00Z",
        principal_id="svc-admin",
        roles=["studio.viewer", "studio.viewer"],
    )

    after = json.loads(identity_path.read_text(encoding="utf-8"))
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))
    auth_result = authenticator.authenticate_authorization_header("Bearer admin-token")
    assert updated.to_public_dict() == {
        "active": False,
        "expires_at_utc": "2030-01-01T00:00:00Z",
        "principal_id": "svc-admin",
        "roles": ["studio.viewer"],
    }
    assert after["service_accounts"][0]["token_sha256"] == original_hash
    assert auth_result.principal is None
    assert auth_result.failure_reason == "disabled_identity_token"

def test_browser_user_public_records_never_expose_password_verifiers(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path)

    records = list_studio_browser_user_public_records(identity_path)

    assert [record.username for record in records] == ["operator"]
    assert records[0].to_public_dict() == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "human-operator",
        "roles": ["studio.admin", "studio.viewer"],
        "username": "operator",
    }
    assert "password" not in json.dumps([record.to_public_dict() for record in records])

def test_browser_user_update_preserves_password_verifier_and_reloads_auth(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path)
    before = json.loads(identity_path.read_text(encoding="utf-8"))
    original_verifier = before["browser_users"][0]["password_pbkdf2_sha256"]

    updated = update_studio_browser_user_record(
        identity_path,
        active=False,
        expires_at_utc="2030-01-01T00:00:00Z",
        roles=["studio.viewer", "studio.viewer"],
        username="operator",
    )

    after = json.loads(identity_path.read_text(encoding="utf-8"))
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))
    auth_result = authenticator.authenticate_browser_user("operator", "operator-password")
    assert updated.to_public_dict() == {
        "active": False,
        "expires_at_utc": "2030-01-01T00:00:00Z",
        "principal_id": "human-operator",
        "roles": ["studio.viewer"],
        "username": "operator",
    }
    assert after["browser_users"][0]["password_pbkdf2_sha256"] == original_verifier
    assert auth_result.principal is None
    assert auth_result.failure_reason == "disabled_browser_user"

def test_service_account_update_refuses_to_remove_last_admin(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_single_service_admin_identity_file(identity_path)
    before = identity_path.read_text(encoding="utf-8")

    with pytest.raises(StudioIdentityLifecycleError, match="active unexpired studio.admin"):
        update_studio_identity_record(
            identity_path,
            active=False,
            expires_at_utc=None,
            principal_id="svc-sole-admin",
            roles=["studio.viewer"],
        )

    assert identity_path.read_text(encoding="utf-8") == before

def test_browser_user_update_refuses_to_remove_last_admin(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_single_browser_admin_identity_file(identity_path)
    before = identity_path.read_text(encoding="utf-8")

    with pytest.raises(StudioIdentityLifecycleError, match="active unexpired studio.admin"):
        update_studio_browser_user_record(
            identity_path,
            active=False,
            expires_at_utc=None,
            roles=["studio.viewer"],
            username="operator",
        )

    assert identity_path.read_text(encoding="utf-8") == before

def test_rotate_browser_user_password_preserves_public_metadata(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path)

    rotated = rotate_studio_browser_user_password(
        identity_path,
        password="rotated-password",
        username="operator",
    )
    store = load_studio_identity_store(identity_path)
    authenticator = StudioIdentityAuthenticator(store)
    old_login = authenticator.authenticate_browser_user("operator", "operator-password")
    new_login = authenticator.authenticate_browser_user("operator", "rotated-password")

    assert rotated.to_public_dict() == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "human-operator",
        "roles": ["studio.admin", "studio.viewer"],
        "username": "operator",
    }
    assert old_login.principal is None
    assert old_login.failure_reason == "invalid_browser_login"
    assert new_login.principal is not None
    assert "rotated-password" not in identity_path.read_text(encoding="utf-8")
