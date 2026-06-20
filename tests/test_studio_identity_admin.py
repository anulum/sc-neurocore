# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity administration tests

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.identity import (
    StudioIdentityAuthenticator,
    list_studio_browser_user_public_records,
    list_studio_identity_public_records,
    load_studio_identity_store,
    make_browser_user_password_verifier,
    rotate_studio_browser_user_password,
    update_studio_browser_user_record,
    update_studio_identity_record,
)
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings


def _write_identity_file(path: Path, *, active: bool = True) -> str:
    token = "admin-token"
    path.write_text(
        json.dumps(
            {
                "browser_users": [
                    {
                        "active": True,
                        "expires_at_utc": None,
                        "password_pbkdf2_sha256": make_browser_user_password_verifier(
                            "operator-password"
                        ),
                        "principal_id": "human-operator",
                        "roles": ["studio.admin", "studio.viewer"],
                        "username": "operator",
                    }
                ],
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "active": active,
                        "expires_at_utc": None,
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin", "studio.viewer"],
                        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    },
                    {
                        "active": True,
                        "principal_id": "svc-viewer",
                        "roles": ["studio.viewer"],
                        "token_sha256": hashlib.sha256(b"viewer-token").hexdigest(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return token


def _client(identity_path: Path, audit_path: Path) -> TestClient:
    app = create_app(
        StudioRuntimeSettings(
            allow_header_principal=False,
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            identity_file_path=str(identity_path),
        )
    )
    return TestClient(app, base_url="http://127.0.0.1")


def _admin_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "x-request-id": "identity-admin-test"}


def _audit_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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


def test_identity_admin_routes_are_admin_gated_and_audited(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    token = _write_identity_file(identity_path)
    client = _client(identity_path, audit_path)

    denied = client.get("/api/studio/identity/service-accounts")
    listed = client.get(
        "/api/studio/identity/service-accounts",
        headers=_admin_headers(token),
    )
    updated = client.patch(
        "/api/studio/identity/service-accounts/svc-admin",
        headers=_admin_headers(token),
        json={"active": True, "expires_at_utc": None, "roles": ["studio.viewer"]},
    )
    forbidden_after_role_change = client.get(
        "/api/studio/operator/status",
        headers=_admin_headers(token),
    )

    assert denied.status_code == 401
    assert denied.json()["detail"] == "missing_principal"
    assert listed.status_code == 200
    assert listed.json()["service_accounts"][0] == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "svc-admin",
        "roles": ["studio.admin", "studio.viewer"],
    }
    assert "token_sha256" not in listed.text
    assert updated.status_code == 200
    assert updated.json() == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "svc-admin",
        "roles": ["studio.viewer"],
    }
    assert forbidden_after_role_change.status_code == 403
    assert forbidden_after_role_change.json()["detail"] == "missing_admin_role"
    actions = [row["action"] for row in _audit_rows(audit_path)]
    assert "studio.identity.service_accounts.list" in actions
    assert "studio.identity.service_accounts.update" in actions
    assert "studio.identity.service_account.update" in actions


def test_browser_user_admin_routes_are_admin_gated_and_audited(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    token = _write_identity_file(identity_path)
    client = _client(identity_path, audit_path)

    denied = client.get("/api/studio/identity/browser-users")
    listed = client.get(
        "/api/studio/identity/browser-users",
        headers=_admin_headers(token),
    )
    detail = client.get(
        "/api/studio/identity/browser-users/operator",
        headers=_admin_headers(token),
    )
    updated = client.patch(
        "/api/studio/identity/browser-users/operator",
        headers=_admin_headers(token),
        json={"active": False, "expires_at_utc": None, "roles": ["studio.viewer"]},
    )
    login_after_disable = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "operator-password"},
    )

    assert denied.status_code == 401
    assert denied.json()["detail"] == "missing_principal"
    assert listed.status_code == 200
    assert listed.json()["browser_users"][0] == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "human-operator",
        "roles": ["studio.admin", "studio.viewer"],
        "username": "operator",
    }
    assert "password" not in listed.text
    assert detail.status_code == 200
    assert detail.json()["username"] == "operator"
    assert updated.status_code == 200
    assert updated.json() == {
        "active": False,
        "expires_at_utc": None,
        "principal_id": "human-operator",
        "roles": ["studio.viewer"],
        "username": "operator",
    }
    assert login_after_disable.status_code == 401
    assert login_after_disable.json()["detail"] == "disabled_browser_user"
    actions = [row["action"] for row in _audit_rows(audit_path)]
    assert "studio.identity.browser_users.list" in actions
    assert "studio.identity.browser_users.detail" in actions
    assert "studio.identity.browser_users.update" in actions
    assert "studio.identity.browser_user.update" in actions


def test_browser_user_create_route_is_admin_gated_password_free_and_audited(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    token = _write_identity_file(identity_path)
    client = _client(identity_path, audit_path)
    payload = {
        "active": True,
        "expires_at_utc": None,
        "password": "analyst-password",
        "principal_id": "human-analyst",
        "roles": ["studio.viewer"],
        "username": "analyst",
    }

    denied = client.post("/api/studio/identity/browser-users", json=payload)
    created = client.post(
        "/api/studio/identity/browser-users",
        headers=_admin_headers(token),
        json=payload,
    )
    duplicate = client.post(
        "/api/studio/identity/browser-users",
        headers=_admin_headers(token),
        json=payload,
    )
    login = client.post(
        "/api/studio/auth/login",
        json={"username": "analyst", "password": "analyst-password"},
    )
    listed = client.get(
        "/api/studio/identity/browser-users",
        headers=_admin_headers(token),
    )

    assert denied.status_code == 401
    assert denied.json()["detail"] == "missing_principal"
    assert created.status_code == 200
    assert created.json() == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "human-analyst",
        "roles": ["studio.viewer"],
        "username": "analyst",
    }
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "Studio browser user username already exists."
    assert login.status_code == 200
    assert login.json()["principal_id"] == "human-analyst"
    assert "analyst-password" not in identity_path.read_text(encoding="utf-8")
    assert "analyst-password" not in audit_path.read_text(encoding="utf-8")
    assert "password" not in listed.text
    actions = [row["action"] for row in _audit_rows(audit_path)]
    assert "studio.identity.browser_users.create" in actions
    assert "studio.identity.browser_user.create" in actions


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


def test_browser_user_password_rotation_revokes_sessions_and_is_audited(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    token = _write_identity_file(identity_path)
    client = _client(identity_path, audit_path)

    login = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "operator-password"},
    )
    session_token = login.json()["access_token"]
    rotated = client.post(
        "/api/studio/identity/browser-users/operator/password",
        headers=_admin_headers(token),
        json={"password": "rotated-password"},
    )
    old_session = client.get(
        "/api/studio/auth/session",
        headers={"authorization": f"Bearer {session_token}"},
    )
    old_password = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "operator-password"},
    )
    new_password = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "rotated-password"},
    )

    assert login.status_code == 200
    assert rotated.status_code == 200
    assert rotated.json() == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "human-operator",
        "roles": ["studio.admin", "studio.viewer"],
        "username": "operator",
    }
    assert old_session.status_code == 401
    assert old_session.json()["detail"] in {
        "invalid_browser_session",
        "invalid_identity_token",
    }
    assert old_password.status_code == 401
    assert old_password.json()["detail"] == "invalid_browser_login"
    assert new_password.status_code == 200
    assert "rotated-password" not in audit_path.read_text(encoding="utf-8")
    actions = [row["action"] for row in _audit_rows(audit_path)]
    assert "studio.identity.browser_users.password.rotate" in actions
    assert "studio.identity.browser_user.password.rotate" in actions


def test_identity_admin_routes_reject_unconfigured_store(tmp_path: Path) -> None:
    app = create_app(
        StudioRuntimeSettings(
            allow_header_principal=False,
            enforce_route_policies=False,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/studio/identity/service-accounts")

    assert response.status_code == 409
    assert response.json()["detail"] == "identity_store_unavailable"
