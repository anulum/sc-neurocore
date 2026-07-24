# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity admin HTTP routes

"""Admin-gated identity HTTP routes, auditing, and session revocation contracts."""

from __future__ import annotations

from tests.studio_identity_admin_support import *  # noqa: F403


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


def test_identity_admin_route_rejects_last_admin_removal(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    token = _write_single_service_admin_identity_file(identity_path)
    client = _client(identity_path, audit_path)

    response = client.patch(
        "/api/studio/identity/service-accounts/svc-sole-admin",
        headers=_admin_headers(token),
        json={"active": False, "expires_at_utc": None, "roles": ["studio.viewer"]},
    )

    assert response.status_code == 409
    assert "active unexpired studio.admin" in response.json()["detail"]
    records = list_studio_identity_public_records(identity_path)
    assert records[0].to_public_dict() == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "svc-sole-admin",
        "roles": ["studio.admin"],
    }
    actions = [row["action"] for row in _audit_rows(audit_path)]
    assert "studio.identity.service_accounts.update" in actions
    assert "studio.identity.service_account.update" not in actions


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
