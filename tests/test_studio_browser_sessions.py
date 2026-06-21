# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio browser session tests

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import Principal
from sc_neurocore.studio.platform.identity import (
    list_studio_browser_user_public_records,
    make_browser_user_password_verifier,
    verify_browser_user_password,
)
from sc_neurocore.studio.platform.sessions import StudioBrowserSessionManager
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings

UTC = timezone.utc


def _write_identity_file(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "browser_users": [
                    {
                        "active": True,
                        "expires_at_utc": None,
                        "password_pbkdf2_sha256": make_browser_user_password_verifier(
                            "browser-password"
                        ),
                        "principal_id": "user-operator",
                        "roles": ["studio.admin", "studio.viewer"],
                        "username": "operator",
                    }
                ],
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "active": True,
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": hashlib.sha256(b"service-token").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _client(identity_path: Path, audit_path: Path) -> TestClient:
    return TestClient(
        create_app(
            StudioRuntimeSettings(
                allow_header_principal=False,
                audit_log_path=str(audit_path),
                browser_session_ttl_seconds=600.0,
                enforce_route_policies=True,
                identity_file_path=str(identity_path),
            )
        ),
        base_url="http://127.0.0.1",
    )


def _throttled_client(identity_path: Path, audit_path: Path) -> TestClient:
    return TestClient(
        create_app(
            StudioRuntimeSettings(
                allow_header_principal=False,
                audit_log_path=str(audit_path),
                browser_login_cooldown_seconds=60.0,
                browser_login_failure_window_seconds=300.0,
                browser_login_max_failures=2,
                browser_session_ttl_seconds=600.0,
                enforce_route_policies=True,
                identity_file_path=str(identity_path),
            )
        ),
        base_url="http://127.0.0.1",
    )


def _audit_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_browser_user_password_verifier_rejects_wrong_password() -> None:
    verifier = make_browser_user_password_verifier("correct-password")

    assert verify_browser_user_password("correct-password", verifier)
    assert not verify_browser_user_password("wrong-password", verifier)
    assert not verify_browser_user_password("correct-password", "sha256$1$bad")


def test_browser_user_public_records_hide_password_verifiers(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path)

    records = list_studio_browser_user_public_records(identity_path)

    assert records[0].to_public_dict() == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "user-operator",
        "roles": ["studio.admin", "studio.viewer"],
        "username": "operator",
    }
    assert "password" not in json.dumps(records[0].to_public_dict())


def test_browser_session_manager_issues_and_revokes_bearer_session() -> None:
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    manager = StudioBrowserSessionManager(
        clock=lambda: now,
        token_factory=lambda: "session-token",
        ttl_seconds=60.0,
    )
    principal = Principal(principal_id="user-operator", roles=frozenset({"studio.admin"}))

    issued = manager.issue(principal)
    authenticated = manager.authenticate_authorization_header("Bearer session-token")
    revoked = manager.revoke_authorization_header("Bearer session-token")
    after_revoke = manager.authenticate_authorization_header("Bearer session-token")

    assert issued.to_public_dict() == {
        "access_token": "session-token",
        "expires_at_utc": "2026-06-20T12:01:00Z",
        "principal_id": "user-operator",
        "roles": ["studio.admin"],
        "token_type": "bearer",
    }
    assert authenticated.principal == principal
    assert revoked is True
    assert after_revoke.principal is None
    assert after_revoke.failure_reason == "invalid_browser_session"


def test_browser_session_manager_revokes_all_sessions_for_principal() -> None:
    tokens = iter(("session-token-a", "session-token-b", "session-token-c"))
    manager = StudioBrowserSessionManager(
        token_factory=lambda: next(tokens),
        ttl_seconds=60.0,
    )
    operator = Principal(principal_id="user-operator", roles=frozenset({"studio.admin"}))
    other = Principal(principal_id="other-user", roles=frozenset({"studio.viewer"}))
    manager.issue(operator)
    manager.issue(operator)
    manager.issue(other)

    revoked = manager.revoke_principal("user-operator")

    assert revoked == 2
    assert (
        manager.authenticate_authorization_header("Bearer session-token-a").failure_reason
        == "invalid_browser_session"
    )
    assert (
        manager.authenticate_authorization_header("Bearer session-token-b").failure_reason
        == "invalid_browser_session"
    )
    assert manager.authenticate_authorization_header("Bearer session-token-c").principal == other


def test_browser_session_manager_expires_sessions() -> None:
    current = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)

    def clock() -> datetime:
        return current

    manager = StudioBrowserSessionManager(
        clock=clock,
        token_factory=lambda: "session-token",
        ttl_seconds=60.0,
    )
    manager.issue(Principal(principal_id="user-operator", roles=frozenset({"studio.admin"})))
    current = current + timedelta(seconds=61)

    result = manager.authenticate_authorization_header("Bearer session-token")

    assert result.principal is None
    assert result.failure_reason == "invalid_browser_session"


def test_browser_login_session_logout_flow_is_audited(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    _write_identity_file(identity_path)
    client = _client(identity_path, audit_path)

    invalid = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "wrong-password"},
    )
    login = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "browser-password"},
    )
    token = login.json()["access_token"]
    auth_headers = {"Authorization": f"Bearer {token}"}
    session = client.get("/api/studio/auth/session", headers=auth_headers)
    admin_status = client.get("/api/studio/operator/status", headers=auth_headers)
    browser_users = client.get("/api/studio/identity/browser-users", headers=auth_headers)
    logout = client.post("/api/studio/auth/logout", headers=auth_headers)
    session_after_logout = client.get("/api/studio/auth/session", headers=auth_headers)

    assert invalid.status_code == 401
    assert invalid.json()["detail"] == "invalid_browser_login"
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert login.json()["principal_id"] == "user-operator"
    assert session.json() == {
        "authenticated": True,
        "principal_id": "user-operator",
        "roles": ["studio.admin", "studio.viewer"],
    }
    assert admin_status.status_code == 200
    assert browser_users.status_code == 200
    assert "password" not in browser_users.text
    assert logout.status_code == 200
    assert logout.json() == {"revoked": True}
    assert session_after_logout.status_code == 401
    actions = [row["action"] for row in _audit_rows(audit_path)]
    assert actions.count("studio.auth.login") == 2
    assert "studio.auth.logout" in actions
    assert "studio.identity.browser_users.list" in actions


def test_identity_lifecycle_routes_emit_allow_and_deny_audit_rows(
    tmp_path: Path,
) -> None:
    """Identity lifecycle APIs leave password-free allow and deny evidence."""

    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    _write_identity_file(identity_path)
    client = _client(identity_path, audit_path)
    admin_headers = {"authorization": "Bearer service-token"}

    create_viewer = client.post(
        "/api/studio/identity/browser-users",
        headers=admin_headers,
        json={
            "active": True,
            "expires_at_utc": None,
            "password": "viewer-password",
            "principal_id": "user-viewer",
            "roles": ["studio.viewer"],
            "username": "viewer",
        },
    )
    viewer_login = client.post(
        "/api/studio/auth/login",
        json={"username": "viewer", "password": "viewer-password"},
    )
    viewer_headers = {"authorization": f"Bearer {viewer_login.json()['access_token']}"}
    denied_update = client.patch(
        "/api/studio/identity/browser-users/operator",
        headers=viewer_headers,
        json={
            "active": True,
            "expires_at_utc": None,
            "roles": ["studio.admin", "studio.viewer"],
        },
    )
    update_viewer = client.patch(
        "/api/studio/identity/browser-users/viewer",
        headers=admin_headers,
        json={
            "active": False,
            "expires_at_utc": None,
            "roles": ["studio.viewer"],
        },
    )
    rotate_operator_password = client.post(
        "/api/studio/identity/browser-users/operator/password",
        headers=admin_headers,
        json={"password": "rotated-browser-password"},
    )
    update_service_account = client.patch(
        "/api/studio/identity/service-accounts/svc-admin",
        headers=admin_headers,
        json={
            "active": True,
            "expires_at_utc": None,
            "roles": ["studio.admin", "studio.viewer"],
        },
    )

    assert create_viewer.status_code == 200
    assert viewer_login.status_code == 200
    assert denied_update.status_code == 403
    assert denied_update.json()["detail"] == "missing_admin_role"
    assert update_viewer.status_code == 200
    assert rotate_operator_password.status_code == 200
    assert update_service_account.status_code == 200

    rows = _audit_rows(audit_path)
    lifecycle_rows = [
        row
        for row in rows
        if row["action"].startswith("studio.identity.browser_user.")
        or row["action"].startswith("studio.identity.service_account.")
    ]
    route_rows = [row for row in rows if row["action"].startswith("studio.identity.browser_users.")]

    assert {(row["action"], row["decision"], row["reason"]) for row in lifecycle_rows} >= {
        ("studio.identity.browser_user.create", "allow", "created:viewer"),
        ("studio.identity.browser_user.update", "allow", "updated:viewer"),
        (
            "studio.identity.browser_user.password.rotate",
            "allow",
            "rotated:operator:sessions_revoked:0",
        ),
        ("studio.identity.service_account.update", "allow", "updated:svc-admin"),
    }
    assert any(
        row["action"] == "studio.identity.browser_users.update"
        and row["decision"] == "deny"
        and row["principal_id"] == "user-viewer"
        and row["reason"] == "missing_admin_role"
        for row in route_rows
    )
    audit_text = audit_path.read_text(encoding="utf-8")
    assert "viewer-password" not in audit_text
    assert "rotated-browser-password" not in audit_text
    assert "service-token" not in audit_text


def test_browser_login_throttles_repeated_invalid_passwords(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    _write_identity_file(identity_path)
    client = _throttled_client(identity_path, audit_path)

    first = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "wrong-password"},
    )
    second = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "still-wrong"},
    )
    correct_while_locked = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "browser-password"},
    )
    operator_status = client.get(
        "/api/studio/operator/status",
        headers={"authorization": "Bearer service-token"},
    )

    assert first.status_code == 401
    assert first.json()["detail"] == "invalid_browser_login"
    assert second.status_code == 429
    assert second.json()["detail"] == "browser_login_throttled"
    assert second.headers["retry-after"] == "60"
    assert correct_while_locked.status_code == 429
    assert correct_while_locked.json()["detail"] == "browser_login_throttled"
    assert operator_status.status_code == 200
    browser_login_status = operator_status.json()["browser_login"]
    assert browser_login_status["active_bucket_count"] == 1
    assert browser_login_status["cooldown_seconds"] == 60.0
    assert browser_login_status["failure_window_seconds"] == 300.0
    assert browser_login_status["locked_bucket_count"] == 1
    assert browser_login_status["max_failures"] == 2
    assert 1 <= browser_login_status["max_retry_after_seconds"] <= 60
    assert "operator" not in json.dumps(browser_login_status)
    rows = [row for row in _audit_rows(audit_path) if row["action"] == "studio.auth.login"]
    assert [row["reason"] for row in rows[-3:]] == [
        "invalid_browser_login",
        "browser_login_throttled",
        "browser_login_throttled",
    ]
    assert "browser-password" not in audit_path.read_text(encoding="utf-8")


def test_browser_login_success_resets_prior_invalid_attempts(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    _write_identity_file(identity_path)
    client = _throttled_client(identity_path, audit_path)

    first_invalid = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "wrong-password"},
    )
    success = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "browser-password"},
    )
    invalid_after_success = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "wrong-again"},
    )

    assert first_invalid.status_code == 401
    assert success.status_code == 200
    assert invalid_after_success.status_code == 401
    assert invalid_after_success.json()["detail"] == "invalid_browser_login"


def test_disabled_browser_user_is_not_counted_toward_login_throttle(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "studio-audit.jsonl"
    _write_identity_file(identity_path)
    disabled = json.loads(identity_path.read_text(encoding="utf-8"))
    disabled["browser_users"][0]["active"] = False
    identity_path.write_text(json.dumps(disabled), encoding="utf-8")
    client = _throttled_client(identity_path, audit_path)

    first = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "browser-password"},
    )
    second = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "browser-password"},
    )

    assert first.status_code == 401
    assert first.json()["detail"] == "disabled_browser_user"
    assert second.status_code == 401
    assert second.json()["detail"] == "disabled_browser_user"
    assert "browser_login_throttled" not in audit_path.read_text(encoding="utf-8")
