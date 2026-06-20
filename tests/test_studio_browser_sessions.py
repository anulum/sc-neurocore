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
