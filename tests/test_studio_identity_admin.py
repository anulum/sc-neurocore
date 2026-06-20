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
    list_studio_identity_public_records,
    load_studio_identity_store,
    update_studio_identity_record,
)
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings


def _write_identity_file(path: Path, *, active: bool = True) -> str:
    token = "admin-token"
    path.write_text(
        json.dumps(
            {
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
