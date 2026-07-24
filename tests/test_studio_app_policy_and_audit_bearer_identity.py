# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (bearer_identity) from former test_studio_app_policy_and_audit.py

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403


def test_studio_app_accepts_bearer_identity_file_principal(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    audit_path = tmp_path / "audit" / "studio.jsonl"
    token_hash = hashlib.sha256(b"admin-token").hexdigest()
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin", "studio.viewer"],
                        "token_sha256": token_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            identity_file_path=str(identity_path),
            allow_header_principal=False,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/export",
        headers={"authorization": "Bearer admin-token"},
    )

    assert response.status_code == 200


def test_studio_app_rejects_invalid_bearer_identity_token(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    token_hash = hashlib.sha256(b"admin-token").hexdigest()
    audit_path = tmp_path / "audit" / "studio.jsonl"
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": token_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            identity_file_path=str(identity_path),
            allow_header_principal=False,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/export",
        headers={"authorization": "Bearer wrong-token"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_identity_token"
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["reason"] == "invalid_identity_token"


def test_studio_app_rejects_header_principal_when_fallback_disabled() -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            allow_header_principal=False,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/simulate",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
        json={},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "missing_principal"
