# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio app policy and audit surfaces

"""Route-policy enforcement, identity principal acceptance, and audit export contracts."""

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403

def test_studio_app_route_policy_enforcement_allows_public_route() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/health")

    assert response.status_code == 200

def test_studio_app_route_policy_enforcement_rejects_unclassified_route() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/unclassified")

    assert response.status_code == 403
    assert response.json()["detail"] == "unclassified_route"

def test_studio_app_route_policy_enforcement_rejects_missing_principal() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post("/api/simulate", json={})

    assert response.status_code == 401
    assert response.json()["detail"] == "missing_principal"

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

def test_studio_app_records_policy_events_to_configured_audit_log(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post("/api/simulate", json={})

    assert response.status_code == 401
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["action"] == "studio.simulation.run"
    assert row["decision"] == "deny"
    assert row["principal_id"] is None
    assert row["reason"] == "missing_principal"
    assert row["route"] == "/api/simulate"
    assert row["schema_version"] == "studio.audit.v1"
    assert row["previous_event_hash"] is None
    assert row["event_hash"] == _audit_event_hash(row)
    assert datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).tzinfo is UTC

def test_studio_app_exposes_safe_audit_status(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(runtime_settings=StudioRuntimeSettings(audit_log_path=str(audit_path)))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/studio/audit/status")

    assert response.status_code == 200
    assert response.json() == {
        "configured": True,
        "healthy": True,
        "integrity_error": None,
        "integrity_verified": True,
        "last_error": None,
        "latest_event_hash": None,
        "path_configured": True,
        "quarantine_reason": None,
        "quarantined_event_count": 0,
        "retained_event_count": 0,
        "sink_type": "jsonl",
    }
    assert str(tmp_path) not in response.text

def test_studio_app_exposes_unhealthy_audit_location_without_path(tmp_path: Path) -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(audit_log_path=str(tmp_path)))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/studio/audit/status")

    assert response.status_code == 200
    assert response.json()["configured"] is True
    assert response.json()["healthy"] is False
    assert response.json()["last_error"] == "AuditPathIsDirectory"
    assert str(tmp_path) not in response.text

def test_studio_app_fails_closed_when_policy_audit_append_fails(tmp_path: Path) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(tmp_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/simulate",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
        json={},
    )
    status_response = client.get("/api/studio/audit/status")

    assert response.status_code == 503
    assert response.json()["detail"] == "audit_append_failed"
    assert status_response.status_code == 200
    assert status_response.json()["healthy"] is False
    assert status_response.json()["last_error"] == "AuditPathIsDirectory"
    assert str(tmp_path) not in status_response.text

def test_studio_app_exports_audit_events_for_admin_without_paths(tmp_path: Path) -> None:
    from sc_neurocore.studio.platform import AuditEvent, JsonlAuditSink

    audit_path = tmp_path / "audit" / "studio.jsonl"
    JsonlAuditSink(audit_path).record(
        AuditEvent(
            action="studio.simulation.run",
            route="/api/simulate",
            principal_id="operator-1",
            decision="allow",
            reason="authorized",
            request_id="seed-request",
        )
    )
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/export",
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "studio.audit.export.v1"
    assert payload["configured"] is True
    assert payload["integrity_error"] is None
    assert payload["integrity_verified"] is True
    assert payload["latest_event_hash"] == payload["events"][-1]["event_hash"]
    assert payload["quarantine_reason"] is None
    assert payload["quarantined_event_count"] == 0
    assert payload["retained_event_count"] >= 1
    assert payload["sink_type"] == "jsonl"
    assert payload["event_count"] >= 1
    assert payload["events"][0]["action"] == "studio.simulation.run"
    assert str(tmp_path) not in response.text

def test_studio_app_exports_quarantined_audit_events_for_admin_without_paths(
    tmp_path: Path,
) -> None:
    from sc_neurocore.studio.platform import AuditEvent, JsonlAuditSink

    audit_path = tmp_path / "audit" / "studio.jsonl"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text('{"schema_version":"studio.audit.v1"}\n', encoding="utf-8")
    JsonlAuditSink(audit_path).record(
        AuditEvent(
            action="studio.simulation.run",
            route="/api/simulate",
            principal_id="operator-1",
            decision="allow",
            reason="authorized",
            request_id="seed-request",
        )
    )
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/quarantine/export",
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "studio.audit.quarantine.export.v1"
    assert payload["configured"] is True
    assert payload["event_count"] == 1
    assert payload["events"][0]["quarantine_reason"] == "legacy_or_unverifiable_rows"
    assert payload["quarantine_reason"] == "legacy_or_unverifiable_rows"
    assert payload["retained_event_count"] >= 2
    assert payload["sink_type"] == "jsonl"
    assert str(tmp_path) not in response.text

def test_studio_app_rejects_audit_export_without_admin_role(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/export",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "missing_admin_role"

def test_studio_app_rejects_quarantine_export_without_admin_role(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        "/api/studio/audit/quarantine/export",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "missing_admin_role"

def test_studio_app_route_policy_enforcement_allows_authenticated_principal() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/simulate",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
        json={},
    )

    assert response.status_code != 401

def test_studio_app_route_policy_enforcement_rejects_missing_admin_role() -> None:
    app = create_app(runtime_settings=StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/synth/run",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
        json={"verilog": "module top; endmodule", "target": "ice40"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "missing_admin_role"
