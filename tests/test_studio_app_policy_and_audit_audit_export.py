# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (audit_export) from former test_studio_app_policy_and_audit.py

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403


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
