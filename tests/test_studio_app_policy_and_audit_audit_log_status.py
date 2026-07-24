# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (audit_log_status) from former test_studio_app_policy_and_audit.py

from __future__ import annotations

from tests.studio_settings_support import *  # noqa: F403


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
