# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio audit quarantine HTTP routes

"""Admin HTTP routes for archive, retention, purge, validate, and restore."""

from __future__ import annotations

from tests.studio_audit_quarantine_support import *  # noqa: F403

def test_studio_audit_quarantine_archive_route_writes_job_artifacts(
    tmp_path: Path,
) -> None:
    """Admin quarantine archive route writes confined archive artifacts."""

    audit_path = tmp_path / "audit" / "studio.jsonl"
    audit_path.parent.mkdir()
    audit_path.write_text('{"schema_version":"studio.audit.v1"}\n', encoding="utf-8")
    JsonlAuditSink(audit_path).record(
        AuditEvent(
            action="studio.test",
            route="/api/test",
            principal_id="operator",
            decision="allow",
            reason="authorized",
            request_id="req-test",
        )
    )
    app = create_app(
        StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/studio/audit/quarantine/archive",
        json={"limit": 10},
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )
    body = response.json()
    manager = _job_manager(app)
    archive_payload = _json_artifact(
        manager,
        body["job_id"],
        "evidence/audit-quarantine/archive.json",
    )
    manifest_payload = _json_artifact(
        manager,
        body["job_id"],
        "evidence/audit-quarantine/manifest.json",
    )

    assert response.status_code == 200
    assert body["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION
    assert body["archive_id"] == f"saqa_{body['job_id']}"
    assert body["summary"]["event_count"] == 1
    assert body["summary"]["reason_counts"] == {"legacy_or_unverifiable_rows": 1}
    assert len(body["artifacts"]) == 2
    assert archive_payload["summary"] == body["summary"]
    assert manifest_payload["summary"] == body["summary"]
    assert str(tmp_path) not in json.dumps(body)

def test_studio_audit_quarantine_archive_retention_route_lists_archive_jobs(
    tmp_path: Path,
) -> None:
    """Admin retention route returns path-free archive disposition."""

    audit_path = tmp_path / "audit" / "studio.jsonl"
    audit_path.parent.mkdir()
    audit_path.write_text('{"schema_version":"studio.audit.v1"}\n', encoding="utf-8")
    JsonlAuditSink(audit_path).record(
        AuditEvent(
            action="studio.test",
            route="/api/test",
            principal_id="operator",
            decision="allow",
            reason="authorized",
            request_id="req-test",
        )
    )
    app = create_app(
        StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")
    headers = {"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"}
    for _index in range(2):
        response = client.post(
            "/api/studio/audit/quarantine/archive",
            json={"limit": 10},
            headers=headers,
        )
        assert response.status_code == 200

    retention_response = client.get(
        "/api/studio/audit/quarantine/archive/retention?retain_latest=1",
        headers=headers,
    )
    body = retention_response.json()

    assert retention_response.status_code == 200
    assert body["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_RETENTION_SCHEMA_VERSION
    assert body["archive_count"] == 2
    assert body["retain_count"] == 1
    assert body["prune_candidate_count"] == 1
    assert body["skipped_record_count"] == 0
    entries = cast(list[dict[str, object]], body["entries"])
    assert {entry["disposition"] for entry in entries} == {"retain", "prune_candidate"}
    assert str(tmp_path) not in json.dumps(body)

def test_studio_audit_quarantine_archive_purge_route_removes_prune_candidates(
    tmp_path: Path,
) -> None:
    """Admin purge route deletes only archive jobs outside retention."""

    audit_path = tmp_path / "audit" / "studio.jsonl"
    audit_path.parent.mkdir()
    audit_path.write_text('{"schema_version":"studio.audit.v1"}\n', encoding="utf-8")
    JsonlAuditSink(audit_path).record(
        AuditEvent(
            action="studio.test",
            route="/api/test",
            principal_id="operator",
            decision="allow",
            reason="authorized",
            request_id="req-test",
        )
    )
    app = create_app(
        StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")
    headers = {"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"}
    archive_job_ids: list[str] = []
    for _index in range(2):
        response = client.post(
            "/api/studio/audit/quarantine/archive",
            json={"limit": 10},
            headers=headers,
        )
        assert response.status_code == 200
        archive_job_ids.append(cast(str, response.json()["job_id"]))

    purge_response = client.post(
        "/api/studio/audit/quarantine/archive/purge",
        json={"retain_latest": 1},
        headers=headers,
    )
    body = purge_response.json()
    manager = _job_manager(app)

    assert purge_response.status_code == 200
    assert body["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_PURGE_SCHEMA_VERSION
    assert body["purged_archive_count"] == 1
    assert body["retained_archive_count"] == 1
    assert [record.job_id for record in manager.list_records()] == [archive_job_ids[1]]
    assert not (tmp_path / "jobs" / archive_job_ids[0]).exists()
    assert (tmp_path / "jobs" / archive_job_ids[1]).is_dir()
    with pytest.raises(KeyError):
        manager.record(archive_job_ids[0])
    assert str(tmp_path) not in json.dumps(body)

def test_studio_audit_quarantine_archive_validate_route_accepts_archive_pair(
    tmp_path: Path,
) -> None:
    """Admin validation route accepts archive and manifest payloads."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    app = create_app(StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/studio/audit/quarantine/archive/validate",
        json={"archive": archive_payload, "manifest": manifest_payload},
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_VALIDATION_SCHEMA_VERSION
    assert body["valid"] is True
    assert body["errors"] == []
    assert str(tmp_path) not in json.dumps(body)

def test_studio_audit_quarantine_archive_restore_route_writes_job_artifacts(
    tmp_path: Path,
) -> None:
    """Admin restore route writes confined restore artifacts."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    app = create_app(
        StudioRuntimeSettings(
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/studio/audit/quarantine/archive/restore",
        json={"archive": archive_payload, "manifest": manifest_payload},
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )
    body = response.json()
    manager = _job_manager(app)
    restore_rows = _text_artifact(
        manager,
        body["job_id"],
        "evidence/audit-quarantine/restore.jsonl",
    )
    restore_manifest = _json_artifact(
        manager,
        body["job_id"],
        "evidence/audit-quarantine/restore-manifest.json",
    )

    assert response.status_code == 200
    assert body["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION
    assert body["archive_id"] == "saqa_sj_quarantine"
    assert body["summary"]["event_count"] == 1
    assert len(body["artifacts"]) == 2
    assert json.loads(restore_rows)["event_hash"] == "1" * 64
    assert restore_manifest["summary"] == body["summary"]
    assert str(tmp_path) not in json.dumps(body)

def test_studio_audit_quarantine_archive_restore_route_rejects_invalid_archive(
    tmp_path: Path,
) -> None:
    """Admin restore route returns validation errors without creating a job."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    manifest_payload["archive_id"] = "saqa_other"
    app = create_app(StudioRuntimeSettings(enforce_route_policies=True))
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post(
        "/api/studio/audit/quarantine/archive/restore",
        json={"archive": archive_payload, "manifest": manifest_payload},
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )
    body = response.json()

    assert response.status_code == 422
    assert body["detail"]["errors"] == ["manifest_archive_id_mismatch"]
    assert _job_manager(app).list_records() == ()

def test_studio_audit_quarantine_archive_routes_require_admin(
    tmp_path: Path,
) -> None:
    """Quarantine archive routes are denied without the admin role."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    audit_path = tmp_path / "audit" / "studio.jsonl"
    app = create_app(
        StudioRuntimeSettings(
            audit_log_path=str(audit_path),
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    archive_response = client.post(
        "/api/studio/audit/quarantine/archive",
        json={"limit": 10},
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
    )
    validate_response = client.post(
        "/api/studio/audit/quarantine/archive/validate",
        json={"archive": archive_payload, "manifest": manifest_payload},
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
    )
    retention_response = client.get(
        "/api/studio/audit/quarantine/archive/retention",
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
    )
    restore_response = client.post(
        "/api/studio/audit/quarantine/archive/restore",
        json={"archive": archive_payload, "manifest": manifest_payload},
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
    )
    purge_response = client.post(
        "/api/studio/audit/quarantine/archive/purge",
        json={"retain_latest": 1},
        headers={"x-studio-principal": "operator-1", "x-studio-roles": "studio.viewer"},
    )

    assert archive_response.status_code == 403
    assert archive_response.json()["detail"] == "missing_admin_role"
    assert validate_response.status_code == 403
    assert validate_response.json()["detail"] == "missing_admin_role"
    assert retention_response.status_code == 403
    assert retention_response.json()["detail"] == "missing_admin_role"
    assert restore_response.status_code == 403
    assert restore_response.json()["detail"] == "missing_admin_role"
    assert purge_response.status_code == 403
    assert purge_response.json()["detail"] == "missing_admin_role"
