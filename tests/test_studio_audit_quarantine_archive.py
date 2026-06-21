# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio audit quarantine archive tests

"""Tests for Studio audit quarantine archive contracts."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    STUDIO_AUDIT_QUARANTINE_ARCHIVE_PURGE_SCHEMA_VERSION,
    STUDIO_AUDIT_QUARANTINE_ARCHIVE_RETENTION_SCHEMA_VERSION,
    STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION,
    STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION,
    STUDIO_AUDIT_QUARANTINE_ARCHIVE_VALIDATION_SCHEMA_VERSION,
    AuditEvent,
    JsonlAuditSink,
    StudioJobContext,
    StudioJobManager,
    StudioJobRecord,
    StudioJobStatus,
    StudioRuntimeSettings,
    build_studio_audit_quarantine_archive_retention_plan,
    purge_studio_audit_quarantine_archive_prune_candidates,
    validate_studio_audit_quarantine_archive,
    write_studio_audit_quarantine_archive,
    write_studio_audit_quarantine_restore,
)

UTC = timezone.utc


def _quarantine_export_payload() -> dict[str, object]:
    """Return a minimal path-free quarantine export payload."""

    return {
        "configured": True,
        "event_count": 1,
        "events": [
            {
                "action": "studio.test",
                "decision": "allow",
                "event_hash": "1" * 64,
                "previous_event_hash": None,
                "principal_id": "operator",
                "quarantine_reason": "legacy_or_unverifiable_rows",
                "reason": "authorized",
                "request_id": "req-test",
                "route": "/api/test",
                "schema_version": "studio.audit.v1",
                "timestamp_utc": "2026-06-20T00:00:00Z",
            }
        ],
        "quarantine_reason": "legacy_or_unverifiable_rows",
        "retained_event_count": 2,
        "schema_version": "studio.audit.quarantine.export.v1",
        "sink_type": "jsonl",
        "truncated": False,
    }


def _archive_context(tmp_path: Path) -> StudioJobContext:
    """Return a bounded job context for quarantine archive writer tests."""

    return StudioJobContext(
        job_id="sj_quarantine",
        work_dir=tmp_path / "job",
        cancel_event=threading.Event(),
        max_artifact_bytes=65536,
    )


def _written_archive_pair(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Write and return one archive payload with its manifest payload."""

    write_studio_audit_quarantine_archive(
        _archive_context(tmp_path),
        quarantine_export=_quarantine_export_payload(),
        clock=lambda: datetime(2026, 6, 21, tzinfo=UTC),
    )
    archive_payload = json.loads(
        (tmp_path / "job" / "evidence" / "audit-quarantine" / "archive.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_payload = json.loads(
        (tmp_path / "job" / "evidence" / "audit-quarantine" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(archive_payload, dict)
    assert isinstance(manifest_payload, dict)
    return cast(dict[str, object], archive_payload), cast(dict[str, object], manifest_payload)


def _job_manager(app: FastAPI) -> StudioJobManager:
    """Return the app-local Studio job manager."""

    return cast(StudioJobManager, app.state.studio_job_manager)


def _json_artifact(
    manager: StudioJobManager,
    job_id: str,
    relative_path: str,
) -> dict[str, object]:
    """Read one JSON job artifact through the verified artifact API."""

    payload = manager.read_artifact(job_id, relative_path)
    decoded = json.loads(payload.payload.decode("utf-8"))
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


def _text_artifact(
    manager: StudioJobManager,
    job_id: str,
    relative_path: str,
) -> str:
    """Read one text job artifact through the verified artifact API."""

    payload = manager.read_artifact(job_id, relative_path)
    return payload.payload.decode("utf-8")


def _archive_record(
    *,
    job_id: str,
    result: dict[str, object] | None,
    created_at_utc: str,
    finished_at_utc: str | None,
    owner: str = "studio-audit-quarantine",
    status: StudioJobStatus = "completed",
) -> StudioJobRecord:
    """Return a synthetic archive job record for retention-plan tests."""

    return StudioJobRecord(
        job_id=job_id,
        kind="evidence",
        owner=owner,
        request_id=None,
        status=status,
        execution_model="thread",
        created_at_utc=created_at_utc,
        finished_at_utc=finished_at_utc,
        result=result,
    )


def _archive_result_for_job(tmp_path: Path, job_id: str) -> dict[str, object]:
    """Return a public archive result for a synthetic job ID."""

    context = StudioJobContext(
        job_id=job_id,
        work_dir=tmp_path / job_id,
        cancel_event=threading.Event(),
        max_artifact_bytes=65536,
    )
    result = write_studio_audit_quarantine_archive(
        context,
        quarantine_export=_quarantine_export_payload(),
        clock=lambda: datetime(2026, 6, 21, tzinfo=UTC),
    ).to_public_dict()
    return cast(dict[str, object], result)


def test_write_studio_audit_quarantine_archive_writes_manifest_and_payload(
    tmp_path: Path,
) -> None:
    """Quarantine archive writer emits confined, path-free archive artifacts."""

    result = write_studio_audit_quarantine_archive(
        _archive_context(tmp_path),
        quarantine_export=_quarantine_export_payload(),
        clock=lambda: datetime(2026, 6, 21, tzinfo=UTC),
    )
    payload = result.to_public_dict()
    archive_payload = json.loads(
        (tmp_path / "job" / "evidence" / "audit-quarantine" / "archive.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_payload = json.loads(
        (tmp_path / "job" / "evidence" / "audit-quarantine" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION
    assert payload["archive_id"] == "saqa_sj_quarantine"
    assert result.artifact_paths == (
        "evidence/audit-quarantine/archive.json",
        "evidence/audit-quarantine/manifest.json",
    )
    assert archive_payload["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION
    assert archive_payload["quarantine_export"]["event_count"] == 1
    assert archive_payload["summary"]["reason_counts"] == {
        "legacy_or_unverifiable_rows": 1
    }
    assert manifest_payload["summary"] == archive_payload["summary"]
    assert str(tmp_path) not in json.dumps(payload)


def test_validate_studio_audit_quarantine_archive_accepts_writer_output(
    tmp_path: Path,
) -> None:
    """Archive validation accepts the writer's archive and manifest pair."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)

    validation = validate_studio_audit_quarantine_archive(
        archive_payload,
        manifest_payload=manifest_payload,
    ).to_public_dict()

    assert validation["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_VALIDATION_SCHEMA_VERSION
    assert validation["valid"] is True
    assert validation["archive_id"] == "saqa_sj_quarantine"
    assert validation["errors"] == []
    validation_summary = cast(dict[str, object], validation["summary"])
    assert validation_summary["event_count"] == 1
    assert validation_summary["reason_counts"] == {"legacy_or_unverifiable_rows": 1}
    assert str(tmp_path) not in json.dumps(validation)


def test_build_studio_audit_quarantine_archive_retention_plan_marks_old_archives(
    tmp_path: Path,
) -> None:
    """Retention planning marks newest valid archive jobs for retention."""

    old_result = _archive_result_for_job(tmp_path, "sj_old")
    new_result = _archive_result_for_job(tmp_path, "sj_new")
    records = (
        _archive_record(
            job_id="sj_old",
            result=old_result,
            created_at_utc="2026-06-20T00:00:00Z",
            finished_at_utc="2026-06-20T00:00:01Z",
        ),
        _archive_record(
            job_id="sj_other",
            result=new_result,
            created_at_utc="2026-06-21T00:00:00Z",
            finished_at_utc="2026-06-21T00:00:01Z",
            owner="studio-evidence",
        ),
        _archive_record(
            job_id="sj_failed",
            result=None,
            created_at_utc="2026-06-21T01:00:00Z",
            finished_at_utc="2026-06-21T01:00:01Z",
            status="failed",
        ),
        _archive_record(
            job_id="sj_new",
            result=new_result,
            created_at_utc="2026-06-22T00:00:00Z",
            finished_at_utc="2026-06-22T00:00:01Z",
        ),
    )

    plan = build_studio_audit_quarantine_archive_retention_plan(
        records,
        retain_latest=1,
    ).to_public_dict()

    assert plan["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_RETENTION_SCHEMA_VERSION
    assert plan["archive_count"] == 2
    assert plan["retain_count"] == 1
    assert plan["prune_candidate_count"] == 1
    assert plan["skipped_record_count"] == 1
    entries = cast(list[dict[str, object]], plan["entries"])
    assert entries[0]["job_id"] == "sj_new"
    assert entries[0]["disposition"] == "retain"
    assert entries[1]["job_id"] == "sj_old"
    assert entries[1]["disposition"] == "prune_candidate"
    assert str(tmp_path) not in json.dumps(plan)


def test_build_studio_audit_quarantine_archive_retention_plan_rejects_zero_retain(
    tmp_path: Path,
) -> None:
    """Retention planning fails closed on non-positive retain counts."""

    with pytest.raises(ValueError, match="archive_retention_retain_latest_invalid"):
        build_studio_audit_quarantine_archive_retention_plan(
            (
                _archive_record(
                    job_id="sj_archive",
                    result=_archive_result_for_job(tmp_path, "sj_archive"),
                    created_at_utc="2026-06-20T00:00:00Z",
                    finished_at_utc="2026-06-20T00:00:01Z",
                ),
            ),
            retain_latest=0,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.__setitem__("schema_version", "unsupported"),
        lambda result: result.__setitem__("archive_id", ""),
        lambda result: result.__setitem__("summary", []),
        lambda result: cast(dict[str, object], result["summary"]).__setitem__(
            "event_count",
            "1",
        ),
        lambda result: result.pop("artifact_paths", None),
        lambda result: result.__setitem__("artifact_paths", [""]),
    ],
)
def test_build_studio_audit_quarantine_archive_retention_plan_skips_malformed_jobs(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
) -> None:
    """Retention planning skips malformed archive job results."""

    result = _archive_result_for_job(tmp_path, "sj_malformed")
    mutation(result)

    plan = build_studio_audit_quarantine_archive_retention_plan(
        (
            _archive_record(
                job_id="sj_malformed",
                result=result,
                created_at_utc="2026-06-20T00:00:00Z",
                finished_at_utc="2026-06-20T00:00:01Z",
            ),
        ),
        retain_latest=1,
    ).to_public_dict()

    assert plan["archive_count"] == 0
    assert plan["skipped_record_count"] == 1
    assert plan["entries"] == []


def test_purge_studio_audit_quarantine_archive_prune_candidates_purges_old_jobs(
    tmp_path: Path,
) -> None:
    """Archive purge removes only retention prune candidates."""

    old_result = _archive_result_for_job(tmp_path, "sj_old")
    new_result = _archive_result_for_job(tmp_path, "sj_new")
    purged_job_ids: list[str] = []
    records = (
        _archive_record(
            job_id="sj_old",
            result=old_result,
            created_at_utc="2026-06-20T00:00:00Z",
            finished_at_utc="2026-06-20T00:00:01Z",
        ),
        _archive_record(
            job_id="sj_new",
            result=new_result,
            created_at_utc="2026-06-21T00:00:00Z",
            finished_at_utc="2026-06-21T00:00:01Z",
        ),
    )

    def purge_job(job_id: str) -> StudioJobRecord:
        purged_job_ids.append(job_id)
        return records[0]

    result = purge_studio_audit_quarantine_archive_prune_candidates(
        records,
        purge_job=purge_job,
        retain_latest=1,
    ).to_public_dict()

    assert result["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_PURGE_SCHEMA_VERSION
    assert result["purged_archive_count"] == 1
    assert result["retained_archive_count"] == 1
    assert result["skipped_record_count"] == 0
    assert purged_job_ids == ["sj_old"]
    purged_entries = cast(list[dict[str, object]], result["purged_entries"])
    retained_entries = cast(list[dict[str, object]], result["retained_entries"])
    assert purged_entries[0]["job_id"] == "sj_old"
    assert retained_entries[0]["job_id"] == "sj_new"
    assert str(tmp_path) not in json.dumps(result)


def test_write_studio_audit_quarantine_restore_writes_jsonl_and_manifest(
    tmp_path: Path,
) -> None:
    """Restore writer materializes validated archive rows as job artifacts."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    result = write_studio_audit_quarantine_restore(
        _archive_context(tmp_path / "restore"),
        archive_payload=archive_payload,
        manifest_payload=manifest_payload,
        clock=lambda: datetime(2026, 6, 22, tzinfo=UTC),
    )
    payload = result.to_public_dict()
    restore_root = tmp_path / "restore" / "job" / "evidence" / "audit-quarantine"
    restore_rows = restore_root.joinpath("restore.jsonl").read_text(encoding="utf-8")
    restore_manifest = json.loads(
        restore_root.joinpath("restore-manifest.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION
    assert payload["archive_id"] == "saqa_sj_quarantine"
    assert result.artifact_paths == (
        "evidence/audit-quarantine/restore.jsonl",
        "evidence/audit-quarantine/restore-manifest.json",
    )
    assert json.loads(restore_rows)["event_hash"] == "1" * 64
    assert restore_manifest["schema_version"] == STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION
    assert restore_manifest["summary"]["event_count"] == 1
    assert restore_manifest["summary"]["restored_at_utc"] == "2026-06-22T00:00:00Z"
    assert str(tmp_path) not in json.dumps(payload)


def test_write_studio_audit_quarantine_restore_rejects_invalid_archive(
    tmp_path: Path,
) -> None:
    """Restore writer rejects archives that fail validation."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    manifest_payload["archive_id"] = "saqa_other"

    with pytest.raises(ValueError, match="archive_restore_validation_failed"):
        write_studio_audit_quarantine_restore(
            _archive_context(tmp_path / "restore"),
            archive_payload=archive_payload,
            manifest_payload=manifest_payload,
        )


def test_validate_studio_audit_quarantine_archive_reports_manifest_mismatch(
    tmp_path: Path,
) -> None:
    """Archive validation reports mismatched companion manifests."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    manifest_payload["archive_id"] = "saqa_other"
    manifest_payload["summary"] = {}

    validation = validate_studio_audit_quarantine_archive(
        archive_payload,
        manifest_payload=manifest_payload,
    ).to_public_dict()

    assert validation["valid"] is False
    assert validation["archive_id"] == "saqa_sj_quarantine"
    assert validation["errors"] == [
        "manifest_archive_id_mismatch",
        "manifest_summary_mismatch",
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_errors"),
    [
        (
            lambda payload: payload.__setitem__(
                "schema_version",
                "studio.audit-quarantine-archive.v0",
            ),
            ["manifest_schema_unsupported"],
        ),
        (
            lambda payload: payload.__setitem__("artifact_count", 2),
            ["manifest_artifact_count_invalid"],
        ),
        (
            lambda payload: payload.__setitem__("entries", {}),
            ["manifest_archive_entry_missing"],
        ),
        (
            lambda payload: payload.__setitem__("entries", ["invalid"]),
            ["manifest_archive_entry_missing"],
        ),
        (
            lambda payload: payload.__setitem__(
                "entries",
                [{"type": "other", "bundle_path": "evidence/other.json"}],
            ),
            ["manifest_archive_entry_missing"],
        ),
    ],
)
def test_validate_studio_audit_quarantine_archive_reports_manifest_defects(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    expected_errors: list[str],
) -> None:
    """Archive validation reports stable manifest defect codes."""

    archive_payload, manifest_payload = _written_archive_pair(tmp_path)
    mutation(manifest_payload)

    validation = validate_studio_audit_quarantine_archive(
        archive_payload,
        manifest_payload=manifest_payload,
    ).to_public_dict()

    assert validation["valid"] is False
    assert validation["errors"] == expected_errors


@pytest.mark.parametrize(
    "manifest_payload",
    [
        {"bad": float("nan")},
        cast(dict[str, object], {1: "non-string-key"}),
        {"bad": object()},
    ],
)
def test_validate_studio_audit_quarantine_archive_rejects_non_json_manifest(
    tmp_path: Path,
    manifest_payload: dict[str, object],
) -> None:
    """Archive validation rejects manifests that cannot be JSON payloads."""

    archive_payload, _manifest_payload = _written_archive_pair(tmp_path)

    validation = validate_studio_audit_quarantine_archive(
        archive_payload,
        manifest_payload=manifest_payload,
    ).to_public_dict()

    assert validation["valid"] is False
    assert validation["errors"] == ["manifest_not_json"]


def test_validate_studio_audit_quarantine_archive_reports_summary_mismatch(
    tmp_path: Path,
) -> None:
    """Archive validation recomputes summary fields before import."""

    archive_payload, _manifest_payload = _written_archive_pair(tmp_path)
    archive_summary = cast(dict[str, object], archive_payload["summary"])
    archive_summary["event_count"] = 999

    validation = validate_studio_audit_quarantine_archive(archive_payload).to_public_dict()

    assert validation["valid"] is False
    assert validation["errors"] == ["archive_summary_mismatch"]
    validation_summary = cast(dict[str, object], validation["summary"])
    assert validation_summary["event_count"] == 1


def test_write_studio_audit_quarantine_archive_rejects_malformed_export(
    tmp_path: Path,
) -> None:
    """Quarantine archive writer rejects unsupported export schemas."""

    with pytest.raises(ValueError, match="export_schema_unsupported"):
        write_studio_audit_quarantine_archive(
            _archive_context(tmp_path),
            quarantine_export=_quarantine_export_payload()
            | {"schema_version": "studio.audit.export.v1"},
        )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        (lambda payload: payload.__setitem__("events", {}), "export_events_invalid"),
        (lambda payload: payload.__setitem__("event_count", 2), "export_event_count"),
        (lambda payload: payload.__setitem__("retained_event_count", 0), "retained"),
        (lambda payload: payload.__setitem__("truncated", "false"), "truncated"),
        (lambda payload: payload.__setitem__("quarantine_reason", 7), "reason"),
        (lambda payload: payload.__setitem__("events", ["invalid"]), "export_event"),
        (
            lambda payload: payload.__setitem__("events", [{"action": "studio.test"}]),
            "export_event",
        ),
    ],
)
def test_write_studio_audit_quarantine_archive_rejects_invalid_export_shapes(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    error_match: str,
) -> None:
    """Quarantine archive writer validates each public export field."""

    export_payload = _quarantine_export_payload()
    mutation(export_payload)

    with pytest.raises(ValueError, match=error_match):
        write_studio_audit_quarantine_archive(
            _archive_context(tmp_path),
            quarantine_export=export_payload,
        )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda payload: payload.__setitem__(
                "schema_version",
                "studio.audit-quarantine-archive.v0",
            ),
            "archive_schema_unsupported",
        ),
        (lambda payload: payload.__setitem__("archive_id", ""), "archive_id_invalid"),
        (
            lambda payload: payload.__setitem__("archived_at_utc", "2026-06-21"),
            "archive_timestamp_invalid",
        ),
        (
            lambda payload: payload.__setitem__("quarantine_export", {}),
            "export_schema_unsupported",
        ),
        (
            lambda payload: payload.__setitem__("quarantine_export", []),
            "archive_export_missing",
        ),
        (lambda payload: payload.__setitem__("summary", []), "archive_summary_missing"),
    ],
)
def test_validate_studio_audit_quarantine_archive_rejects_invalid_archive_shapes(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    expected_error: str,
) -> None:
    """Archive validation returns stable error codes for invalid archives."""

    archive_payload, _manifest_payload = _written_archive_pair(tmp_path)
    mutation(archive_payload)

    validation = validate_studio_audit_quarantine_archive(archive_payload).to_public_dict()

    assert validation["valid"] is False
    assert validation["errors"] == [expected_error]
    assert validation["summary"] is None


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
