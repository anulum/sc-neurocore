# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio audit quarantine archive test support

"""Shared fixtures for Studio audit quarantine archive lifecycle tests."""

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


__all__ = [
    "UTC",
    "_quarantine_export_payload",
    "_archive_context",
    "_written_archive_pair",
    "_job_manager",
    "_json_artifact",
    "_text_artifact",
    "_archive_record",
    "_archive_result_for_job",
    "json",
    "threading",
    "Callable",
    "datetime",
    "timezone",
    "Path",
    "cast",
    "pytest",
    "FastAPI",
    "TestClient",
    "create_app",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_PURGE_SCHEMA_VERSION",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_RETENTION_SCHEMA_VERSION",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_VALIDATION_SCHEMA_VERSION",
    "AuditEvent",
    "JsonlAuditSink",
    "StudioJobContext",
    "StudioJobManager",
    "StudioJobRecord",
    "StudioJobStatus",
    "StudioRuntimeSettings",
    "build_studio_audit_quarantine_archive_retention_plan",
    "purge_studio_audit_quarantine_archive_prune_candidates",
    "validate_studio_audit_quarantine_archive",
    "write_studio_audit_quarantine_archive",
    "write_studio_audit_quarantine_restore",
]
