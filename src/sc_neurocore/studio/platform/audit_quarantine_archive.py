# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio audit quarantine archive contracts

"""Audit quarantine archive contracts for SC-NeuroCore Studio."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypeAlias, cast

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobRecord
from sc_neurocore.studio.platform.policy import AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION

STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION = "studio.audit-quarantine-archive.v1"
STUDIO_AUDIT_QUARANTINE_ARCHIVE_VALIDATION_SCHEMA_VERSION = (
    "studio.audit-quarantine-archive.validation.v1"
)
STUDIO_AUDIT_QUARANTINE_ARCHIVE_RETENTION_SCHEMA_VERSION = (
    "studio.audit-quarantine-archive.retention.v1"
)
STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION = (
    "studio.audit-quarantine-archive.restore.v1"
)
STUDIO_AUDIT_QUARANTINE_ARCHIVE_OWNER = "studio-audit-quarantine"
STUDIO_AUDIT_QUARANTINE_RESTORE_OWNER = "studio-audit-quarantine-restore"
STUDIO_AUDIT_QUARANTINE_ARCHIVE_KIND = "evidence"
UTC = timezone.utc

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class StudioAuditQuarantineArchiveResult:
    """Path-free result returned after writing a quarantine archive.

    Parameters
    ----------
    archive_id:
        Stable archive identifier derived from the evidence job ID.
    manifest:
        JSON manifest describing the archive artifacts written by the job.
    summary:
        Path-free aggregate counts for operator review.
    artifact_paths:
        Archive-relative artifact paths written through the Studio job context.
    """

    archive_id: str
    manifest: dict[str, JsonValue]
    summary: dict[str, JsonValue]
    artifact_paths: tuple[str, ...]

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the path-free quarantine archive result."""

        return {
            "archive_id": self.archive_id,
            "artifact_paths": list(self.artifact_paths),
            "manifest": self.manifest,
            "schema_version": STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class StudioAuditQuarantineArchiveValidation:
    """Path-free validation result for one quarantine archive import candidate.

    Parameters
    ----------
    valid:
        Whether the supplied archive and optional manifest satisfy the import
        contract.
    archive_id:
        Archive identifier when it can be read safely.
    summary:
        Recomputed path-free summary for a valid archive candidate.
    errors:
        Stable validation error codes for operator remediation.
    warnings:
        Stable validation warning codes for non-blocking operator review.
    """

    valid: bool
    archive_id: str | None
    summary: dict[str, JsonValue] | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    schema_version: str = STUDIO_AUDIT_QUARANTINE_ARCHIVE_VALIDATION_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the path-free validation result."""

        return {
            "archive_id": self.archive_id,
            "errors": list(self.errors),
            "schema_version": self.schema_version,
            "summary": self.summary,
            "valid": self.valid,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class StudioAuditQuarantineArchiveRetentionEntry:
    """Path-free retention disposition for one quarantine archive job.

    Parameters
    ----------
    archive_id:
        Stable archive identifier returned by the archive job.
    job_id:
        Studio job identifier that owns the archive artifacts.
    created_at_utc:
        Job creation timestamp from the job manager record.
    finished_at_utc:
        Terminal job timestamp when available.
    event_count:
        Number of quarantined audit rows captured in the archive.
    retained_event_count:
        Number of retained audit rows visible to the source quarantine export.
    artifact_paths:
        Path-free artifact identifiers declared by the archive job.
    disposition:
        Operator retention decision for this archive.
    summary:
        Archive summary copied from the validated job result.
    """

    archive_id: str
    job_id: str
    created_at_utc: str
    finished_at_utc: str | None
    event_count: int
    retained_event_count: int
    artifact_paths: tuple[str, ...]
    disposition: str
    summary: dict[str, JsonValue]

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return this retention entry as a path-free JSON object."""

        return {
            "archive_id": self.archive_id,
            "artifact_paths": list(self.artifact_paths),
            "created_at_utc": self.created_at_utc,
            "disposition": self.disposition,
            "event_count": self.event_count,
            "finished_at_utc": self.finished_at_utc,
            "job_id": self.job_id,
            "retained_event_count": self.retained_event_count,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class StudioAuditQuarantineArchiveRetentionPlan:
    """Path-free retention inventory for quarantine archive jobs.

    Parameters
    ----------
    entries:
        Archive job entries sorted newest first.
    retain_latest:
        Number of newest archives marked for retention.
    skipped_record_count:
        Number of archive-owner job records that were incomplete or malformed.
    """

    entries: tuple[StudioAuditQuarantineArchiveRetentionEntry, ...]
    retain_latest: int
    skipped_record_count: int
    schema_version: str = STUDIO_AUDIT_QUARANTINE_ARCHIVE_RETENTION_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the path-free retention plan for operator APIs."""

        prune_candidate_count = sum(
            entry.disposition == "prune_candidate" for entry in self.entries
        )
        retain_count = sum(entry.disposition == "retain" for entry in self.entries)
        return {
            "archive_count": len(self.entries),
            "entries": [entry.to_public_dict() for entry in self.entries],
            "prune_candidate_count": prune_candidate_count,
            "retain_count": retain_count,
            "retain_latest": self.retain_latest,
            "schema_version": self.schema_version,
            "skipped_record_count": self.skipped_record_count,
        }


@dataclass(frozen=True, slots=True)
class StudioAuditQuarantineArchiveRestoreResult:
    """Path-free result returned after materializing a restore artifact.

    Parameters
    ----------
    archive_id:
        Validated archive identifier restored into job artifacts.
    manifest:
        JSON manifest describing the restore artifacts written by the job.
    summary:
        Path-free aggregate counts for operator review.
    artifact_paths:
        Restore-relative artifact paths written through the Studio job context.
    """

    archive_id: str
    manifest: dict[str, JsonValue]
    summary: dict[str, JsonValue]
    artifact_paths: tuple[str, ...]

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the path-free quarantine archive restore result."""

        return {
            "archive_id": self.archive_id,
            "artifact_paths": list(self.artifact_paths),
            "manifest": self.manifest,
            "schema_version": STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION,
            "summary": self.summary,
        }


def write_studio_audit_quarantine_archive(
    context: StudioJobContext,
    *,
    quarantine_export: Mapping[str, object],
    clock: Callable[[], datetime] | None = None,
) -> StudioAuditQuarantineArchiveResult:
    """Write quarantined audit evidence into a confined Studio job archive.

    Parameters
    ----------
    context:
        Studio job context that owns the archive artifacts and enforces path
        confinement, byte ceilings, and SHA-256 manifests.
    quarantine_export:
        Path-free payload returned by ``JsonlAuditSink.export_quarantine``.
    clock:
        Optional UTC clock for deterministic tests.

    Returns
    -------
    StudioAuditQuarantineArchiveResult
        Path-free manifest and artifact list for the generated archive.

    Raises
    ------
    ValueError
        If the export payload is malformed or not the quarantine export schema.
    """

    now = (clock or _utc_now)().astimezone(UTC).replace(microsecond=0)
    archive_id = f"saqa_{context.job_id}"
    export_payload = _audit_quarantine_export_payload(quarantine_export)
    summary = _audit_quarantine_archive_summary(export_payload)
    written_paths: list[str] = []
    archive_payload: dict[str, JsonValue] = {
        "archive_id": archive_id,
        "archived_at_utc": now.isoformat().replace("+00:00", "Z"),
        "quarantine_export": export_payload,
        "schema_version": STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION,
        "summary": summary,
    }
    archive_entry = _write_json_entry(
        context,
        written_paths,
        "audit_quarantine_archive",
        "evidence/audit-quarantine/archive.json",
        archive_payload,
    )
    manifest: dict[str, JsonValue] = {
        "archive_id": archive_id,
        "artifact_count": 1,
        "created_at_utc": now.isoformat().replace("+00:00", "Z"),
        "entries": [archive_entry],
        "schema_version": STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION,
        "summary": summary,
    }
    manifest_entry = _write_json_entry(
        context,
        written_paths,
        "manifest",
        "evidence/audit-quarantine/manifest.json",
        manifest,
    )
    manifest["manifest_artifact"] = manifest_entry
    return StudioAuditQuarantineArchiveResult(
        archive_id=archive_id,
        manifest=manifest,
        summary=summary,
        artifact_paths=tuple(written_paths),
    )


def validate_studio_audit_quarantine_archive(
    archive_payload: Mapping[str, object],
    *,
    manifest_payload: Mapping[str, object] | None = None,
) -> StudioAuditQuarantineArchiveValidation:
    """Validate one quarantine archive before import or restore handling.

    Parameters
    ----------
    archive_payload:
        Candidate archive JSON object, normally loaded from
        ``evidence/audit-quarantine/archive.json``.
    manifest_payload:
        Optional candidate manifest JSON object, normally loaded from
        ``evidence/audit-quarantine/manifest.json``.

    Returns
    -------
    StudioAuditQuarantineArchiveValidation
        Path-free validation verdict with stable error codes.
    """

    try:
        archive = _audit_quarantine_archive_payload(archive_payload)
        summary = cast(dict[str, JsonValue], archive["summary"])
        archive_id = cast(str, archive["archive_id"])
        recomputed_summary = _audit_quarantine_archive_summary(
            cast(Mapping[str, JsonValue], archive["quarantine_export"])
        )
    except ValueError as exc:
        return _invalid_archive_result(None, str(exc))

    errors: list[str] = []
    if summary != recomputed_summary:
        errors.append("archive_summary_mismatch")
    if manifest_payload is not None:
        errors.extend(
            _manifest_validation_errors(
                manifest_payload,
                archive_id=archive_id,
                expected_summary=recomputed_summary,
            )
        )
    return StudioAuditQuarantineArchiveValidation(
        valid=not errors,
        archive_id=archive_id,
        summary=recomputed_summary,
        errors=tuple(errors),
    )


def build_studio_audit_quarantine_archive_retention_plan(
    records: Sequence[StudioJobRecord],
    *,
    retain_latest: int = 10,
) -> StudioAuditQuarantineArchiveRetentionPlan:
    """Build a non-destructive retention plan for quarantine archive jobs.

    Parameters
    ----------
    records:
        Studio job records from the local job manager.
    retain_latest:
        Number of newest valid quarantine archives that should be retained.

    Returns
    -------
    StudioAuditQuarantineArchiveRetentionPlan
        Path-free inventory marking older valid archives as prune candidates.

    Raises
    ------
    ValueError
        If ``retain_latest`` is not positive.
    """

    if retain_latest <= 0:
        raise ValueError("archive_retention_retain_latest_invalid")
    valid_entries: list[StudioAuditQuarantineArchiveRetentionEntry] = []
    skipped_record_count = 0
    for record in records:
        if not _is_quarantine_archive_record(record):
            continue
        entry = _retention_entry_from_record(record)
        if entry is None:
            skipped_record_count += 1
            continue
        valid_entries.append(entry)
    sorted_entries = sorted(
        valid_entries,
        key=lambda entry: (
            entry.finished_at_utc or "",
            entry.created_at_utc,
            entry.job_id,
        ),
        reverse=True,
    )
    planned_entries = tuple(
        StudioAuditQuarantineArchiveRetentionEntry(
            archive_id=entry.archive_id,
            job_id=entry.job_id,
            created_at_utc=entry.created_at_utc,
            finished_at_utc=entry.finished_at_utc,
            event_count=entry.event_count,
            retained_event_count=entry.retained_event_count,
            artifact_paths=entry.artifact_paths,
            disposition="retain" if index < retain_latest else "prune_candidate",
            summary=entry.summary,
        )
        for index, entry in enumerate(sorted_entries)
    )
    return StudioAuditQuarantineArchiveRetentionPlan(
        entries=planned_entries,
        retain_latest=retain_latest,
        skipped_record_count=skipped_record_count,
    )


def write_studio_audit_quarantine_restore(
    context: StudioJobContext,
    *,
    archive_payload: Mapping[str, object],
    manifest_payload: Mapping[str, object] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> StudioAuditQuarantineArchiveRestoreResult:
    """Materialize validated quarantine archive rows into restore artifacts.

    Parameters
    ----------
    context:
        Studio job context that owns the restore artifacts and enforces path
        confinement, byte ceilings, and SHA-256 manifests.
    archive_payload:
        Candidate archive JSON object from
        ``evidence/audit-quarantine/archive.json``.
    manifest_payload:
        Optional companion manifest JSON object from
        ``evidence/audit-quarantine/manifest.json``.
    clock:
        Optional UTC clock for deterministic tests.

    Returns
    -------
    StudioAuditQuarantineArchiveRestoreResult
        Path-free restore manifest and artifact list.

    Raises
    ------
    ValueError
        If archive validation fails before restore materialization.
    """

    validation = validate_studio_audit_quarantine_archive(
        archive_payload,
        manifest_payload=manifest_payload,
    )
    if not validation.valid or validation.archive_id is None or validation.summary is None:
        raise ValueError("archive_restore_validation_failed")
    archive = _audit_quarantine_archive_payload(archive_payload)
    export_payload = cast(Mapping[str, JsonValue], archive["quarantine_export"])
    event_rows = _restore_event_rows(export_payload)
    now = (clock or _utc_now)().astimezone(UTC).replace(microsecond=0)
    restored_at_utc = now.isoformat().replace("+00:00", "Z")
    written_paths: list[str] = []
    restore_entry = _write_jsonl_entry(
        context,
        written_paths,
        "audit_quarantine_restore_jsonl",
        "evidence/audit-quarantine/restore.jsonl",
        event_rows,
    )
    summary: dict[str, JsonValue] = {
        "archive_id": validation.archive_id,
        "event_count": len(event_rows),
        "quarantine_reason": validation.summary["quarantine_reason"],
        "reason_counts": validation.summary["reason_counts"],
        "restored_at_utc": restored_at_utc,
        "restore_artifact_count": 2,
        "retained_event_count": validation.summary["retained_event_count"],
        "source_schema_version": validation.summary["source_schema_version"],
        "truncated": validation.summary["truncated"],
    }
    manifest: dict[str, JsonValue] = {
        "archive_id": validation.archive_id,
        "artifact_count": 1,
        "created_at_utc": restored_at_utc,
        "entries": [restore_entry],
        "schema_version": STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION,
        "summary": summary,
    }
    manifest_entry = _write_json_entry(
        context,
        written_paths,
        "audit_quarantine_restore_manifest",
        "evidence/audit-quarantine/restore-manifest.json",
        manifest,
    )
    manifest["manifest_artifact"] = manifest_entry
    return StudioAuditQuarantineArchiveRestoreResult(
        archive_id=validation.archive_id,
        manifest=manifest,
        summary=summary,
        artifact_paths=tuple(written_paths),
    )


def _invalid_archive_result(
    archive_id: str | None,
    error_code: str,
) -> StudioAuditQuarantineArchiveValidation:
    return StudioAuditQuarantineArchiveValidation(
        valid=False,
        archive_id=archive_id,
        summary=None,
        errors=(error_code,),
    )


def _is_quarantine_archive_record(record: StudioJobRecord) -> bool:
    return (
        record.kind == STUDIO_AUDIT_QUARANTINE_ARCHIVE_KIND
        and record.owner == STUDIO_AUDIT_QUARANTINE_ARCHIVE_OWNER
    )


def _retention_entry_from_record(
    record: StudioJobRecord,
) -> StudioAuditQuarantineArchiveRetentionEntry | None:
    if record.status != "completed" or record.result is None:
        return None
    result = _json_object(record.result, "archive_job_result_invalid")
    if result.get("schema_version") != STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION:
        return None
    archive_id = result.get("archive_id")
    if not isinstance(archive_id, str) or not archive_id:
        return None
    summary_value = result.get("summary")
    if not isinstance(summary_value, Mapping):
        return None
    summary = _json_object(summary_value, "archive_summary_invalid")
    event_count = summary.get("event_count")
    retained_event_count = summary.get("retained_event_count")
    if not isinstance(event_count, int) or not isinstance(retained_event_count, int):
        return None
    artifact_paths = _artifact_paths(result.get("artifact_paths"))
    if artifact_paths is None:
        return None
    return StudioAuditQuarantineArchiveRetentionEntry(
        archive_id=archive_id,
        job_id=record.job_id,
        created_at_utc=record.created_at_utc,
        finished_at_utc=record.finished_at_utc,
        event_count=event_count,
        retained_event_count=retained_event_count,
        artifact_paths=artifact_paths,
        disposition="retain",
        summary=summary,
    )


def _artifact_paths(value: JsonValue | None) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    paths: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            return None
        paths.append(item)
    return tuple(paths)


def _restore_event_rows(
    export_payload: Mapping[str, JsonValue],
) -> tuple[dict[str, JsonValue], ...]:
    events = cast(list[JsonValue], export_payload["events"])
    rows: list[dict[str, JsonValue]] = []
    for event in events:
        event_object = cast(Mapping[str, object], event)
        rows.append(_json_object(event_object, "restore_event_not_json"))
    return tuple(rows)


def _manifest_validation_errors(
    manifest_payload: Mapping[str, object],
    *,
    archive_id: str,
    expected_summary: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    try:
        manifest = _json_object(
            manifest_payload,
            "manifest_not_json",
        )
    except ValueError as exc:
        return (str(exc),)

    errors: list[str] = []
    if manifest.get("schema_version") != STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION:
        errors.append("manifest_schema_unsupported")
    if manifest.get("archive_id") != archive_id:
        errors.append("manifest_archive_id_mismatch")
    if manifest.get("summary") != dict(expected_summary):
        errors.append("manifest_summary_mismatch")
    if manifest.get("artifact_count") != 1:
        errors.append("manifest_artifact_count_invalid")
    entries = manifest.get("entries")
    if not _has_archive_entry(entries):
        errors.append("manifest_archive_entry_missing")
    return tuple(errors)


def _audit_quarantine_archive_payload(
    payload: Mapping[str, object],
) -> dict[str, JsonValue]:
    result = _json_object(payload, "archive_not_json")
    if result.get("schema_version") != STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION:
        raise ValueError("archive_schema_unsupported")
    archive_id = result.get("archive_id")
    if not isinstance(archive_id, str) or not archive_id:
        raise ValueError("archive_id_invalid")
    archived_at_utc = result.get("archived_at_utc")
    if not isinstance(archived_at_utc, str) or not archived_at_utc.endswith("Z"):
        raise ValueError("archive_timestamp_invalid")
    quarantine_export = result.get("quarantine_export")
    if not isinstance(quarantine_export, Mapping):
        raise ValueError("archive_export_missing")
    _audit_quarantine_export_payload(quarantine_export)
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("archive_summary_missing")
    return result


def _audit_quarantine_archive_summary(
    export_payload: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    events = cast(list[JsonValue], export_payload["events"])
    reason_counts: dict[str, int] = {}
    for event in events:
        event_object = cast(Mapping[str, JsonValue], event)
        reason = cast(str, event_object["quarantine_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "archive_artifact_count": 2,
        "event_count": cast(int, export_payload["event_count"]),
        "quarantine_reason": export_payload["quarantine_reason"],
        "reason_counts": dict(sorted(reason_counts.items())),
        "retained_event_count": cast(int, export_payload["retained_event_count"]),
        "source_schema_version": cast(str, export_payload["schema_version"]),
        "truncated": cast(bool, export_payload["truncated"]),
    }


def _audit_quarantine_export_payload(
    payload: Mapping[str, object],
) -> dict[str, JsonValue]:
    result = _json_object(payload, "export_not_json")
    if result.get("schema_version") != AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION:
        raise ValueError("export_schema_unsupported")
    events = result.get("events")
    if not isinstance(events, list):
        raise ValueError("export_events_invalid")
    event_count = result.get("event_count")
    if not isinstance(event_count, int) or event_count != len(events):
        raise ValueError("export_event_count_mismatch")
    retained_event_count = result.get("retained_event_count")
    if not isinstance(retained_event_count, int) or retained_event_count < event_count:
        raise ValueError("export_retained_count_invalid")
    if not isinstance(result.get("truncated"), bool):
        raise ValueError("export_truncated_invalid")
    quarantine_reason = result.get("quarantine_reason")
    if quarantine_reason is not None and not isinstance(quarantine_reason, str):
        raise ValueError("export_quarantine_reason_invalid")
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("export_event_invalid")
        reason = event.get("quarantine_reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("export_event_invalid")
    return result


def _has_archive_entry(entries: JsonValue | None) -> bool:
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if (
            entry.get("type") == "audit_quarantine_archive"
            and entry.get("bundle_path") == "evidence/audit-quarantine/archive.json"
        ):
            return True
    return False


def _write_json_entry(
    context: StudioJobContext,
    written_paths: list[str],
    entry_type: str,
    relative_path: str,
    payload: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    artifact = context.write_artifact(relative_path, f"{encoded}\n")
    written_paths.append(artifact.relative_path)
    return {
        "bundle_path": artifact.relative_path,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "type": entry_type,
    }


def _write_jsonl_entry(
    context: StudioJobContext,
    written_paths: list[str],
    entry_type: str,
    relative_path: str,
    rows: Sequence[Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    encoded_rows = (
        json.dumps(dict(row), separators=(",", ":"), sort_keys=True) for row in rows
    )
    payload = "\n".join(encoded_rows)
    if payload:
        payload = f"{payload}\n"
    artifact = context.write_artifact(relative_path, payload)
    written_paths.append(artifact.relative_path)
    return {
        "bundle_path": artifact.relative_path,
        "row_count": len(rows),
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "type": entry_type,
    }


def _json_object(payload: Mapping[str, object], error_code: str) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], _json_value(dict(payload), error_code))


def _json_value(value: object, error_code: str) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(error_code)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(error_code)
            result[key] = _json_value(item, error_code)
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item, error_code) for item in value]
    raise ValueError(error_code)


def _utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_VALIDATION_SCHEMA_VERSION",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_RETENTION_SCHEMA_VERSION",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_RESTORE_SCHEMA_VERSION",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_OWNER",
    "STUDIO_AUDIT_QUARANTINE_RESTORE_OWNER",
    "STUDIO_AUDIT_QUARANTINE_ARCHIVE_KIND",
    "JsonScalar",
    "JsonValue",
    "StudioAuditQuarantineArchiveResult",
    "StudioAuditQuarantineArchiveValidation",
    "StudioAuditQuarantineArchiveRetentionEntry",
    "StudioAuditQuarantineArchiveRetentionPlan",
    "StudioAuditQuarantineArchiveRestoreResult",
    "build_studio_audit_quarantine_archive_retention_plan",
    "validate_studio_audit_quarantine_archive",
    "write_studio_audit_quarantine_archive",
    "write_studio_audit_quarantine_restore",
]
