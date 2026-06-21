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

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.policy import AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION

STUDIO_AUDIT_QUARANTINE_ARCHIVE_SCHEMA_VERSION = "studio.audit-quarantine-archive.v1"
STUDIO_AUDIT_QUARANTINE_ARCHIVE_VALIDATION_SCHEMA_VERSION = (
    "studio.audit-quarantine-archive.validation.v1"
)
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
    "JsonScalar",
    "JsonValue",
    "StudioAuditQuarantineArchiveResult",
    "StudioAuditQuarantineArchiveValidation",
    "validate_studio_audit_quarantine_archive",
    "write_studio_audit_quarantine_archive",
]
