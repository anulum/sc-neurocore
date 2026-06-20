# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle export

"""Evidence bundle export contracts for SC-NeuroCore Studio."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import TypeAlias, cast

from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifactPayload,
    StudioJobContext,
    StudioJobRecord,
)

STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION = "studio.evidence-bundle.v1"
UTC = timezone.utc

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
StudioArtifactReader: TypeAlias = Callable[[str, str], StudioJobArtifactPayload]


@dataclass(frozen=True, slots=True)
class StudioEvidenceBundleResult:
    """Path-free result returned after writing a Studio evidence bundle.

    Parameters
    ----------
    bundle_id:
        Stable bundle identifier derived from the evidence job ID.
    manifest:
        JSON manifest describing every file written into the bundle.
    artifact_paths:
        Bundle-relative artifact paths written through the Studio job context.
    """

    bundle_id: str
    manifest: dict[str, JsonValue]
    artifact_paths: tuple[str, ...]

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the path-free evidence bundle result."""

        return {
            "artifact_paths": list(self.artifact_paths),
            "bundle_id": self.bundle_id,
            "manifest": self.manifest,
            "schema_version": STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        }


def write_studio_evidence_bundle(
    context: StudioJobContext,
    *,
    project_payload: Mapping[str, object] | None = None,
    job_records: Sequence[StudioJobRecord] = (),
    artifact_reader: StudioArtifactReader | None = None,
    audit_export: Mapping[str, object] | None = None,
    command_replay: Mapping[str, object] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> StudioEvidenceBundleResult:
    """Write a path-confined Studio evidence bundle into a job context.

    Parameters
    ----------
    context:
        Studio job context that owns the bundle files and enforces artifact
        path confinement, byte ceilings, and SHA-256 manifests.
    project_payload:
        Optional saved Studio project payload from ``load_project``.
    job_records:
        Completed or failed Studio job records to preserve with their declared
        artifacts.
    artifact_reader:
        Reader used to fetch verified job artifact bytes. Required when
        ``job_records`` contains artifacts.
    audit_export:
        Optional path-free audit export payload.
    command_replay:
        Optional JSON replay metadata, such as API method, route, request
        digest, and operator note.
    clock:
        Optional UTC clock for deterministic tests.

    Returns
    -------
    StudioEvidenceBundleResult
        Path-free manifest and artifact list for the generated bundle.

    Raises
    ------
    ValueError
        If replay metadata is not JSON-safe or job artifacts are supplied
        without an artifact reader.
    """

    now = (clock or _utc_now)().astimezone(UTC).replace(microsecond=0)
    bundle_id = f"seb_{context.job_id}"
    written_paths: list[str] = []
    entries: list[dict[str, JsonValue]] = []

    if project_payload is not None:
        entries.append(
            _write_json_entry(
                context,
                written_paths,
                "project",
                "evidence/project.json",
                _json_object(project_payload, "Studio project payload must be JSON."),
            )
        )

    if audit_export is not None:
        entries.append(
            _write_json_entry(
                context,
                written_paths,
                "audit_export",
                "evidence/audit-export.json",
                _json_object(audit_export, "Studio audit export must be JSON."),
            )
        )

    if command_replay is not None:
        entries.append(
            _write_json_entry(
                context,
                written_paths,
                "command_replay",
                "evidence/command-replay.json",
                _json_object(command_replay, "Studio command replay must be JSON."),
            )
        )

    if any(record.artifacts for record in job_records) and artifact_reader is None:
        raise ValueError("Studio evidence bundle requires an artifact reader for jobs.")
    reader = cast(StudioArtifactReader, artifact_reader)
    for record in job_records:
        record_path = f"evidence/jobs/{record.job_id}/record.json"
        entries.append(
            _write_json_entry(
                context,
                written_paths,
                "job_record",
                record_path,
                _json_object(
                    record.to_public_dict(),
                    "Studio job record must be JSON.",
                ),
            )
        )
        for artifact in record.artifacts:
            safe_relative_path = _safe_bundle_artifact_path(artifact.relative_path)
            bundle_path = (
                f"evidence/jobs/{record.job_id}/artifacts/{safe_relative_path}"
            )
            artifact_payload = reader(record.job_id, artifact.relative_path)
            written = context.write_artifact(bundle_path, artifact_payload.payload)
            written_paths.append(written.relative_path)
            entries.append(
                {
                    "bundle_path": written.relative_path,
                    "sha256": written.sha256,
                    "size_bytes": written.size_bytes,
                    "source_job_artifact_path": artifact.relative_path,
                    "source_job_id": record.job_id,
                    "type": "job_artifact",
                }
            )

    manifest: dict[str, JsonValue] = {
        "artifact_count": len(entries),
        "bundle_id": bundle_id,
        "created_at_utc": now.isoformat().replace("+00:00", "Z"),
        "entries": cast(list[JsonValue], entries),
        "job_ids": [record.job_id for record in job_records],
        "schema_version": STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    }
    manifest_entry = _write_json_entry(
        context,
        written_paths,
        "manifest",
        "evidence/manifest.json",
        manifest,
    )
    manifest["manifest_artifact"] = manifest_entry
    return StudioEvidenceBundleResult(
        bundle_id=bundle_id,
        manifest=manifest,
        artifact_paths=tuple(written_paths),
    )


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


def _json_object(payload: Mapping[str, object], error_message: str) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], _json_value(dict(payload), error_message))


def _json_value(value: object, error_message: str) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(error_message)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(error_message)
            result[key] = _json_value(item, error_message)
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item, error_message) for item in value]
    raise ValueError(error_message)


def _safe_bundle_artifact_path(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Studio job artifact path is not bundle-safe.")
    return str(path)


def _utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "JsonScalar",
    "JsonValue",
    "StudioArtifactReader",
    "StudioEvidenceBundleResult",
    "write_studio_evidence_bundle",
]
