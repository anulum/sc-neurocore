# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle export

"""Evidence bundle export contracts for SC-NeuroCore Studio.

On-disk bundle assembly lives here. Payload normalisation for each evidence
kind lives in ``evidence_bundle_payloads``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypeAlias, cast

from sc_neurocore.studio.evidence_classification import (
    STUDIO_EVIDENCE_CLASSIFICATIONS,
    STUDIO_EVIDENCE_TERMINAL_STATUSES,
)
from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifactPayload,
    StudioJobContext,
    StudioJobRecord,
)
from sc_neurocore.studio.platform.evidence_bundle_payloads import (
    JsonScalar,
    JsonValue,
    _action_evidence_payload,
    _analysis_result_payload,
    _default_flow_attestation_payload,
    _default_flow_fingerprints,
    _default_flow_key,
    _default_flow_run_payload,
    _is_action_evidence_artifact,
    _json_object,
    _model_scan_payload,
    _project_workspace_payload,
    _safe_bundle_artifact_path,
    _simulation_result_payload,
    _weight_restore_attach_payload,
    _weight_restore_payload,
)

STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION = "studio.evidence-bundle.v1"
UTC = timezone.utc
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
    summary:
        Path-free aggregate counts for operator review and UI rendering.
    artifact_paths:
        Bundle-relative artifact paths written through the Studio job context.
    """

    bundle_id: str
    manifest: dict[str, JsonValue]
    summary: dict[str, JsonValue]
    artifact_paths: tuple[str, ...]

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the path-free evidence bundle result."""

        return {
            "artifact_paths": list(self.artifact_paths),
            "bundle_id": self.bundle_id,
            "manifest": self.manifest,
            "schema_version": STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            "summary": self.summary,
        }


def write_studio_evidence_bundle(
    context: StudioJobContext,
    *,
    project_payload: Mapping[str, object] | None = None,
    simulation_payloads: Sequence[Mapping[str, object]] = (),
    analysis_payloads: Sequence[Mapping[str, object]] = (),
    model_scan_payloads: Sequence[Mapping[str, object]] = (),
    weight_restore_payloads: Sequence[Mapping[str, object]] = (),
    weight_restore_attach_payloads: Sequence[Mapping[str, object]] = (),
    default_flow_runs: Sequence[Mapping[str, object]] = (),
    default_flow_attestations: Sequence[Mapping[str, object]] = (),
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
    simulation_payloads:
        Optional Studio simulation responses carrying ``studio.simulation-run.v1``
        run metadata.
    analysis_payloads:
        Optional Studio analysis responses carrying ``studio.analysis-result.v1``
        analysis metadata.
    model_scan_payloads:
        Optional Studio model-scan responses carrying ``studio.model-scan.v1``
        scan metadata classified as analysis evidence.
    weight_restore_payloads:
        Optional Studio training weight-restore responses carrying
        ``studio.training.weight-restore.v1`` materialization evidence classified
        as training evidence.
    weight_restore_attach_payloads:
        Optional Studio training weight-restore attach responses carrying
        ``studio.training.weight-restore-attach.v1`` evidence classified as
        training evidence.
    default_flow_runs:
        Optional guided default-flow run responses carrying reproducibility
        fingerprints.
    default_flow_attestations:
        Optional guided default-flow attestations for the supplied run
        responses.
    job_records:
        Completed or failed Studio job records to preserve with their declared
        artifacts. Artifacts ending in ``evidence.json`` must carry the
        ``studio.action-evidence.v1`` contract and are classified as
        first-class action evidence in the bundle manifest.
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
        payload = _project_workspace_payload(project_payload)
        entries.append(
            _write_classified_json_entry(
                context,
                written_paths,
                "project",
                "evidence/project.json",
                payload,
                evidence_classification="project_workspace",
            )
        )

    for index, simulation_payload in enumerate(simulation_payloads):
        payload = _simulation_result_payload(simulation_payload)
        entries.append(
            _write_classified_json_entry(
                context,
                written_paths,
                "simulation_result",
                f"evidence/simulations/{index:03d}.json",
                payload,
                evidence_classification="simulation",
            )
        )

    for index, analysis_payload in enumerate(analysis_payloads):
        payload = _analysis_result_payload(analysis_payload)
        entries.append(
            _write_classified_json_entry(
                context,
                written_paths,
                "analysis_result",
                f"evidence/analyses/{index:03d}.json",
                payload,
                evidence_classification="analysis",
            )
        )

    for index, model_scan_payload in enumerate(model_scan_payloads):
        payload = _model_scan_payload(model_scan_payload)
        entries.append(
            _write_classified_json_entry(
                context,
                written_paths,
                "model_scan_result",
                f"evidence/model-scans/{index:03d}.json",
                payload,
                evidence_classification="analysis",
            )
        )

    for index, weight_restore_payload in enumerate(weight_restore_payloads):
        payload = _weight_restore_payload(weight_restore_payload)
        entries.append(
            _write_classified_json_entry(
                context,
                written_paths,
                "training_weight_restore_result",
                f"evidence/training-weight-restores/{index:03d}.json",
                payload,
                evidence_classification="training",
            )
        )

    for index, weight_restore_attach_payload in enumerate(weight_restore_attach_payloads):
        payload = _weight_restore_attach_payload(weight_restore_attach_payload)
        entries.append(
            _write_classified_json_entry(
                context,
                written_paths,
                "training_weight_restore_attach_result",
                f"evidence/training-weight-restore-attaches/{index:03d}.json",
                payload,
                evidence_classification="training",
            )
        )

    default_flow_run_fingerprints: dict[tuple[str, str], tuple[str, str]] = {}
    for index, default_flow_run in enumerate(default_flow_runs):
        payload = _default_flow_run_payload(default_flow_run)
        default_flow_run_fingerprints[_default_flow_key(payload)] = _default_flow_fingerprints(
            payload
        )
        entries.append(
            _write_classified_json_entry(
                context,
                written_paths,
                "default_flow_run",
                f"evidence/default-flows/runs/{index:03d}.json",
                payload,
                evidence_classification="default_flow",
            )
        )

    for index, default_flow_attestation in enumerate(default_flow_attestations):
        payload = _default_flow_attestation_payload(
            default_flow_attestation,
            run_fingerprints=default_flow_run_fingerprints,
        )
        entries.append(
            _write_classified_json_entry(
                context,
                written_paths,
                "default_flow_attestation",
                f"evidence/default-flows/attestations/{index:03d}.json",
                payload,
                evidence_classification="default_flow",
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
            bundle_path = f"evidence/jobs/{record.job_id}/artifacts/{safe_relative_path}"
            artifact_payload = reader(record.job_id, artifact.relative_path)
            written = context.write_artifact(bundle_path, artifact_payload.payload)
            written_paths.append(written.relative_path)
            entries.append(
                _job_artifact_entry(
                    record=record,
                    source_path=artifact.relative_path,
                    bundle_path=written.relative_path,
                    sha256=written.sha256,
                    size_bytes=written.size_bytes,
                    payload=artifact_payload.payload,
                )
            )

    summary = _bundle_summary(
        entries,
        artifact_path_count=len(written_paths) + 1,
        job_records=job_records,
    )
    manifest: dict[str, JsonValue] = {
        "artifact_count": len(entries),
        "bundle_id": bundle_id,
        "created_at_utc": now.isoformat().replace("+00:00", "Z"),
        "entries": cast(list[JsonValue], entries),
        "job_ids": [record.job_id for record in job_records],
        "schema_version": STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "summary": summary,
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
        summary=summary,
        artifact_paths=tuple(written_paths),
    )


def _bundle_summary(
    entries: Sequence[Mapping[str, JsonValue]],
    *,
    artifact_path_count: int,
    job_records: Sequence[StudioJobRecord],
) -> dict[str, JsonValue]:
    entry_type_counts: dict[str, int] = {}
    evidence_classification_counts: dict[str, int] = {}
    source_job_kind_counts: dict[str, int] = {}
    source_job_owner_counts: dict[str, int] = {}

    for entry in entries:
        entry_type = entry.get("type")
        if isinstance(entry_type, str):
            entry_type_counts[entry_type] = entry_type_counts.get(entry_type, 0) + 1
        classification = entry.get("evidence_classification")
        if isinstance(classification, str):
            evidence_classification_counts[classification] = (
                evidence_classification_counts.get(classification, 0) + 1
            )

    for record in job_records:
        source_job_kind_counts[record.kind] = source_job_kind_counts.get(record.kind, 0) + 1
        source_job_owner_counts[record.owner] = source_job_owner_counts.get(record.owner, 0) + 1

    return {
        "artifact_path_count": artifact_path_count,
        "entry_count": len(entries),
        "entry_type_counts": dict(sorted(entry_type_counts.items())),
        "evidence_classification_counts": dict(sorted(evidence_classification_counts.items())),
        "known_evidence_classifications": cast(
            list[JsonValue],
            sorted(STUDIO_EVIDENCE_CLASSIFICATIONS),
        ),
        "known_terminal_statuses": cast(
            list[JsonValue],
            sorted(STUDIO_EVIDENCE_TERMINAL_STATUSES),
        ),
        "source_job_count": len(job_records),
        "source_job_kind_counts": dict(sorted(source_job_kind_counts.items())),
        "source_job_owner_counts": dict(sorted(source_job_owner_counts.items())),
    }


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


def _write_classified_json_entry(
    context: StudioJobContext,
    written_paths: list[str],
    entry_type: str,
    relative_path: str,
    payload: Mapping[str, JsonValue],
    *,
    evidence_classification: str,
) -> dict[str, JsonValue]:
    entry = _write_json_entry(context, written_paths, entry_type, relative_path, payload)
    entry["evidence_classification"] = evidence_classification
    return entry


def _job_artifact_entry(
    *,
    record: StudioJobRecord,
    source_path: str,
    bundle_path: str,
    sha256: str,
    size_bytes: int,
    payload: bytes,
) -> dict[str, JsonValue]:
    entry: dict[str, JsonValue] = {
        "bundle_path": bundle_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "source_job_artifact_path": source_path,
        "source_job_id": record.job_id,
        "type": "job_artifact",
    }
    if _is_action_evidence_artifact(source_path):
        action_evidence = _action_evidence_payload(payload, source_job_id=record.job_id)
        entry.update(
            {
                "action_kind": action_evidence["action_kind"],
                "action_status": action_evidence["status"],
                "evidence_classification": action_evidence["evidence_classification"],
                "payload_sha256": action_evidence["payload_sha256"],
                "type": "action_evidence",
            }
        )
    return entry


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
