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
from sc_neurocore.studio.analysis_manifest import STUDIO_ANALYSIS_RESULT_SCHEMA_VERSION
from sc_neurocore.studio.simulation_manifest import STUDIO_SIMULATION_RUN_SCHEMA_VERSION

STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION = "studio.evidence-bundle.v1"
STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION = "studio.action-evidence.v1"
STUDIO_DEFAULT_FLOW_RUN_SCHEMA_VERSION = "sc-neurocore.studio.default-flow-run.v1"
STUDIO_DEFAULT_FLOW_ATTESTATION_SCHEMA_VERSION = (
    "sc-neurocore.studio.default-flow-attestation.v1"
)
UTC = timezone.utc

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
StudioArtifactReader: TypeAlias = Callable[[str, str], StudioJobArtifactPayload]
ACTION_EVIDENCE_CLASSIFICATIONS = frozenset(
    {
        "compile",
        "local_regression",
        "release_benchmark",
        "simulation",
        "synthesis",
        "training",
    }
)
ACTION_EVIDENCE_STATUSES = frozenset({"cancelled", "completed", "failed", "timed_out"})


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
    simulation_payloads: Sequence[Mapping[str, object]] = (),
    analysis_payloads: Sequence[Mapping[str, object]] = (),
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
        entries.append(
            _write_json_entry(
                context,
                written_paths,
                "project",
                "evidence/project.json",
                _json_object(project_payload, "Studio project payload must be JSON."),
            )
        )

    for index, simulation_payload in enumerate(simulation_payloads):
        entries.append(
            _write_json_entry(
                context,
                written_paths,
                "simulation_result",
                f"evidence/simulations/{index:03d}.json",
                _simulation_result_payload(simulation_payload),
            )
        )

    for index, analysis_payload in enumerate(analysis_payloads):
        entries.append(
            _write_json_entry(
                context,
                written_paths,
                "analysis_result",
                f"evidence/analyses/{index:03d}.json",
                _analysis_result_payload(analysis_payload),
            )
        )

    default_flow_run_fingerprints: dict[tuple[str, str], tuple[str, str]] = {}
    for index, default_flow_run in enumerate(default_flow_runs):
        payload = _default_flow_run_payload(default_flow_run)
        default_flow_run_fingerprints[_default_flow_key(payload)] = _default_flow_fingerprints(
            payload
        )
        entries.append(
            _write_json_entry(
                context,
                written_paths,
                "default_flow_run",
                f"evidence/default-flows/runs/{index:03d}.json",
                payload,
            )
        )

    for index, default_flow_attestation in enumerate(default_flow_attestations):
        payload = _default_flow_attestation_payload(
            default_flow_attestation,
            run_fingerprints=default_flow_run_fingerprints,
        )
        entries.append(
            _write_json_entry(
                context,
                written_paths,
                "default_flow_attestation",
                f"evidence/default-flows/attestations/{index:03d}.json",
                payload,
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
                _job_artifact_entry(
                    record=record,
                    source_path=artifact.relative_path,
                    bundle_path=written.relative_path,
                    sha256=written.sha256,
                    size_bytes=written.size_bytes,
                    payload=artifact_payload.payload,
                )
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


def _simulation_result_payload(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    result = _json_object(payload, "Studio simulation payload must be JSON.")
    metadata = result.get("run_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Studio simulation payload requires run metadata.")
    schema_version = metadata.get("schema_version")
    if schema_version != STUDIO_SIMULATION_RUN_SCHEMA_VERSION:
        raise ValueError("Studio simulation payload has unsupported run metadata.")
    evidence_classification = metadata.get("evidence_classification")
    if evidence_classification != "simulation":
        raise ValueError("Studio simulation payload must be classified as simulation evidence.")
    return result


def _analysis_result_payload(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    result = _json_object(payload, "Studio analysis payload must be JSON.")
    metadata = result.get("analysis_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Studio analysis payload requires analysis metadata.")
    schema_version = metadata.get("schema_version")
    if schema_version != STUDIO_ANALYSIS_RESULT_SCHEMA_VERSION:
        raise ValueError("Studio analysis payload has unsupported analysis metadata.")
    evidence_classification = metadata.get("evidence_classification")
    if evidence_classification != "analysis":
        raise ValueError("Studio analysis payload must be classified as analysis evidence.")
    return result


def _default_flow_run_payload(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    result = _json_object(payload, "Studio default-flow run payload must be JSON.")
    if result.get("schema_version") != STUDIO_DEFAULT_FLOW_RUN_SCHEMA_VERSION:
        raise ValueError("Studio default-flow run payload has unsupported schema.")
    preset_id = result.get("preset_id")
    flow_id = result.get("flow_id")
    if not isinstance(preset_id, str) or not preset_id:
        raise ValueError("Studio default-flow run payload requires a preset ID.")
    if not isinstance(flow_id, str) or not flow_id:
        raise ValueError("Studio default-flow run payload requires a flow ID.")
    action_order = result.get("action_order")
    if not isinstance(action_order, list) or not all(
        isinstance(action_id, str) and action_id for action_id in action_order
    ):
        raise ValueError("Studio default-flow run payload requires action order.")
    executed_count = result.get("executed_count")
    if not isinstance(executed_count, int) or executed_count < 0:
        raise ValueError("Studio default-flow run payload requires executed count.")
    reproducibility = result.get("reproducibility_manifest")
    if not isinstance(reproducibility, Mapping):
        raise ValueError("Studio default-flow run payload requires reproducibility metadata.")
    if reproducibility.get("hash_algorithm") != "sha256":
        raise ValueError("Studio default-flow run payload has unsupported hash algorithm.")
    inputs_fingerprint = reproducibility.get("inputs_fingerprint_sha256")
    run_fingerprint = reproducibility.get("run_fingerprint_sha256")
    if not _is_sha256_hex(inputs_fingerprint) or not _is_sha256_hex(run_fingerprint):
        raise ValueError("Studio default-flow run payload requires SHA-256 fingerprints.")
    return result


def _default_flow_attestation_payload(
    payload: Mapping[str, object],
    *,
    run_fingerprints: Mapping[tuple[str, str], tuple[str, str]],
) -> dict[str, JsonValue]:
    result = _json_object(payload, "Studio default-flow attestation payload must be JSON.")
    if result.get("schema_version") != STUDIO_DEFAULT_FLOW_ATTESTATION_SCHEMA_VERSION:
        raise ValueError("Studio default-flow attestation payload has unsupported schema.")
    preset_id = result.get("preset_id")
    flow_id = result.get("flow_id")
    if not isinstance(preset_id, str) or not preset_id:
        raise ValueError("Studio default-flow attestation payload requires a preset ID.")
    if not isinstance(flow_id, str) or not flow_id:
        raise ValueError("Studio default-flow attestation payload requires a flow ID.")
    for key in (
        "attestation_fingerprint_sha256",
        "inputs_fingerprint_sha256",
        "plan_fingerprint_sha256",
        "run_fingerprint_sha256",
    ):
        if not _is_sha256_hex(result.get(key)):
            raise ValueError("Studio default-flow attestation payload requires SHA-256 fingerprints.")
    expected = run_fingerprints.get((preset_id, flow_id))
    observed = (
        cast(str, result["inputs_fingerprint_sha256"]),
        cast(str, result["run_fingerprint_sha256"]),
    )
    if expected is not None and observed != expected:
        raise ValueError("Studio default-flow attestation payload does not match supplied run.")
    return result


def _default_flow_key(payload: Mapping[str, JsonValue]) -> tuple[str, str]:
    return (cast(str, payload["preset_id"]), cast(str, payload["flow_id"]))


def _default_flow_fingerprints(payload: Mapping[str, JsonValue]) -> tuple[str, str]:
    reproducibility = cast(Mapping[str, JsonValue], payload["reproducibility_manifest"])
    return (
        cast(str, reproducibility["inputs_fingerprint_sha256"]),
        cast(str, reproducibility["run_fingerprint_sha256"]),
    )


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


def _is_action_evidence_artifact(relative_path: str) -> bool:
    name = PurePosixPath(relative_path).name
    return name == "evidence.json" or name.endswith("-evidence.json")


def _action_evidence_payload(payload: bytes, *, source_job_id: str) -> dict[str, JsonValue]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Studio action evidence artifact must be JSON.") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Studio action evidence artifact must be a JSON object.")
    result = _json_object(decoded, "Studio action evidence artifact must be JSON.")
    if result.get("schema_version") != STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Studio action evidence artifact has unsupported schema.")
    if result.get("job_id") != source_job_id:
        raise ValueError("Studio action evidence artifact job ID does not match source job.")
    action_kind = result.get("action_kind")
    if not isinstance(action_kind, str) or not action_kind:
        raise ValueError("Studio action evidence artifact requires an action kind.")
    evidence_classification = result.get("evidence_classification")
    if evidence_classification not in ACTION_EVIDENCE_CLASSIFICATIONS:
        raise ValueError("Studio action evidence artifact has unsupported classification.")
    status = result.get("status")
    if status not in ACTION_EVIDENCE_STATUSES:
        raise ValueError("Studio action evidence artifact has unsupported status.")
    payload_sha256 = result.get("payload_sha256")
    if not _is_sha256_hex(payload_sha256):
        raise ValueError("Studio action evidence artifact requires a payload SHA-256.")
    if not isinstance(result.get("replay_route"), str):
        raise ValueError("Studio action evidence artifact requires a replay route.")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Studio action evidence artifact requires artifact metadata.")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("Studio action evidence artifact has invalid artifact metadata.")
        relative_path = artifact.get("relative_path")
        sha256 = artifact.get("sha256")
        size_bytes = artifact.get("size_bytes")
        if not isinstance(relative_path, str):
            raise ValueError("Studio action evidence artifact has invalid artifact metadata.")
        _safe_bundle_artifact_path(relative_path)
        if not _is_sha256_hex(sha256) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("Studio action evidence artifact has invalid artifact metadata.")
    return result


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
