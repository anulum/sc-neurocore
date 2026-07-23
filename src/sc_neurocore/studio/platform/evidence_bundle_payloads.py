# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle payload normalisers

"""Normalise operator-supplied evidence payloads for Studio evidence bundles.

Payload shaping is independent of on-disk bundle writing so each evidence kind
can be validated without constructing a full archive.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import TypeAlias, cast

from sc_neurocore.studio.analysis_manifest import STUDIO_ANALYSIS_RESULT_SCHEMA_VERSION
from sc_neurocore.studio.evidence_classification import (
    STUDIO_EVIDENCE_CLASSIFICATIONS,
    STUDIO_EVIDENCE_TERMINAL_STATUSES,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)
from sc_neurocore.studio.model_scan import STUDIO_MODEL_SCAN_SCHEMA_VERSION
from sc_neurocore.studio.project_manifest import build_project_save_manifest
from sc_neurocore.studio.simulation_manifest import STUDIO_SIMULATION_RUN_SCHEMA_VERSION

STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION = "studio.action-evidence.v1"
STUDIO_DEFAULT_FLOW_RUN_SCHEMA_VERSION = "sc-neurocore.studio.default-flow-run.v1"
STUDIO_DEFAULT_FLOW_ATTESTATION_SCHEMA_VERSION = "sc-neurocore.studio.default-flow-attestation.v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
ACTION_EVIDENCE_CLASSIFICATIONS = STUDIO_EVIDENCE_CLASSIFICATIONS
ACTION_EVIDENCE_STATUSES = STUDIO_EVIDENCE_TERMINAL_STATUSES


def _json_object(payload: Mapping[str, object], error_message: str) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], _json_value(dict(payload), error_message))


def _project_workspace_payload(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    """Validate a saved Studio project payload before bundle export."""

    result = _json_object(payload, "Studio project payload must be JSON.")
    name = result.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Studio project payload requires a project name.")
    saved_at = result.get("saved_at")
    if isinstance(saved_at, bool) or not isinstance(saved_at, int | float):
        raise ValueError("Studio project payload requires a saved timestamp.")
    version = result.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("Studio project payload requires a project version.")
    state = result.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Studio project payload requires a state object.")
    build_project_save_manifest(
        name=name,
        saved_at=float(saved_at),
        version=version,
        state=state,
        project_payload=result,
    )
    return result


def _simulation_result_payload(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    result = _json_object(payload, "Studio simulation payload must be JSON.")
    metadata = result.get("run_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Studio simulation payload requires run metadata.")
    schema_version = metadata.get("schema_version")
    if schema_version != STUDIO_SIMULATION_RUN_SCHEMA_VERSION:
        raise ValueError("Studio simulation payload has unsupported run metadata.")
    evidence_classification = metadata.get("evidence_classification")
    if evidence_classification != validate_studio_evidence_classification("simulation"):
        raise ValueError("Studio simulation payload must be classified as simulation evidence.")
    if metadata.get("status") != validate_studio_evidence_status("completed"):
        raise ValueError("Studio simulation payload must have completed evidence status.")
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
    if evidence_classification != validate_studio_evidence_classification("analysis"):
        raise ValueError("Studio analysis payload must be classified as analysis evidence.")
    if metadata.get("status") != validate_studio_evidence_status("completed"):
        raise ValueError("Studio analysis payload must have completed evidence status.")
    return result


def _model_scan_payload(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    result = _json_object(payload, "Studio model-scan payload must be JSON.")
    if result.get("schema_version") != STUDIO_MODEL_SCAN_SCHEMA_VERSION:
        raise ValueError("Studio model-scan payload has unsupported scan metadata.")
    metadata = result.get("scan_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Studio model-scan payload requires scan metadata.")
    if metadata.get("schema_version") != STUDIO_MODEL_SCAN_SCHEMA_VERSION:
        raise ValueError("Studio model-scan payload has unsupported scan metadata.")
    if metadata.get("evidence_classification") != validate_studio_evidence_classification(
        "analysis"
    ):
        raise ValueError("Studio model-scan payload must be classified as analysis evidence.")
    if metadata.get("status") != validate_studio_evidence_status("completed"):
        raise ValueError("Studio model-scan payload must have completed evidence status.")
    return result


def _weight_restore_payload(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    # Lazy import: ``training_weights`` imports ``JsonValue`` from this module, so
    # a module-level import would create a circular import.
    from sc_neurocore.studio.platform.training_weights import (
        validate_training_weight_restore_evidence,
    )

    result = _json_object(payload, "Studio weight-restore payload must be JSON.")
    validate_training_weight_restore_evidence(result)
    if result.get("evidence_classification") != validate_studio_evidence_classification("training"):
        raise ValueError("Studio weight-restore payload must be classified as training evidence.")
    if result.get("status") != validate_studio_evidence_status("completed"):
        raise ValueError("Studio weight-restore payload must have completed evidence status.")
    return result


def _weight_restore_attach_payload(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    # Lazy import: ``training_weights`` imports ``JsonValue`` from this module, so
    # a module-level import would create a circular import.
    from sc_neurocore.studio.platform.training_weights import (
        validate_training_weight_restore_attach_evidence,
    )

    result = _json_object(payload, "Studio weight-restore attach payload must be JSON.")
    validate_training_weight_restore_attach_evidence(result)
    if result.get("evidence_classification") != validate_studio_evidence_classification("training"):
        raise ValueError(
            "Studio weight-restore attach payload must be classified as training evidence."
        )
    if result.get("status") != validate_studio_evidence_status("completed"):
        raise ValueError(
            "Studio weight-restore attach payload must have completed evidence status."
        )
    return result


def _default_flow_run_payload(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    result = _json_object(payload, "Studio default-flow run payload must be JSON.")
    if result.get("schema_version") != STUDIO_DEFAULT_FLOW_RUN_SCHEMA_VERSION:
        raise ValueError("Studio default-flow run payload has unsupported schema.")
    if result.get("evidence_classification") != validate_studio_evidence_classification(
        "default_flow"
    ):
        raise ValueError(
            "Studio default-flow run payload must be classified as default-flow evidence."
        )
    if result.get("status") != validate_studio_evidence_status("completed"):
        raise ValueError("Studio default-flow run payload must have completed evidence status.")
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
    if result.get("evidence_classification") != validate_studio_evidence_classification(
        "default_flow"
    ):
        raise ValueError(
            "Studio default-flow attestation payload must be classified as default-flow evidence."
        )
    if result.get("status") != validate_studio_evidence_status("completed"):
        raise ValueError(
            "Studio default-flow attestation payload must have completed evidence status."
        )
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
            raise ValueError(
                "Studio default-flow attestation payload requires SHA-256 fingerprints."
            )
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
    if not isinstance(evidence_classification, str):
        raise ValueError("Studio action evidence artifact has unsupported classification.")
    try:
        validate_studio_evidence_classification(evidence_classification)
    except ValueError as exc:
        raise ValueError("Studio action evidence artifact has unsupported classification.") from exc
    status = result.get("status")
    if not isinstance(status, str):
        raise ValueError("Studio action evidence artifact has unsupported status.")
    try:
        validate_studio_evidence_status(status)
    except ValueError as exc:
        raise ValueError("Studio action evidence artifact has unsupported status.") from exc
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


def _safe_bundle_artifact_path(relative_path: str) -> str:
    """Return a confined relative path or raise when the path is unsafe."""
    path = PurePosixPath(relative_path)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Studio job artifact path is not bundle-safe.")
    return str(path)


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
