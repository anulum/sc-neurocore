# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio action evidence manifests

"""Normalised evidence manifests for Studio worker-backed actions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.evidence_classification import (
    STUDIO_EVIDENCE_CLASSIFICATIONS,
    STUDIO_EVIDENCE_TERMINAL_STATUSES,
    StudioEvidenceClassification,
    StudioEvidenceStatus,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)
from sc_neurocore.studio.platform.jobs import StudioJobArtifact, StudioJobContext

STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION = "studio.action-evidence.v1"
UTC = timezone.utc

EvidenceClassification = StudioEvidenceClassification
EvidenceStatus = StudioEvidenceStatus
ACTION_EVIDENCE_CLASSIFICATIONS = STUDIO_EVIDENCE_CLASSIFICATIONS
ACTION_EVIDENCE_STATUSES = STUDIO_EVIDENCE_TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class StudioActionEvidence:
    """Path-free manifest describing one Studio worker-backed action."""

    payload: dict[str, JsonValue]
    artifact: StudioJobArtifact

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the path-free evidence payload."""

        return dict(self.payload)


def write_studio_action_evidence_manifest(
    context: StudioJobContext,
    *,
    action_kind: str,
    result: Mapping[str, object],
    result_artifact: StudioJobArtifact,
    evidence_artifact_path: str,
    evidence_classification: EvidenceClassification,
    replay_route: str,
    status: EvidenceStatus = "completed",
    request_id: str | None = None,
    principal_id: str | None = None,
    error_message: str | None = None,
) -> StudioActionEvidence:
    """Write a normalised evidence manifest for a worker-backed action.

    Parameters
    ----------
    context:
        Job sandbox used to write the path-confined evidence artifact.
    action_kind:
        Stable Studio action identifier, such as ``studio.compile``.
    result:
        Portable JSON result payload whose canonical SHA-256 digest is recorded.
    result_artifact:
        Manifest entry for the result artifact produced by the same job.
    evidence_artifact_path:
        Relative path for the evidence manifest artifact.
    evidence_classification:
        Controlled evidence class used by bundle and operator views.
    replay_route:
        HTTP method and route template used to reproduce the action.
    status:
        Terminal action status.
    request_id:
        Optional request correlation identifier.
    principal_id:
        Optional authenticated principal identifier.
    error_message:
        Optional bounded terminal error message.

    Returns
    -------
    StudioActionEvidence
        Path-free evidence payload plus its artifact manifest entry.

    Raises
    ------
    ValueError
        If any controlled field is invalid or ``result`` is not portable JSON.
    """

    _validate_manifest_fields(
        action_kind=action_kind,
        evidence_classification=evidence_classification,
        replay_route=replay_route,
        status=status,
    )
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    result_artifact_payload: dict[str, JsonValue] = {
        "relative_path": result_artifact.relative_path,
        "sha256": result_artifact.sha256,
        "size_bytes": result_artifact.size_bytes,
    }
    payload: dict[str, JsonValue] = {
        "action_kind": action_kind,
        "artifacts": [result_artifact_payload],
        "evidence_classification": evidence_classification,
        "generated_at_utc": generated_at,
        "job_id": context.job_id,
        "payload_sha256": _payload_sha256(result),
        "principal_id": principal_id,
        "replay_route": replay_route,
        "request_id": request_id,
        "schema_version": STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION,
        "status": status,
    }
    if error_message is not None:
        payload["error_message"] = error_message
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    artifact = context.write_artifact(evidence_artifact_path, f"{encoded}\n")
    return StudioActionEvidence(payload=payload, artifact=artifact)


def _payload_sha256(result: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(result),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest_fields(
    *,
    action_kind: str,
    evidence_classification: str,
    replay_route: str,
    status: str,
) -> None:
    if not _is_dotted_action_kind(action_kind):
        raise ValueError("Studio action evidence action kind is invalid.")
    try:
        validate_studio_evidence_classification(evidence_classification)
    except ValueError as exc:
        raise ValueError("Studio action evidence classification is invalid.") from exc
    try:
        validate_studio_evidence_status(status)
    except ValueError as exc:
        raise ValueError("Studio action evidence status is invalid.") from exc
    if not _is_replay_route(replay_route):
        raise ValueError("Studio action evidence replay route is invalid.")


def _is_dotted_action_kind(value: str) -> bool:
    parts = value.split(".")
    return (
        len(parts) >= 2
        and all(part and part.replace("_", "").isalnum() for part in parts)
        and value == value.strip()
    )


def _is_replay_route(value: str) -> bool:
    method, separator, route = value.partition(" ")
    return (
        separator == " "
        and method in {"DELETE", "GET", "PATCH", "POST", "PUT"}
        and route.startswith("/")
        and route == route.strip()
        and " " not in route
    )


__all__ = [
    "STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION",
    "ACTION_EVIDENCE_CLASSIFICATIONS",
    "ACTION_EVIDENCE_STATUSES",
    "EvidenceClassification",
    "EvidenceStatus",
    "StudioActionEvidence",
    "write_studio_action_evidence_manifest",
]
