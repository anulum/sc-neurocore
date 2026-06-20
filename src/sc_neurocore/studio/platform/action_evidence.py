# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio action evidence manifests

"""Normalized evidence manifests for Studio worker-backed actions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, TypeAlias

from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.jobs import StudioJobArtifact, StudioJobContext

STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION = "studio.action-evidence.v1"
UTC = timezone.utc

EvidenceClassification: TypeAlias = Literal[
    "local_regression",
    "simulation",
    "compile",
    "synthesis",
    "training",
    "release_benchmark",
]
EvidenceStatus: TypeAlias = Literal["completed", "failed", "cancelled", "timed_out"]


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
    """Write a normalized evidence manifest for a completed worker action."""

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
    encoded = json.dumps(result, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION",
    "EvidenceClassification",
    "EvidenceStatus",
    "StudioActionEvidence",
    "write_studio_action_evidence_manifest",
]
