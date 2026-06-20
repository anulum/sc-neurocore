# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training evidence summaries

"""Operator-safe summaries for Studio Training Monitor evidence artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from sc_neurocore.studio.platform.action_evidence import (
    STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION,
)
from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifact,
    StudioJobArtifactPayload,
    StudioJobArtifactUnavailable,
    StudioJobRecord,
)

TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION = "studio.training.evidence-summary.v1"
TRAINING_EVIDENCE_ARTIFACT_PATH = "training/evidence.json"

ArtifactReader: TypeAlias = Callable[[str, str], StudioJobArtifactPayload]


@dataclass(frozen=True, slots=True)
class TrainingEvidenceSummary:
    """Path-free operator summary of one Training Monitor evidence artifact."""

    job_id: str
    status: str
    action_kind: str
    evidence_classification: str
    replay_route: str
    payload_sha256: str
    evidence_artifact: StudioJobArtifact
    result_artifacts: tuple[dict[str, JsonValue], ...]
    schema_version: str = TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON-serializable evidence summary."""

        return {
            "action_kind": self.action_kind,
            "evidence_artifact": self.evidence_artifact.to_public_dict(),
            "evidence_classification": self.evidence_classification,
            "job_id": self.job_id,
            "payload_sha256": self.payload_sha256,
            "replay_route": self.replay_route,
            "result_artifacts": [dict(artifact) for artifact in self.result_artifacts],
            "schema_version": self.schema_version,
            "status": self.status,
        }


def build_training_evidence_summary(
    record: StudioJobRecord,
    artifact_reader: ArtifactReader,
) -> dict[str, object] | None:
    """Return a verified Training Monitor evidence summary when available.

    Parameters
    ----------
    record:
        Path-free Studio job record that may declare the Training Monitor
        evidence artifact.
    artifact_reader:
        Verified artifact reader, normally ``StudioJobManager.read_artifact``.

    Returns
    -------
    dict[str, object] | None
        Public evidence summary for terminal training records, or ``None`` when
        the record does not declare a Training Monitor evidence artifact.
    """

    artifact = _training_evidence_artifact(record)
    if artifact is None:
        return None
    try:
        payload = artifact_reader(record.job_id, TRAINING_EVIDENCE_ARTIFACT_PATH).payload
        evidence = _training_evidence_payload(payload)
    except (
        KeyError,
        StudioJobArtifactUnavailable,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ):
        return _unavailable_summary(record, artifact)
    return TrainingEvidenceSummary(
        job_id=_required_string(evidence, "job_id"),
        status=_required_string(evidence, "status"),
        action_kind=_required_string(evidence, "action_kind"),
        evidence_classification=_required_string(evidence, "evidence_classification"),
        replay_route=_required_string(evidence, "replay_route"),
        payload_sha256=_required_string(evidence, "payload_sha256"),
        evidence_artifact=artifact,
        result_artifacts=_result_artifacts(evidence),
    ).to_public_dict()


def _training_evidence_artifact(record: StudioJobRecord) -> StudioJobArtifact | None:
    """Return the declared Training Monitor evidence artifact, if present."""

    for artifact in record.artifacts:
        if artifact.relative_path == TRAINING_EVIDENCE_ARTIFACT_PATH:
            return artifact
    return None


def _training_evidence_payload(payload: bytes) -> dict[str, object]:
    """Decode and validate a Training Monitor evidence payload."""

    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Studio training evidence payload must be a JSON object.")
    evidence = cast(dict[str, object], decoded)
    if evidence.get("schema_version") != STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Studio training evidence payload has unsupported schema.")
    if evidence.get("action_kind") != "studio.training.run":
        raise ValueError("Studio training evidence payload has unsupported action.")
    if evidence.get("evidence_classification") != "training":
        raise ValueError("Studio training evidence payload has unsupported classification.")
    if evidence.get("job_id") is None:
        raise ValueError("Studio training evidence payload requires a job ID.")
    return evidence


def _required_string(payload: Mapping[str, object], key: str) -> str:
    """Return a required string field from an evidence payload."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Studio training evidence payload requires string field {key!r}.")
    return value


def _result_artifacts(payload: Mapping[str, object]) -> tuple[dict[str, JsonValue], ...]:
    """Return path-free result artifact metadata from an evidence payload."""

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Studio training evidence payload requires artifacts.")
    result: list[dict[str, JsonValue]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Studio training evidence artifact entries must be objects.")
        result.append(cast(dict[str, JsonValue], dict(artifact)))
    return tuple(result)


def _unavailable_summary(
    record: StudioJobRecord,
    artifact: StudioJobArtifact,
) -> dict[str, object]:
    """Return a path-free summary for an unreadable declared evidence artifact."""

    return {
        "evidence_artifact": artifact.to_public_dict(),
        "job_id": record.job_id,
        "schema_version": TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION,
        "status": "unavailable",
    }


__all__ = [
    "TRAINING_EVIDENCE_ARTIFACT_PATH",
    "TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION",
    "TrainingEvidenceSummary",
    "build_training_evidence_summary",
]
