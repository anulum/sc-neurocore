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
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TypeAlias, cast

from sc_neurocore.studio.evidence_classification import (
    StudioEvidenceClassification,
    StudioEvidenceStatus,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)
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
_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")

ArtifactReader: TypeAlias = Callable[[str, str], StudioJobArtifactPayload]


@dataclass(frozen=True, slots=True)
class TrainingEvidenceSummary:
    """Path-free operator summary of one Training Monitor evidence artifact."""

    job_id: str
    status: StudioEvidenceStatus
    action_kind: str
    evidence_classification: StudioEvidenceClassification
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
            "evidence_classification": validate_studio_evidence_classification(
                self.evidence_classification
            ),
            "job_id": self.job_id,
            "payload_sha256": self.payload_sha256,
            "replay_route": self.replay_route,
            "result_artifacts": [dict(artifact) for artifact in self.result_artifacts],
            "schema_version": self.schema_version,
            "status": validate_studio_evidence_status(self.status),
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
        return TrainingEvidenceSummary(
            job_id=_required_string(evidence, "job_id"),
            status=validate_studio_evidence_status(_required_string(evidence, "status")),
            action_kind=_required_string(evidence, "action_kind"),
            evidence_classification=validate_studio_evidence_classification(
                _required_string(evidence, "evidence_classification")
            ),
            replay_route=_required_string(evidence, "replay_route"),
            payload_sha256=_required_string(evidence, "payload_sha256"),
            evidence_artifact=artifact,
            result_artifacts=_result_artifacts(evidence),
        ).to_public_dict()
    except (
        KeyError,
        StudioJobArtifactUnavailable,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ):
        return _unavailable_summary(record, artifact)


def validate_training_evidence_summary(payload: Mapping[str, object]) -> dict[str, JsonValue]:
    """Validate a portable Training Monitor evidence summary.

    Parameters
    ----------
    payload:
        Candidate ``studio.training.evidence-summary.v1`` object, normally
        embedded in a portable Training Monitor checkpoint.

    Returns
    -------
    dict[str, JsonValue]
        JSON-compatible, validated evidence summary.

    Raises
    ------
    ValueError
        If the summary is not verified training evidence, uses an unsupported
        status or classification, or contains malformed artifact metadata.
    """

    summary = _json_object(payload, "Training evidence summary must be JSON.")
    if summary.get("schema_version") != TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION:
        raise ValueError("Training evidence summary schema is unsupported.")
    if _required_json_string(summary, "action_kind") != "studio.training.run":
        raise ValueError("Training evidence summary action is unsupported.")
    if _required_json_string(summary, "evidence_classification") != (
        validate_studio_evidence_classification("training")
    ):
        raise ValueError("Training evidence summary classification is unsupported.")
    validate_studio_evidence_status(_required_json_string(summary, "status"))
    _required_json_string(summary, "job_id")
    _required_json_string(summary, "replay_route")
    payload_sha256 = _required_json_string(summary, "payload_sha256")
    if not _SHA256_HEX_PATTERN.fullmatch(payload_sha256):
        raise ValueError("Training evidence summary payload digest is invalid.")
    _validate_artifact_metadata(
        summary.get("evidence_artifact"),
        expected_path=TRAINING_EVIDENCE_ARTIFACT_PATH,
        field_name="evidence_artifact",
    )
    result_artifacts = summary.get("result_artifacts")
    if not isinstance(result_artifacts, list):
        raise ValueError("Training evidence summary requires result artifacts.")
    for artifact in result_artifacts:
        _validate_artifact_metadata(
            artifact,
            expected_path=None,
            field_name="result_artifacts",
        )
    return summary


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
    if evidence.get("evidence_classification") != validate_studio_evidence_classification(
        "training"
    ):
        raise ValueError("Studio training evidence payload has unsupported classification.")
    status = evidence.get("status")
    if not isinstance(status, str):
        raise ValueError("Studio training evidence payload requires a status.")
    validate_studio_evidence_status(status)
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


def _json_object(payload: Mapping[str, object], error_message: str) -> dict[str, JsonValue]:
    """Return a JSON object after recursively validating portable values."""

    return cast(dict[str, JsonValue], _json_value(dict(payload), error_message))


def _json_value(value: object, error_message: str) -> JsonValue:
    """Return a portable JSON value or raise ``ValueError``."""

    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(error_message)
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item, error_message) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(error_message)
            result[key] = _json_value(item, error_message)
        return result
    raise ValueError(error_message)


def _required_json_string(payload: Mapping[str, JsonValue], field_name: str) -> str:
    """Return a required non-empty string field from a summary payload."""

    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Training evidence summary requires {field_name}.")
    return value


def _validate_artifact_metadata(
    value: object,
    *,
    expected_path: str | None,
    field_name: str,
) -> None:
    """Validate one path-free artifact manifest entry."""

    if not isinstance(value, dict):
        raise ValueError(f"Training evidence summary requires {field_name}.")
    artifact = _json_object(value, f"Training evidence summary {field_name} must be JSON.")
    relative_path = artifact.get("relative_path")
    if not isinstance(relative_path, str) or not _safe_relative_artifact_path(relative_path):
        raise ValueError(f"Training evidence summary {field_name} path is invalid.")
    if expected_path is not None and relative_path != expected_path:
        raise ValueError(f"Training evidence summary {field_name} path is invalid.")
    sha256 = artifact.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX_PATTERN.fullmatch(sha256):
        raise ValueError(f"Training evidence summary {field_name} digest is invalid.")
    size_bytes = artifact.get("size_bytes")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError(f"Training evidence summary {field_name} size is invalid.")


def _safe_relative_artifact_path(relative_path: str) -> bool:
    """Return whether an artifact path is relative and traversal-free."""

    path = PurePosixPath(relative_path)
    return (
        not path.is_absolute()
        and bool(path.parts)
        and all(part not in ("", ".", "..") for part in path.parts)
    )


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
    "validate_training_evidence_summary",
]
