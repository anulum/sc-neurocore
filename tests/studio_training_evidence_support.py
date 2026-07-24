# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_studio_training_evidence.py

from __future__ import annotations


"""Tests for Training Monitor operator evidence summaries."""


import json


from collections.abc import Callable, Mapping


from pathlib import Path


from typing import cast


import pytest


from sc_neurocore.studio.platform.action_evidence import (
    STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION,
    write_studio_action_evidence_manifest,
)


from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifact,
    StudioJobArtifactPayload,
    StudioJobArtifactUnavailable,
    StudioJobContext,
    StudioJobManager,
    StudioJobRecord,
)


from sc_neurocore.studio.platform.training_evidence import (
    TRAINING_EVIDENCE_ARTIFACT_PATH,
    TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION,
    build_training_evidence_summary,
    validate_training_evidence_summary,
)


from sc_neurocore.studio.training import get_training_status


def training_evidence_task(context: StudioJobContext) -> dict[str, object]:
    """Write a terminal Training Monitor evidence artifact for tests."""

    status_payload: dict[str, object] = {
        "error": None,
        "final_metrics": {
            "train_accuracy": 0.9,
            "train_loss": 0.1,
            "val_accuracy": 0.8,
            "val_loss": 0.2,
        },
        "job_id": context.job_id,
        "status": "completed",
    }
    status_artifact = context.write_artifact(
        "training/status.json",
        json.dumps(status_payload, sort_keys=True),
    )
    write_studio_action_evidence_manifest(
        context,
        action_kind="studio.training.run",
        result=status_payload,
        result_artifact=status_artifact,
        evidence_artifact_path=TRAINING_EVIDENCE_ARTIFACT_PATH,
        evidence_classification="training",
        replay_route="POST /api/training/start",
    )
    return {"final_metrics": status_payload["final_metrics"], "training_status": "completed"}


def _training_record() -> StudioJobRecord:
    """Return a completed Training Monitor record declaring evidence."""

    return StudioJobRecord(
        job_id="sj_training",
        kind="training",
        owner="studio-training",
        request_id="req-1",
        status="completed",
        execution_model="thread",
        created_at_utc="2026-06-20T00:00:00Z",
        artifacts=(
            StudioJobArtifact(
                relative_path=TRAINING_EVIDENCE_ARTIFACT_PATH,
                size_bytes=128,
                sha256="0" * 64,
            ),
        ),
    )


def _training_payload_reader(
    *,
    evidence_classification: str = "training",
    status: str = "completed",
    payload_overrides: Mapping[str, object] | None = None,
) -> Callable[[str, str], StudioJobArtifactPayload]:
    """Return an artifact reader with configurable Training Monitor evidence."""

    def reader(_job_id: str, _relative_path: str) -> StudioJobArtifactPayload:
        payload: dict[str, object] = {
            "action_kind": "studio.training.run",
            "artifacts": [],
            "evidence_classification": evidence_classification,
            "job_id": "sj_training",
            "payload_sha256": "1" * 64,
            "replay_route": "POST /api/training/start",
            "schema_version": STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION,
            "status": status,
        }
        if payload_overrides is not None:
            payload.update(payload_overrides)
        return _payload(payload)

    return reader


def _training_raw_payload_reader(payload: object) -> Callable[[str, str], StudioJobArtifactPayload]:
    """Return an artifact reader that emits a caller-supplied JSON payload."""

    def reader(_job_id: str, _relative_path: str) -> StudioJobArtifactPayload:
        return _payload(payload)

    return reader


def _payload(payload: object) -> StudioJobArtifactPayload:
    """Return an artifact payload encoded as UTF-8 JSON bytes."""

    return StudioJobArtifactPayload(
        artifact=_training_record().artifacts[0],
        payload=json.dumps(payload).encode("utf-8"),
    )


def _unavailable_training_summary() -> dict[str, object]:
    """Return the bounded unavailable summary for malformed evidence."""

    artifact = _training_record().artifacts[0]
    return {
        "evidence_artifact": artifact.to_public_dict(),
        "job_id": "sj_training",
        "schema_version": TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION,
        "status": "unavailable",
    }


def _valid_training_summary() -> dict[str, object]:
    """Return a mutable verified Training Monitor evidence summary."""

    summary = build_training_evidence_summary(_training_record(), _training_payload_reader())
    assert isinstance(summary, dict)
    return dict(summary)


__all__ = [
    "json",
    "Callable",
    "Mapping",
    "Path",
    "cast",
    "pytest",
    "STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION",
    "write_studio_action_evidence_manifest",
    "StudioJobArtifact",
    "StudioJobArtifactPayload",
    "StudioJobArtifactUnavailable",
    "StudioJobContext",
    "StudioJobManager",
    "StudioJobRecord",
    "TRAINING_EVIDENCE_ARTIFACT_PATH",
    "TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION",
    "build_training_evidence_summary",
    "validate_training_evidence_summary",
    "get_training_status",
    "training_evidence_task",
    "_training_record",
    "_training_payload_reader",
    "_training_raw_payload_reader",
    "_payload",
    "_unavailable_training_summary",
    "_valid_training_summary",
]
