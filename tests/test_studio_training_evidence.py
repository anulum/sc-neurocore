# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training evidence summary tests

"""Tests for Training Monitor operator evidence summaries."""

from __future__ import annotations

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


def test_training_status_includes_verified_evidence_summary(tmp_path: Path) -> None:
    """Training status exposes path-free evidence metadata after completion."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=1.0,
    )
    submitted = manager.submit(
        kind="training",
        owner="studio-training",
        request_id="req-1",
        task=training_evidence_task,
    )
    completed = manager.wait(submitted.job_id, timeout_seconds=2.0)

    status = get_training_status(completed.job_id, manager)

    assert status["status"] == "completed"
    assert status["final_metrics"] == {
        "train_accuracy": 0.9,
        "train_loss": 0.1,
        "val_accuracy": 0.8,
        "val_loss": 0.2,
    }
    summary = status["evidence_summary"]
    assert isinstance(summary, dict)
    assert summary["schema_version"] == TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION
    assert summary["action_kind"] == "studio.training.run"
    assert summary["evidence_classification"] == "training"
    assert summary["job_id"] == completed.job_id
    assert summary["replay_route"] == "POST /api/training/start"
    assert summary["status"] == "completed"
    assert summary["evidence_artifact"] == completed.artifacts[1].to_public_dict()
    assert summary["result_artifacts"] == [completed.artifacts[0].to_public_dict()]
    evidence_payload = json.loads(
        manager.read_artifact(completed.job_id, TRAINING_EVIDENCE_ARTIFACT_PATH).payload.decode(
            "utf-8"
        )
    )
    assert evidence_payload["schema_version"] == STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION


def test_training_evidence_summary_reports_unavailable_artifact() -> None:
    """Unreadable declared evidence artifacts produce a bounded summary."""

    artifact = StudioJobArtifact(
        relative_path=TRAINING_EVIDENCE_ARTIFACT_PATH,
        size_bytes=128,
        sha256="0" * 64,
    )
    record = StudioJobRecord(
        job_id="sj_training",
        kind="training",
        owner="studio-training",
        request_id=None,
        status="completed",
        execution_model="thread",
        created_at_utc="2026-06-20T00:00:00Z",
        artifacts=(artifact,),
    )

    def unavailable_reader(
        _job_id: str,
        _relative_path: str,
    ) -> StudioJobArtifactPayload:
        raise StudioJobArtifactUnavailable("missing")

    summary = build_training_evidence_summary(record, unavailable_reader)

    assert summary == {
        "evidence_artifact": artifact.to_public_dict(),
        "job_id": "sj_training",
        "schema_version": TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION,
        "status": "unavailable",
    }


def test_training_evidence_summary_returns_none_without_declared_artifact() -> None:
    """Records without Training Monitor evidence do not fabricate summaries."""

    record = StudioJobRecord(
        job_id="sj_training",
        kind="training",
        owner="studio-training",
        request_id=None,
        status="completed",
        execution_model="thread",
        created_at_utc="2026-06-20T00:00:00Z",
        artifacts=(),
    )

    assert build_training_evidence_summary(record, _training_payload_reader()) is None


def test_training_evidence_summary_rejects_unknown_classification() -> None:
    """Training evidence summaries fail closed on unknown evidence classes."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(evidence_classification="screenshots"),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_non_terminal_status() -> None:
    """Training evidence summaries fail closed on non-terminal evidence statuses."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(status="running"),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_non_object_payload() -> None:
    """Training evidence summaries fail closed on non-object JSON payloads."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_raw_payload_reader(["not", "an", "object"]),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_unsupported_schema() -> None:
    """Training evidence summaries fail closed on unsupported schema versions."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"schema_version": "studio.old.v1"}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_unsupported_action() -> None:
    """Training evidence summaries fail closed on unsupported action kinds."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"action_kind": "studio.compile"}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_missing_status() -> None:
    """Training evidence summaries fail closed when evidence status is absent."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"status": None}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_missing_job_id() -> None:
    """Training evidence summaries fail closed when evidence omits job identity."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"job_id": None}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_empty_required_string() -> None:
    """Training evidence summaries fail closed on empty required string fields."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"payload_sha256": ""}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_missing_artifact_list() -> None:
    """Training evidence summaries fail closed when result artifact data is absent."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"artifacts": None}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_non_object_artifact_entry() -> None:
    """Training evidence summaries fail closed on malformed artifact metadata."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"artifacts": ["bad"]}),
    )

    assert summary == _unavailable_training_summary()


def test_validate_training_evidence_summary_accepts_verified_summary() -> None:
    """Evidence summary validator accepts verified Training Monitor summaries."""

    summary = build_training_evidence_summary(_training_record(), _training_payload_reader())
    assert isinstance(summary, dict)
    summary["duration_seconds"] = 1.5

    validated = validate_training_evidence_summary(summary)

    assert validated == summary


def test_validate_training_evidence_summary_rejects_unavailable_summary() -> None:
    """Evidence summary validator rejects bounded unavailable summaries."""

    with pytest.raises(ValueError, match="action_kind"):
        validate_training_evidence_summary(_unavailable_training_summary())


def test_validate_training_evidence_summary_rejects_forged_artifact_path() -> None:
    """Evidence summary validator rejects non-confined artifact metadata."""

    summary = build_training_evidence_summary(_training_record(), _training_payload_reader())
    assert isinstance(summary, dict)
    evidence_artifact = summary["evidence_artifact"]
    assert isinstance(evidence_artifact, dict)
    evidence_artifact["relative_path"] = "../training/evidence.json"

    with pytest.raises(ValueError, match="path"):
        validate_training_evidence_summary(summary)


@pytest.mark.parametrize(
    ("mutator", "error_match"),
    [
        (lambda payload: payload.__setitem__("schema_version", "studio.old.v1"), "schema"),
        (lambda payload: payload.__setitem__("action_kind", "studio.compile"), "action"),
        (
            lambda payload: payload.__setitem__("evidence_classification", "compile"),
            "classification",
        ),
        (lambda payload: payload.__setitem__("payload_sha256", "bad"), "payload digest"),
        (lambda payload: payload.__setitem__("result_artifacts", None), "result artifacts"),
        (lambda payload: payload.__setitem__("replay_route", ""), "replay_route"),
    ],
)
def test_validate_training_evidence_summary_rejects_invalid_fields(
    mutator: Callable[[dict[str, object]], None],
    error_match: str,
) -> None:
    """Evidence summary validator rejects malformed top-level fields."""

    summary = _valid_training_summary()
    mutator(summary)

    with pytest.raises(ValueError, match=error_match):
        validate_training_evidence_summary(summary)


@pytest.mark.parametrize(
    ("evidence_artifact", "error_match"),
    [
        (None, "evidence_artifact"),
        ({"relative_path": "training/evidence.txt", "sha256": "0" * 64, "size_bytes": 128}, "path"),
        (
            {"relative_path": TRAINING_EVIDENCE_ARTIFACT_PATH, "sha256": "bad", "size_bytes": 128},
            "digest",
        ),
        (
            {
                "relative_path": TRAINING_EVIDENCE_ARTIFACT_PATH,
                "sha256": "0" * 64,
                "size_bytes": -1,
            },
            "size",
        ),
        (
            {
                "relative_path": TRAINING_EVIDENCE_ARTIFACT_PATH,
                "sha256": "0" * 64,
                "size_bytes": 128,
                1: "bad",
            },
            "must be JSON",
        ),
    ],
)
def test_validate_training_evidence_summary_rejects_invalid_evidence_artifact(
    evidence_artifact: object,
    error_match: str,
) -> None:
    """Evidence summary validator rejects malformed evidence artifact metadata."""

    summary = _valid_training_summary()
    summary["evidence_artifact"] = evidence_artifact

    with pytest.raises(ValueError, match=error_match):
        validate_training_evidence_summary(summary)


@pytest.mark.parametrize(
    ("result_artifact", "error_match"),
    [
        ("bad", "result_artifacts"),
        ({"relative_path": "/training/status.json", "sha256": "2" * 64, "size_bytes": 256}, "path"),
        ({"relative_path": "training/status.json", "sha256": "bad", "size_bytes": 256}, "digest"),
        ({"relative_path": "training/status.json", "sha256": "2" * 64, "size_bytes": -1}, "size"),
    ],
)
def test_validate_training_evidence_summary_rejects_invalid_result_artifact(
    result_artifact: object,
    error_match: str,
) -> None:
    """Evidence summary validator rejects malformed result artifact metadata."""

    summary = _valid_training_summary()
    summary["result_artifacts"] = [result_artifact]

    with pytest.raises(ValueError, match=error_match):
        validate_training_evidence_summary(summary)


@pytest.mark.parametrize(
    ("payload", "error_match"),
    [
        ({"schema_version": float("nan")}, "must be JSON"),
        ({1: "bad"}, "must be JSON"),
        (
            {"schema_version": TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION, "bad": object()},
            "must be JSON",
        ),
    ],
)
def test_validate_training_evidence_summary_rejects_non_portable_json(
    payload: Mapping[object, object],
    error_match: str,
) -> None:
    """Evidence summary validator rejects non-portable JSON payloads."""

    with pytest.raises(ValueError, match=error_match):
        validate_training_evidence_summary(cast(Mapping[str, object], payload))


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
