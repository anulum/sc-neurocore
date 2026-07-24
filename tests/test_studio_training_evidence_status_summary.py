# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (status_summary) from former test_studio_training_evidence.py

from __future__ import annotations

from tests.studio_training_evidence_support import *  # noqa: F403


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
