# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training job lifecycle

"""Focused suite: TestJobLifecycle from former test_studio_training.py."""

from __future__ import annotations

from tests.studio_training_support import *  # noqa: F403


class TestJobLifecycle:
    def test_create_job(self) -> None:
        job = TrainingJob({"epochs": 1, "dataset": "synthetic"})
        assert job.status == "pending"
        assert job.id.startswith("j")
        assert job.error is None

    def test_start_training_returns_job_id(self) -> None:
        result = start_training({"epochs": 1, "dataset": "synthetic", "batch_size": 32})
        assert "job_id" in result
        assert result["status"] == "running"

    def test_job_appears_in_list(self) -> None:
        result = start_training({"epochs": 1, "dataset": "synthetic"})
        jobs = list_jobs()
        ids = [j["job_id"] for j in jobs]
        assert result["job_id"] in ids

    def test_get_status_existing_job(self) -> None:
        result = start_training({"epochs": 1, "dataset": "synthetic"})
        status = get_training_status(result["job_id"])
        assert status["job_id"] == result["job_id"]
        assert status["status"] in ("running", "completed", "pending")

    def test_get_status_nonexistent(self) -> None:
        status = get_training_status("nonexistent_id")
        assert "error" in status

    def test_stop_training(self) -> None:
        result = start_training({"epochs": 50, "dataset": "synthetic"})
        stop_result = stop_training(result["job_id"])
        assert stop_result["status"] == "stopping"

    def test_stop_nonexistent(self) -> None:
        result = stop_training("nonexistent_id")
        assert "error" in result

    def test_blocking_training_writes_terminal_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Bounded training writes status and action evidence artifacts."""

        def complete_training(job: TrainingJob, context: object = None) -> None:
            job.status = "completed"
            job.final_metrics = {
                "train_loss": 0.1,
                "train_accuracy": 0.9,
                "val_loss": 0.2,
                "val_accuracy": 0.8,
            }

        monkeypatch.setattr(TrainingJob, "_train", complete_training)
        context = StudioJobContext(
            job_id="sj_training",
            work_dir=tmp_path,
            cancel_event=threading.Event(),
            max_artifact_bytes=4096,
        )
        job = TrainingJob({"epochs": 1}, job_id="sj_training")

        result = job.run_blocking(context)

        assert result["training_status"] == "completed"
        assert [artifact.relative_path for artifact in context.artifacts] == [
            "training/status.json",
            "training/evidence.json",
        ]
        evidence = json.loads((tmp_path / "training" / "evidence.json").read_text())
        assert evidence["schema_version"] == STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION
        assert evidence["action_kind"] == "studio.training.run"
        assert evidence["evidence_classification"] == "training"
        assert evidence["job_id"] == "sj_training"
        assert evidence["replay_route"] == "POST /api/training/start"
        assert evidence["status"] == "completed"
        assert evidence["artifacts"][0]["relative_path"] == "training/status.json"

    def test_blocking_training_failure_writes_error_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Failed bounded training writes failed evidence before propagating."""

        def fail_training(job: TrainingJob, context: object = None) -> None:
            raise RuntimeError("training boom")

        monkeypatch.setattr(TrainingJob, "_train", fail_training)
        context = StudioJobContext(
            job_id="sj_training_failed",
            work_dir=tmp_path,
            cancel_event=threading.Event(),
            max_artifact_bytes=4096,
        )
        job = TrainingJob({"epochs": 1}, job_id="sj_training_failed")

        with pytest.raises(RuntimeError, match="training boom"):
            job.run_blocking(context)

        evidence = json.loads((tmp_path / "training" / "evidence.json").read_text())
        assert evidence["error_message"] == "training boom"
        assert evidence["status"] == "failed"

    def test_blocking_training_stop_writes_cancelled_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Stopped bounded training writes cancelled evidence before propagating."""

        def stop_training_run(job: TrainingJob, context: object = None) -> None:
            job.status = "stopped"

        monkeypatch.setattr(TrainingJob, "_train", stop_training_run)
        context = StudioJobContext(
            job_id="sj_training_stopped",
            work_dir=tmp_path,
            cancel_event=threading.Event(),
            max_artifact_bytes=4096,
        )
        job = TrainingJob({"epochs": 1}, job_id="sj_training_stopped")

        with pytest.raises(StudioJobCancelled, match="stopped"):
            job.run_blocking(context)

        evidence = json.loads((tmp_path / "training" / "evidence.json").read_text())
        assert evidence["status"] == "cancelled"
