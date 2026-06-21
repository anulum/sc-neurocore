# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Training Monitor (Block 4)

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import cast

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION,
    StudioRuntimeSettings,
)
from sc_neurocore.studio.platform.jobs import (
    StudioJobCancelled,
    StudioJobContext,
    StudioJobManager,
)
from sc_neurocore.studio.training import (
    TrainingJob,
    _CELL_TYPES,
    _SURROGATES,
    _register_job,
    get_training_status,
    list_cell_types,
    list_jobs,
    list_surrogates,
    start_training,
    stop_training,
    stream_metrics,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


# --- Surrogate & Cell Type Listing ---


class TestListing:
    def test_list_surrogates(self) -> None:
        result = list_surrogates()
        assert len(result) == len(_SURROGATES)
        names = {s["name"] for s in result}
        assert "atan_surrogate" in names
        assert "fast_sigmoid" in names

    def test_list_cell_types(self) -> None:
        result = list_cell_types()
        assert len(result) == len(_CELL_TYPES)
        names = {c["name"] for c in result}
        assert "LIFCell" in names
        assert "AdExCell" in names

    def test_surrogates_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/training/surrogates")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == len(_SURROGATES)
        assert all("name" in s for s in data)
        assert all("available" in s for s in data)

    def test_cell_types_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/training/cell-types")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == len(_CELL_TYPES)


# --- Training Job Lifecycle ---


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


# --- Training Endpoints ---


class TestTrainingEndpoints:
    def test_start_endpoint(self, client: TestClient) -> None:
        r = client.post(
            "/api/training/start",
            json={"epochs": 1, "dataset": "synthetic", "batch_size": 32},
        )
        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data
        assert data["job_id"].startswith("sj_")

    def test_stop_endpoint_requires_job_id(self, client: TestClient) -> None:
        r = client.post("/api/training/stop", json={})
        assert r.status_code == 422

    def test_stop_endpoint(self, client: TestClient) -> None:
        start = client.post(
            "/api/training/start",
            json={"epochs": 50, "dataset": "synthetic"},
        )
        job_id = start.json()["job_id"]
        r = client.post("/api/training/stop", json={"job_id": job_id})
        assert r.status_code == 200

    def test_status_endpoint(self, client: TestClient) -> None:
        start = client.post(
            "/api/training/start",
            json={"epochs": 1, "dataset": "synthetic"},
        )
        job_id = start.json()["job_id"]
        r = client.get(f"/api/training/status/{job_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == job_id

    def test_status_nonexistent(self, client: TestClient) -> None:
        r = client.get("/api/training/status/nonexistent")
        assert r.status_code == 404

    def test_jobs_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/training/jobs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_training_endpoint_registers_platform_job(self, tmp_path: Path) -> None:
        job_root = tmp_path / "jobs"
        settings = StudioRuntimeSettings(
            job_root_path=str(job_root),
            job_default_timeout_seconds=10.0,
        )
        app = create_app(settings)
        client = TestClient(app, base_url="http://127.0.0.1")
        r = client.post(
            "/api/training/start",
            json={"epochs": 1, "dataset": "synthetic", "batch_size": 32},
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        manager = cast(StudioJobManager, app.state.studio_job_manager)
        records = manager.list_records()

        assert any(record.job_id == job_id and record.kind == "training" for record in records)
        assert (job_root / job_id / ".studio_process_payload.json").is_file()


# --- SSE Stream ---


class TestSSEStream:
    def test_stream_nonexistent_job(self, client: TestClient) -> None:
        r = client.get("/api/training/stream/nonexistent")
        assert r.status_code == 200
        content = r.text
        assert "error" in content or "not found" in content.lower()

    def test_stream_endpoint_returns_event_stream(self, client: TestClient) -> None:
        start = client.post(
            "/api/training/start",
            json={"epochs": 2, "dataset": "synthetic", "batch_size": 32},
        )
        job_id = start.json()["job_id"]
        # Give it a moment to produce events
        time.sleep(0.5)
        r = client.get(f"/api/training/stream/{job_id}")
        assert r.headers.get("content-type", "").startswith("text/event-stream")

    def test_stream_metrics_tails_process_worker_event_log(self, tmp_path: Path) -> None:
        """Parent-process SSE stream yields child-process live event rows."""

        manager = StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset({"training"}),
            default_timeout_seconds=2.0,
        )
        release = threading.Event()

        def task(context: StudioJobContext) -> dict[str, object]:
            context.append_artifact_event(
                "training/events.jsonl",
                {"event": "epoch", "data": {"epoch": 0}, "timestamp": 1.0},
            )
            release.wait(timeout=1.0)
            return {"final_metrics": {"train_accuracy": 0.5}, "training_status": "completed"}

        record = manager.submit(
            kind="training",
            owner="studio-training",
            request_id=None,
            task=task,
        )
        proxy = TrainingJob({"epochs": 1}, job_id=record.job_id)
        proxy.status = "running"
        _register_job(proxy)

        for _ in range(20):
            payload, _offset = manager.read_live_artifact_bytes(
                record.job_id,
                "training/events.jsonl",
                offset=0,
            )
            if payload:
                break
            time.sleep(0.05)
        generator = stream_metrics(record.job_id, manager)
        first_event = next(generator)
        release.set()
        manager.wait(record.job_id, timeout_seconds=2.0)

        assert json.loads(first_event.removeprefix("data: ").strip()) == {
            "data": {"epoch": 0},
            "event": "epoch",
            "timestamp": 1.0,
        }


# --- Training Config Validation ---


class TestTrainingConfig:
    def test_default_config_runs(self) -> None:
        result = start_training({"epochs": 2, "batch_size": 32, "dataset": "synthetic"})
        assert result["status"] == "running"
        # Wait for completion (synthetic is fast)
        status = get_training_status(result["job_id"])
        for _ in range(20):
            time.sleep(1)
            status = get_training_status(result["job_id"])
            if status["status"] in ("completed", "failed"):
                break
        assert status["status"] == "completed", f"Expected completed, got {status}"

    def test_all_surrogates_listed(self) -> None:
        expected = {
            "fast_sigmoid",
            "superspike",
            "atan_surrogate",
            "sigmoid_surrogate",
            "straight_through",
            "triangular",
        }
        actual = {s["name"] for s in list_surrogates()}
        assert expected == actual
