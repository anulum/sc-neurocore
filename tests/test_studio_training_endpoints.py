# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training endpoints

"""Focused suite: TestTrainingEndpoints from former test_studio_training.py."""

from __future__ import annotations

from tests.studio_training_support import *  # noqa: F403
from sc_neurocore.studio.training_contract import resolve_training_config
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger


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

    def test_stop_endpoint_reports_a_run_that_already_finished(self, client: TestClient) -> None:
        """Pressing Stop on a run that just finished is not a server error.

        Under load the run reaches a terminal state between the operator
        reading the page and the request arriving; the stop path used to
        propagate the ledger's refusal as HTTP 500.
        """
        start = client.post(
            "/api/training/start",
            json={"epochs": 1, "dataset": "synthetic", "batch_size": 32, "timesteps": 4},
        )
        job_id = start.json()["job_id"]
        deadline = time.monotonic() + 300.0
        status = ""
        while time.monotonic() < deadline:
            status = client.get(f"/api/training/status/{job_id}").json().get("status", "")
            if status in {"completed", "failed", "stopped"}:
                break
            time.sleep(0.2)
        assert status in {"completed", "failed", "stopped"}, status

        stopped = client.post("/api/training/stop", json={"job_id": job_id})

        assert stopped.status_code == 200
        body = stopped.json()
        assert body["job_id"] == job_id
        assert body["status"] != "stopping"

    def test_stop_endpoint_reads_retained_record_without_proxy(self, tmp_path: Path) -> None:
        """A restarted API can report a durable training run without its proxy."""
        settings = StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs"))
        app = create_app(settings)
        manager = cast(StudioJobManager, app.state.studio_job_manager)
        submitted = manager.submit(
            kind="training",
            owner="studio-training",
            request_id=None,
            task=lambda _context: {},
        )
        assert manager.wait(submitted.job_id, timeout_seconds=5.0).status == "completed"
        restarted_app = create_app(settings)

        response = TestClient(restarted_app, base_url="http://127.0.0.1").post(
            "/api/training/stop", json={"job_id": submitted.job_id}
        )
        listed = TestClient(restarted_app, base_url="http://127.0.0.1").get("/api/training/jobs")

        assert response.status_code == 200
        assert response.json() == {"job_id": submitted.job_id, "status": "completed"}
        assert listed.status_code == 200
        assert {
            "job_id": submitted.job_id,
            "status": "completed",
            "config": None,
        } in listed.json()

    def test_stop_endpoint_maps_retained_cancelled_to_stopped(self, tmp_path: Path) -> None:
        """The public Stop route never leaks the platform's cancelled status."""
        settings = StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs"))
        app = create_app(settings)
        manager = cast(StudioJobManager, app.state.studio_job_manager)

        def cancelled(_context: StudioJobContext) -> dict[str, object]:
            raise StudioJobCancelled("operator cancelled")

        submitted = manager.submit(
            kind="training", owner="studio-training", request_id=None, task=cancelled
        )
        assert manager.wait(submitted.job_id, timeout_seconds=5.0).status == "cancelled"
        restarted = TestClient(create_app(settings), base_url="http://127.0.0.1")

        response = restarted.post("/api/training/stop", json={"job_id": submitted.job_id})

        assert response.status_code == 200
        assert response.json() == {"job_id": submitted.job_id, "status": "stopped"}

    def test_stop_endpoint_keeps_unknown_record_uncertain(self, tmp_path: Path) -> None:
        """An unknown worker cannot be reported as cooperatively stopping."""
        settings = StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs"))
        app = create_app(settings)
        ledger = StudioJobLedger(root=tmp_path / "jobs")
        created = ledger.create(
            job_id="sj_0000000000000001",
            kind="training",
            actor="studio-training",
            workspace="default",
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission=None,
            execution_model="process",
        ).record
        ledger.transition(created.job_id, "unknown", reason="worker liveness uncertain")
        ledger.close()
        client = TestClient(app, base_url="http://127.0.0.1")

        response = client.post("/api/training/stop", json={"job_id": created.job_id})

        assert response.status_code == 200
        assert response.json() == {"job_id": created.job_id, "status": "unknown"}
        assert client.get(f"/api/training/status/{created.job_id}").json()["status"] == "unknown"

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

    def test_restarted_training_reports_interrupted_record(self, tmp_path: Path) -> None:
        """Restarted HTTP status and list expose an interrupted durable run."""
        root = tmp_path / "jobs"
        ledger = StudioJobLedger(root=root)
        config = resolve_training_config({"epochs": 1}).to_public_dict()
        created = ledger.create(
            job_id="sj_0000000000000001",
            kind="training",
            actor="studio-training",
            workspace="default",
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission=None,
            execution_model="process",
            training_config=config,
        ).record
        ledger.transition(created.job_id, "interrupted", reason="supervisor exited")
        ledger.close()
        app = create_app(StudioRuntimeSettings(job_root_path=str(root)))
        restarted = TestClient(app, base_url="http://127.0.0.1")

        status = restarted.get(f"/api/training/status/{created.job_id}")
        listed = restarted.get("/api/training/jobs")

        assert status.status_code == 200
        assert status.json()["status"] == "interrupted"
        assert listed.status_code == 200
        assert listed.json() == [
            {"job_id": created.job_id, "status": "interrupted", "config": config}
        ]

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
        expected = resolve_training_config(
            {"epochs": 1, "dataset": "synthetic", "batch_size": 32}
        ).to_public_dict()
        assert manager.record(job_id).training_config == expected
        restarted = TestClient(create_app(settings), base_url="http://127.0.0.1").get(
            "/api/training/jobs"
        )
        assert restarted.status_code == 200
        assert any(
            row["job_id"] == job_id and row["config"] == expected for row in restarted.json()
        )
        assert (job_root / job_id / ".studio_process_payload.json").is_file()
