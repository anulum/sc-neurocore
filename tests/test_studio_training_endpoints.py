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

