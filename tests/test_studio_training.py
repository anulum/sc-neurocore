# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Training Monitor (Block 4)

from __future__ import annotations

import time

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.training import (
    TrainingJob,
    _CELL_TYPES,
    _SURROGATES,
    get_training_status,
    list_cell_types,
    list_jobs,
    list_surrogates,
    start_training,
    stop_training,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1")


# --- Surrogate & Cell Type Listing ---


class TestListing:
    def test_list_surrogates(self):
        result = list_surrogates()
        assert len(result) == len(_SURROGATES)
        names = {s["name"] for s in result}
        assert "atan_surrogate" in names
        assert "fast_sigmoid" in names

    def test_list_cell_types(self):
        result = list_cell_types()
        assert len(result) == len(_CELL_TYPES)
        names = {c["name"] for c in result}
        assert "LIFCell" in names
        assert "AdExCell" in names

    def test_surrogates_endpoint(self, client):
        r = client.get("/api/training/surrogates")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == len(_SURROGATES)
        assert all("name" in s for s in data)
        assert all("available" in s for s in data)

    def test_cell_types_endpoint(self, client):
        r = client.get("/api/training/cell-types")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == len(_CELL_TYPES)


# --- Training Job Lifecycle ---


class TestJobLifecycle:
    def test_create_job(self):
        job = TrainingJob({"epochs": 1, "dataset": "synthetic"})
        assert job.status == "pending"
        assert job.id.startswith("j")
        assert job.error is None

    def test_start_training_returns_job_id(self):
        result = start_training({"epochs": 1, "dataset": "synthetic", "batch_size": 32})
        assert "job_id" in result
        assert result["status"] == "running"

    def test_job_appears_in_list(self):
        result = start_training({"epochs": 1, "dataset": "synthetic"})
        jobs = list_jobs()
        ids = [j["job_id"] for j in jobs]
        assert result["job_id"] in ids

    def test_get_status_existing_job(self):
        result = start_training({"epochs": 1, "dataset": "synthetic"})
        status = get_training_status(result["job_id"])
        assert status["job_id"] == result["job_id"]
        assert status["status"] in ("running", "completed", "pending")

    def test_get_status_nonexistent(self):
        status = get_training_status("nonexistent_id")
        assert "error" in status

    def test_stop_training(self):
        result = start_training({"epochs": 50, "dataset": "synthetic"})
        stop_result = stop_training(result["job_id"])
        assert stop_result["status"] == "stopping"

    def test_stop_nonexistent(self):
        result = stop_training("nonexistent_id")
        assert "error" in result


# --- Training Endpoints ---


class TestTrainingEndpoints:
    def test_start_endpoint(self, client):
        r = client.post(
            "/api/training/start",
            json={"epochs": 1, "dataset": "synthetic", "batch_size": 32},
        )
        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data

    def test_stop_endpoint_requires_job_id(self, client):
        r = client.post("/api/training/stop", json={})
        assert r.status_code == 422

    def test_stop_endpoint(self, client):
        start = client.post(
            "/api/training/start",
            json={"epochs": 50, "dataset": "synthetic"},
        )
        job_id = start.json()["job_id"]
        r = client.post("/api/training/stop", json={"job_id": job_id})
        assert r.status_code == 200

    def test_status_endpoint(self, client):
        start = client.post(
            "/api/training/start",
            json={"epochs": 1, "dataset": "synthetic"},
        )
        job_id = start.json()["job_id"]
        r = client.get(f"/api/training/status/{job_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == job_id

    def test_status_nonexistent(self, client):
        r = client.get("/api/training/status/nonexistent")
        assert r.status_code == 404

    def test_jobs_endpoint(self, client):
        r = client.get("/api/training/jobs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# --- SSE Stream ---


class TestSSEStream:
    def test_stream_nonexistent_job(self, client):
        r = client.get("/api/training/stream/nonexistent")
        assert r.status_code == 200
        content = r.text
        assert "error" in content or "not found" in content.lower()

    def test_stream_endpoint_returns_event_stream(self, client):
        start = client.post(
            "/api/training/start",
            json={"epochs": 2, "dataset": "synthetic", "batch_size": 32},
        )
        job_id = start.json()["job_id"]
        # Give it a moment to produce events
        time.sleep(0.5)
        r = client.get(f"/api/training/stream/{job_id}")
        assert r.headers.get("content-type", "").startswith("text/event-stream")


# --- Training Config Validation ---


class TestTrainingConfig:
    def test_default_config_runs(self):
        result = start_training({"epochs": 2, "batch_size": 32, "dataset": "synthetic"})
        assert result["status"] == "running"
        # Wait for completion (synthetic is fast)
        for _ in range(20):
            time.sleep(1)
            status = get_training_status(result["job_id"])
            if status["status"] in ("completed", "failed"):
                break
        assert status["status"] == "completed", f"Expected completed, got {status}"

    def test_all_surrogates_listed(self):
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
