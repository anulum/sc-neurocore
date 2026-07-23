# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training sse

"""Focused suite: TestSSEStream from former test_studio_training.py."""

from __future__ import annotations

from tests.studio_training_support import *  # noqa: F403

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

