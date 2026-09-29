# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training sse

"""Focused suite: TestSSEStream from former test_studio_training.py."""

from __future__ import annotations

from sc_neurocore.studio import _training_stream as training_stream
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

    def test_a_finished_proxy_waits_for_the_sealed_record_instead_of_closing(
        self, tmp_path: Path
    ) -> None:
        """The stream used to end after heartbeats when the proxy finished first.

        The browser then kept the run "running" and every follow-up action
        disabled (seen as an intermittent failure of the live event suite).
        """
        manager = StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset({"training"}),
            default_timeout_seconds=5.0,
        )
        release = threading.Event()

        def task(context: StudioJobContext) -> dict[str, object]:
            release.wait(timeout=3.0)
            return {"final_metrics": {"train_accuracy": 0.75}, "training_status": "completed"}

        record = manager.submit(
            kind="training", owner="studio-training", request_id=None, task=task
        )
        proxy = TrainingJob({"epochs": 1}, job_id=record.job_id)
        proxy.status = "completed"
        _register_job(proxy)

        frames: list[dict[str, object]] = []

        def consume() -> None:
            for frame in stream_metrics(record.job_id, manager):
                frames.append(json.loads(frame.removeprefix("data: ").strip()))

        reader = threading.Thread(target=consume)
        reader.start()
        time.sleep(1.5)
        release.set()
        manager.wait(record.job_id, timeout_seconds=5.0)
        reader.join(timeout=10.0)

        assert not reader.is_alive()
        assert frames[-1]["event"] == "completed"
        assert frames[-1]["data"] == {"train_accuracy": 0.75}

    def test_without_a_manager_a_drained_proxy_stream_ends_silently(self) -> None:
        """A proxy on its own already handed its terminal event to the reader."""
        proxy = TrainingJob({"epochs": 1}, job_id="sj_proxy_only_finished")
        proxy.status = "completed"
        _register_job(proxy)

        assert list(stream_metrics(proxy.id)) == []

    def test_a_record_that_never_seals_ends_with_the_proxys_own_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The grace window is bounded: the stream then says how the proxy ended."""
        monkeypatch.setattr(training_stream, "PROXY_TERMINAL_GRACE_POLLS", 2)
        manager = StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset({"training"}),
            default_timeout_seconds=30.0,
        )
        release = threading.Event()

        def task(context: StudioJobContext) -> dict[str, object]:
            release.wait(timeout=20.0)
            return {"training_status": "completed"}

        record = manager.submit(
            kind="training", owner="studio-training", request_id=None, task=task
        )
        proxy = TrainingJob({"epochs": 1}, job_id=record.job_id)
        proxy.status = "completed"
        proxy.final_metrics = {"val_accuracy": 0.5}
        _register_job(proxy)

        try:
            frames = [
                json.loads(frame.removeprefix("data: ").strip())
                for frame in stream_metrics(record.job_id, manager)
            ]
        finally:
            release.set()
            manager.wait(record.job_id, timeout_seconds=25.0)

        assert frames[-1]["event"] == "completed"
        assert frames[-1]["data"] == {"val_accuracy": 0.5}

    @pytest.mark.parametrize(
        ("status", "error", "expected"),
        [
            ("stopped", None, {"event": "stopped", "data": {}}),
            ("failed", "out of memory", {"event": "error", "data": {"message": "out of memory"}}),
            ("interrupted", None, {"event": "error", "data": {"message": "Training failed."}}),
        ],
    )
    def test_the_proxys_verdict_names_how_an_unfinished_run_ended(
        self, status: str, error: str | None, expected: dict[str, object]
    ) -> None:
        proxy = TrainingJob({"epochs": 1}, job_id=f"sj_proxy_{status}")
        proxy.status = status
        proxy.error = error

        event = training_stream._event_from_proxy(proxy)

        assert {key: event[key] for key in ("event", "data")} == expected
