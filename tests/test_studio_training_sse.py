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
from sc_neurocore.studio.platform.jobs_failures import GENERIC_JOB_FAILURE
from tests.studio_seccomp_support import run_child
from tests.studio_training_support import *  # noqa: F403


def test_drained_legacy_stream_keeps_a_real_dataset_failure_private(tmp_path: Path) -> None:
    """A public legacy run retains its safe fallback after its error queue is drained."""
    result = run_child(
        "import json, os, sys, threading, time\n"
        "from pathlib import Path\n"
        "from sc_neurocore.datasets.encoders import EventBinning\n"
        "from sc_neurocore.datasets.manifest import build_manifest\n"
        "from sc_neurocore.datasets.splits import group_split\n"
        "from sc_neurocore.studio.event_training_contract import EventTrainingContract\n"
        "from sc_neurocore.studio.platform.jobs import StudioJobManager\n"
        "from sc_neurocore.studio.training import start_training, get_training_status, stream_metrics\n"
        "from tests.event_dataset_support import write_nmnist, nmnist_event_bytes\n"
        "from tests.studio_syscall_support import hold_system_calls, finish\n"
        "root = Path(sys.argv[1])\n"
        "corpus = root / 'recordings'\n"
        "write_nmnist(corpus, {'train': {0: 3, 1: 2}})\n"
        "os.environ['SC_NEUROCORE_STUDIO_DATASET_ROOT'] = str(corpus)\n"
        "manifest = build_manifest('nmnist', corpus, version='generated-format-fixture')\n"
        "contract = EventTrainingContract(manifest,\n"
        "    group_split(manifest, fractions={'train': 0.6, 'evaluation': 0.4}, seed=7),\n"
        "    EventBinning(1.0, 4, 34, 34, 'merge'), 'train', 'evaluation')\n"
        "config = {'dataset': 'nmnist', 'epochs': 1, 'batch_size': 3, 'hidden': [4],\n"
        "    'timesteps': 4, 'seed': 7, 'event_data': contract.to_dict()}\n"
        "target = corpus / manifest.files[0].path\n"
        "main_thread, changed = threading.get_native_id(), []\n"
        "manager = StudioJobManager(root=root / 'empty-ledger',\n"
        "    allowed_kinds=frozenset({'training'}), default_timeout_seconds=5.0)\n"
        "def decide(call):\n"
        "    if not changed and call.thread != main_thread and call.text(1) == str(target):\n"
        "        target.write_bytes(target.read_bytes() + nmnist_event_bytes([(1, 1, 0, 1000)]))\n"
        "        changed.append(True)\n"
        "    return None\n"
        "hold_system_calls(['openat'], decide)\n"
        "started = start_training(config)\n"
        "job = started['job_id']\n"
        "deadline = time.monotonic() + 10.0\n"
        "while get_training_status(job)['status'] != 'failed':\n"
        "    assert time.monotonic() < deadline\n"
        "    time.sleep(0.01)\n"
        "status = get_training_status(job)\n"
        "decode = lambda frame: json.loads(frame.removeprefix('data: ').strip())\n"
        "initial = [decode(frame) for frame in stream_metrics(job)]\n"
        "replayed = [decode(frame) for frame in stream_metrics(job, manager)]\n"
        "finish({'changed': changed, 'status': status, 'initial': initial,\n"
        "    'replayed': replayed, 'durable_records': len(manager.list_records())})\n",
        arguments=(str(tmp_path),),
    )
    assert result["changed"] == [True]
    status = result["status"]
    assert isinstance(status, dict) and status["status"] == "failed"
    assert status["error"] == GENERIC_JOB_FAILURE
    initial = result["initial"]
    assert isinstance(initial, list) and initial[-1]["event"] == "error"
    assert initial[-1]["data"]["message"] == GENERIC_JOB_FAILURE
    replayed = result["replayed"]
    assert isinstance(replayed, list) and len(replayed) == 1
    assert replayed[0]["event"] == "error"
    assert replayed[0]["data"] == {"message": GENERIC_JOB_FAILURE}
    assert result["durable_records"] == 0
    assert str(tmp_path) not in json.dumps(result)


def test_http_stream_keeps_an_empty_terminal_event_without_inventing_an_error(
    tmp_path: Path,
) -> None:
    """A real job's empty stopped event passes through public SSE unchanged."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        """Persist the actual empty terminal event through the public context."""
        context.append_artifact_event(
            "training/events.jsonl", {"event": "stopped", "data": {}, "timestamp": 1.0}
        )
        return {"training_status": "stopped"}

    try:
        job = manager.submit(kind="training", owner="studio-training", request_id=None, task=task)
        record = manager.wait(job.job_id, timeout_seconds=10.0)
        assert record.status == "completed" and record.public_error is None
        raw, end = manager.read_live_artifact_bytes(job.job_id, "training/events.jsonl", offset=0)
        assert end == len(raw) and raw
        persisted = json.loads(raw)
        assert persisted == {"event": "stopped", "data": {}, "timestamp": 1.0}
        response = TestClient(app, base_url="http://127.0.0.1").get(
            f"/api/training/stream/{job.job_id}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = [frame for frame in response.text.split("\n\n") if frame.strip()]
        assert len(frames) == 1
        assert json.loads(frames[0].removeprefix("data: ")) == persisted
        assert manager.status().active_count == 0 and manager.unreaped_workers == ()
    finally:
        manager._ledger.close()


@pytest.mark.parametrize("failure", ["os-error", "timeout"])
def test_http_stream_replays_real_failure_without_exposing_diagnostics(
    tmp_path: Path, failure: str
) -> None:
    """Real worker failure and timeout replay once through the public SSE endpoint."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    manager = cast(StudioJobManager, app.state.studio_job_manager)
    release = threading.Event()

    def task(context: StudioJobContext) -> dict[str, object]:
        if failure == "os-error":
            context.write_artifact("x" * 300, b"data")
        else:
            while not release.wait(0.01):
                context.check_cancelled()
        return {}

    record = manager.submit(
        kind="training",
        owner="studio-training",
        request_id=None,
        task=task,
        timeout_seconds=0.1 if failure == "timeout" else 10.0,
    )
    try:
        outcome = manager.wait(record.job_id, timeout_seconds=10.0)
        assert outcome.status == ("failed" if failure == "os-error" else "timed_out")
        assert outcome.error is not None
        if failure == "os-error":
            assert str(tmp_path) in outcome.error
        response = TestClient(app, base_url="http://127.0.0.1").get(
            f"/api/training/stream/{record.job_id}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = [frame for frame in response.text.split("\n\n") if frame.strip()]
        assert len(frames) == 1
        event = json.loads(frames[0].removeprefix("data: "))
        assert event["event"] == ("error" if failure == "os-error" else "stopped")
        assert event["data"] == {"message": outcome.public_error}
        assert str(tmp_path) not in response.text
        assert manager.status().active_count == 0 and manager.unreaped_workers == ()
    finally:
        release.set()
        manager._ledger.close()


class TestSSEStream:
    """Exercise training stream and terminal projection compatibility."""

    def test_stream_nonexistent_job(self, client: TestClient) -> None:
        """A missing training job emits an error stream response."""
        r = client.get("/api/training/stream/nonexistent")
        assert r.status_code == 200
        content = r.text
        assert "error" in content or "not found" in content.lower()

    def test_stream_endpoint_returns_event_stream(self, client: TestClient) -> None:
        """The training endpoint supplies the SSE media type."""
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
            (
                "failed",
                "out of memory",
                {"event": "error", "data": {"message": GENERIC_JOB_FAILURE}},
            ),
            ("interrupted", None, {"event": "error", "data": {"message": "Training failed."}}),
        ],
    )
    def test_the_proxys_verdict_names_how_an_unfinished_run_ended(
        self, status: str, error: str | None, expected: dict[str, object]
    ) -> None:
        """An unqualified proxy diagnostic uses the fixed public failure."""
        proxy = TrainingJob({"epochs": 1}, job_id=f"sj_proxy_{status}")
        proxy.status = status
        proxy.error = error

        event = training_stream._event_from_proxy(proxy)

        assert {key: event[key] for key in ("event", "data")} == expected
