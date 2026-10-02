# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real job refusal HTTP contracts

"""Keep actual filesystem diagnostics private while retaining deliberate reasons."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_failures import GENERIC_JOB_FAILURE
from sc_neurocore.studio.platform.jobs_models import StudioJobRefused
from sc_neurocore.studio.training import TrainingJob


def test_catalogue_job_http_preserves_real_queue_refusal(tmp_path: Path) -> None:
    """A full real application queue refuses catalogue work before another admission."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path)))
    manager = cast(StudioJobManager, app.state.studio_job_manager)
    release = threading.Event()

    def task(context: StudioJobContext) -> dict[str, object]:
        """Keep real workers occupied until the HTTP refusal has been observed."""
        release.wait(60.0)
        context.check_cancelled()
        return {}

    limits = manager.status().admission
    admitted = []
    queued = []
    with ThreadPoolExecutor(max_workers=limits["max_queued"]) as submitters:
        try:
            for _ in range(limits["max_concurrent"]):
                admitted.append(
                    manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
                )
            for _ in range(limits["max_queued"]):
                queued.append(
                    submitters.submit(
                        manager.submit,
                        kind="analysis",
                        owner="operator",
                        request_id=None,
                        task=task,
                    )
                )
            deadline = time.monotonic() + 20.0
            while manager.status().admission["queued"] != limits["max_queued"]:
                assert time.monotonic() < deadline, "Concurrent submissions did not fill the queue"
                time.sleep(0.01)
            before = manager.status().admission
            with TestClient(app, base_url="http://127.0.0.1") as client:
                response = client.post("/api/models/scan/jobs")
            assert response.status_code == 422
            assert response.json()["detail"] == (
                f"Studio job queue is full: {before['running']} running, "
                f"{before['queued']} queued, limit {before['max_queued']}."
            )
            assert str(tmp_path) not in response.text
            assert len(manager.list_records()) == len(admitted)
            assert manager.status().admission["refused"] == before["refused"] + 1
        finally:
            release.set()
            admitted.extend(pending.result(timeout=30.0) for pending in queued)
            for record in admitted:
                assert manager.wait(record.job_id, 30.0).status == "completed"
            manager._ledger.close()


@pytest.mark.parametrize("kind", ["analysis", "training"])
def test_filesystem_failure_is_private_across_http_and_restart(tmp_path: Path, kind: str) -> None:
    """A real ENAMETOOLONG survives in diagnostic custody, never job inspection."""
    root = tmp_path / "private-ledger"
    app = create_app(StudioRuntimeSettings(job_root_path=str(root)))
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        """Invoke the real production artifact writer with an invalid filename."""
        try:
            context.write_artifact("reports/" + "x" * 300, b"data")
        except OSError as error:
            if kind == "training":
                context.append_artifact_event(
                    "training/events.jsonl",
                    {
                        "event": "error",
                        "data": {
                            "message": {"diagnostic": str(error)},
                            "failure_schema": "studio.worker.failure.v1",
                            "refusal_code": str(error),
                        },
                    },
                )
                context.append_artifact_event(
                    "training/events.jsonl", {"event": "error", "data": {"message": str(error)}}
                )
            raise
        return {}

    job = manager.submit(kind=kind, owner="operator", request_id=None, task=task)
    record = manager.wait(job.job_id, 5.0)
    assert record.status == "failed"
    assert record.error is not None and str(root) in record.error
    assert record.public_error == GENERIC_JOB_FAILURE
    with TestClient(app, base_url="http://127.0.0.1") as client:
        detail = client.get("/api/studio/jobs/" + job.job_id)
        listing = client.get("/api/studio/jobs")
        if kind == "training":
            training_status = client.get("/api/training/status/" + job.job_id)
            training_stream = client.get("/api/training/stream/" + job.job_id)
            assert training_status.status_code == training_stream.status_code == 200
            assert str(root) not in training_status.text + training_stream.text
            assert training_status.json()["error"] == GENERIC_JOB_FAILURE
            assert GENERIC_JOB_FAILURE in training_stream.text
            retained, _ = manager.read_live_artifact_bytes(
                job.job_id, "training/events.jsonl", offset=0
            )
            assert str(root).encode() in retained
    assert detail.status_code == listing.status_code == 200
    assert detail.json()["error"] == listing.json()["jobs"][0]["error"] == GENERIC_JOB_FAILURE
    assert str(root) not in detail.text + listing.text
    reopened = StudioJobManager(
        root=root, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=5.0
    )
    restored = reopened.record(job.job_id)
    assert restored.error == record.error
    assert restored.to_public_dict()["error"] == GENERIC_JOB_FAILURE
    assert reopened.status().active_count == 0
    manager._ledger.close()
    reopened._ledger.close()


def test_authored_artifact_limit_survives_job_and_http(tmp_path: Path) -> None:
    """The actual artifact byte limit remains useful through durable and HTTP state."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path), job_max_artifact_bytes=4))
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        """Ask the production writer to exceed its configured byte bound."""
        context.write_artifact("report.txt", b"12345")
        return {}

    job = manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
    record = manager.wait(job.job_id, 5.0)
    message = "Studio job artifact exceeds configured size limit."
    assert record.status == "failed"
    assert record.error == record.public_error == message
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/studio/jobs/" + job.job_id)
    assert response.status_code == 200 and response.json()["error"] == message
    manager._ledger.close()


def test_null_seed_is_an_authored_refusal_before_worker_start(tmp_path: Path) -> None:
    """Real null-byte admission no longer exposes Python's generated path message."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=5.0
    )
    with pytest.raises(StudioJobRefused) as caught:
        manager.submit_process_task(
            kind="analysis",
            owner="operator",
            request_id=None,
            task_path="sc_neurocore.studio.api.analysis_jobs:execute_analysis_process_task",
            payload={},
            seed_inputs={"bad\0input": b"data"},
        )
    assert str(caught.value) == "Studio job seed-input path escapes the seed directory."
    assert "embedded null" not in str(caught.value)
    assert manager.status().active_count == 0
    assert len(manager.list_records()) == 1 and manager.list_records()[0].status == "failed"
    manager._ledger.close()


@pytest.mark.parametrize("execution", ["blocking", "legacy-thread"])
def test_training_producer_keeps_real_runtime_fault_private(tmp_path: Path, execution: str) -> None:
    """Actual training startup exposes a safe event and retains its failure locally."""
    context = StudioJobContext(
        job_id="sj_training_runtime_fault",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=100_000,
    )

    def persist(event: dict[str, object]) -> None:
        """Exercise a real filesystem failure when the trainer emits its config."""
        if event["event"] == "config":
            context.write_artifact("reports/" + "x" * 300, b"data")
        else:
            context.append_artifact_event("training/events.jsonl", event)

    job = TrainingJob(
        {"dataset": "synthetic", "epochs": 1, "batch_size": 1024, "hidden": [4], "timesteps": 1},
        job_id=context.job_id,
        event_sink=persist,
    )
    if execution == "blocking":
        with pytest.raises((OSError, RuntimeError)) as caught:
            job.run_blocking(context)
        assert job.error == str(caught.value)
        if isinstance(caught.value, OSError):
            assert caught.value.errno == 36 and str(tmp_path) in str(caught.value)
        else:
            assert str(caught.value).startswith("PyTorch not installed.")
        for name in ("training/status.json", "training/evidence.json"):
            payload = (tmp_path / name).read_text()
            assert GENERIC_JOB_FAILURE in payload and str(tmp_path) not in payload
    else:
        job.start()
        deadline = time.monotonic() + 20.0
        while job.status != "failed" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "failed"
        assert job.error is not None
        assert str(tmp_path) in job.error or job.error.startswith("PyTorch not installed.")
    assert job.status == "failed"
    events = (tmp_path / "training/events.jsonl").read_text()
    assert GENERIC_JOB_FAILURE in events and str(tmp_path) not in events
    assert "refusal_code" in events


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_thread_result_keeps_http_record_readable(tmp_path: Path, value: float) -> None:
    """A real thread result cannot commit invalid JSON and break the HTTP reader."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path)))
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        """Return the actual invalid result through the public thread task interface."""
        context.check_cancelled()
        return {"observation": value}

    try:
        submitted = manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
        record = manager.wait(submitted.job_id, 5.0)
        with TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False) as client:
            response = client.get("/api/studio/jobs/" + submitted.job_id)
        assert response.status_code == 200
        assert record.status == "failed" and record.public_error == GENERIC_JOB_FAILURE
        assert response.json()["error"] == GENERIC_JOB_FAILURE
        assert manager.status().active_count == 0
    finally:
        manager._ledger.close()


@pytest.mark.parametrize(
    "operation",
    [
        "seed-size",
        "control-seed-size",
        "control-invalid-json",
        "control-invalid-utf8",
        "control-nonobject",
        "control-nonfinite",
        "control-overflow",
        "event-nonfinite",
    ],
)
def test_context_refusal_reaches_http_without_consuming_invalid_input(
    tmp_path: Path, operation: str
) -> None:
    """Actual thread tasks reject retained invalid seeds, commands and events."""
    limit = 128 if operation == "event-nonfinite" else 4
    app = create_app(
        StudioRuntimeSettings(job_root_path=str(tmp_path), job_max_artifact_bytes=limit)
    )
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        """Exercise the owning context reader or writer with real invalid bytes."""
        if operation == "event-nonfinite":
            context.append_artifact_event("events.jsonl", {"value": float("nan")})
        elif operation in {"seed-size", "control-seed-size"}:
            directory = ".studio_seed" if operation == "seed-size" else ".studio_control_seed"
            seed = tmp_path / context.job_id / directory / "input.bin"
            seed.parent.mkdir()
            seed.write_bytes(b"12345")
            if operation == "seed-size":
                context.read_seed_input("input.bin")
            else:
                context.read_control_seed("input.bin")
        else:
            raw = {
                "control-invalid-json": b"[",
                "control-invalid-utf8": b"\xff",
                "control-nonobject": b"[]",
                "control-nonfinite": b'{"value":NaN}',
                "control-overflow": b'{"value":1e1000}',
            }[operation]
            command = tmp_path / context.job_id / ".studio_control" / "command.json"
            command.parent.mkdir()
            command.write_bytes(raw)
            context.poll_control_command()
        return {"consumed_invalid_input": True}

    expected = {
        "seed-size": "Studio job seed input exceeds configured size limit.",
        "control-seed-size": "Studio job control seed exceeds configured size limit.",
        "control-nonobject": "Studio job control command must be a JSON object.",
        "event-nonfinite": "Studio job event payload must be JSON.",
    }.get(operation, "Studio job control command is not valid JSON.")
    try:
        submitted = manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
        record = manager.wait(submitted.job_id, 5.0)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.get("/api/studio/jobs/" + submitted.job_id)
        assert record.status == "failed" and record.public_error == expected
        assert response.status_code == 200 and response.json()["error"] == expected
        assert response.json()["result"] is None
        assert manager.status().active_count == 0
        if operation == "event-nonfinite":
            assert not (tmp_path / submitted.job_id / "events.jsonl").exists()
    finally:
        manager._ledger.close()
