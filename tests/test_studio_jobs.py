# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import cast

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import tests.studio_job_tasks as studio_job_tasks
from starlette.testclient import TestClient

import sc_neurocore.studio.platform.jobs as jobs_module
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings, process_worker
from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifactUnavailable,
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
)


def test_studio_job_manager_completes_job_with_path_free_artifact_manifest(
    tmp_path: Path,
) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"ok")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    assert completed.status == "completed"
    assert completed.result == {"ok": True}
    assert len(completed.artifacts) == 1
    assert completed.artifacts[0].relative_path == "reports/result.txt"
    assert manager.list_snapshot().to_public_dict()["schema_version"] == "studio.jobs.list.v1"
    assert str(tmp_path) not in str(completed.to_public_dict())
    assert (tmp_path / "jobs" / record.job_id / "reports" / "result.txt").read_bytes() == b"ok"


def test_studio_job_manager_reads_manifest_declared_artifacts(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", "artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)

    payload = manager.read_artifact(record.job_id, "reports/result.txt")

    assert payload.payload == b"artifact body"
    assert payload.artifact.relative_path == "reports/result.txt"
    assert payload.artifact.sha256 == hashlib.sha256(b"artifact body").hexdigest()


def test_studio_job_manager_purges_terminal_record_and_directory(tmp_path: Path) -> None:
    """Terminal job purges remove the record and confined work directory."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", "artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)
    job_dir = tmp_path / "jobs" / record.job_id

    purged = manager.purge_terminal_record(record.job_id)

    assert completed.status == "completed"
    assert purged.job_id == record.job_id
    assert manager.list_records() == ()
    assert not job_dir.exists()
    with pytest.raises(KeyError):
        manager.record(record.job_id)
    with pytest.raises(KeyError):
        manager.read_artifact(record.job_id, "reports/result.txt")


def test_studio_job_manager_rejects_active_record_purge(tmp_path: Path) -> None:
    """Active jobs cannot be purged while their worker may still write files."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=2.0,
    )
    release = threading.Event()

    def task(context: StudioJobContext) -> dict[str, object]:
        release.wait(timeout=1.0)
        context.write_artifact("reports/result.txt", "artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="training",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )

    with pytest.raises(StudioJobRejected, match="active"):
        manager.purge_terminal_record(record.job_id)

    release.set()
    completed = manager.wait(record.job_id, timeout_seconds=3.0)

    assert completed.status == "completed"
    assert manager.record(record.job_id).job_id == record.job_id


def test_studio_job_manager_rejects_non_directory_purge_target(tmp_path: Path) -> None:
    """Job purges reject corrupted non-directory job targets."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(_context: StudioJobContext) -> dict[str, object]:
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)
    job_dir = tmp_path / "jobs" / record.job_id
    job_dir.rmdir()
    job_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(StudioJobRejected, match="not a directory"):
        manager.purge_terminal_record(record.job_id)

    assert completed.status == "completed"
    assert manager.record(record.job_id).job_id == record.job_id


def test_studio_job_context_appends_and_publishes_live_event_artifact(
    tmp_path: Path,
) -> None:
    """Live JSONL artifacts can be appended first and manifested once."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_live_events",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    context.append_artifact_event("events/live.jsonl", {"event": "epoch", "data": {"n": 1}})
    artifact = context.publish_existing_artifact("events/live.jsonl")

    payload = (work_dir / "events" / "live.jsonl").read_bytes()
    assert json.loads(payload.decode("utf-8")) == {"data": {"n": 1}, "event": "epoch"}
    assert artifact.relative_path == "events/live.jsonl"
    assert artifact.size_bytes == len(payload)
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert context.artifacts == (artifact,)


def test_studio_job_context_rejects_live_event_artifact_escape(tmp_path: Path) -> None:
    """Live JSONL artifact writes use the same confinement as normal artifacts."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_live_escape",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="escapes"):
        context.append_artifact_event("../escape.jsonl", {"event": "bad"})

    assert not (tmp_path / "escape.jsonl").exists()


def test_studio_job_context_rejects_invalid_live_event_payload(
    tmp_path: Path,
) -> None:
    """Live JSONL event writes reject non-JSON payloads."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_live_invalid_payload",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="payload must be JSON"):
        context.append_artifact_event("events/live.jsonl", {"bad": object()})


def test_studio_job_context_rejects_oversized_live_event_artifact(
    tmp_path: Path,
) -> None:
    """Live JSONL event writes enforce per-artifact byte ceilings."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_live_too_large",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=8,
    )

    with pytest.raises(ValueError, match="exceeds configured size"):
        context.append_artifact_event("events/live.jsonl", {"event": "epoch"})


def test_studio_job_context_rejects_missing_or_oversized_existing_artifact(
    tmp_path: Path,
) -> None:
    """Publishing existing artifacts validates availability and byte ceilings."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_existing_artifact",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4,
    )
    artifact_path = work_dir / "events" / "live.jsonl"
    artifact_path.parent.mkdir()
    artifact_path.write_text("too-large", encoding="utf-8")

    with pytest.raises(ValueError, match="unavailable"):
        context.publish_existing_artifact("events/missing.jsonl")
    with pytest.raises(ValueError, match="exceeds configured size"):
        context.publish_existing_artifact("events/live.jsonl")


def test_studio_job_manager_tails_live_artifact_before_manifest(
    tmp_path: Path,
) -> None:
    """Live artifact reads are path-confined and available before completion."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=1.0,
    )
    release = threading.Event()

    def task(context: StudioJobContext) -> dict[str, object]:
        context.append_artifact_event("training/events.jsonl", {"event": "epoch"})
        release.wait(timeout=1.0)
        return {"ok": True}

    record = manager.submit(
        kind="training",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )

    payload = b""
    offset = 0
    for _ in range(20):
        payload, offset = manager.read_live_artifact_bytes(
            record.job_id,
            "training/events.jsonl",
            offset=0,
        )
        if payload:
            break
        time.sleep(0.05)
    release.set()
    manager.wait(record.job_id, timeout_seconds=2.0)

    assert json.loads(payload.decode("utf-8")) == {"event": "epoch"}
    assert offset == len(payload)
    with pytest.raises(KeyError):
        manager.read_live_artifact_bytes(record.job_id, "../escape.jsonl", offset=0)


def test_studio_job_manager_rejects_invalid_live_artifact_read_bounds(
    tmp_path: Path,
) -> None:
    """Live artifact reads validate offsets and return empty missing tails."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=1.0,
    )

    def task(_context: StudioJobContext) -> dict[str, object]:
        return {"ok": True}

    record = manager.submit(
        kind="training",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)

    with pytest.raises(ValueError, match="offset"):
        manager.read_live_artifact_bytes(record.job_id, "events/missing.jsonl", offset=-1)
    with pytest.raises(ValueError, match="read size"):
        manager.read_live_artifact_bytes(
            record.job_id,
            "events/missing.jsonl",
            offset=0,
            max_bytes=0,
        )
    assert manager.read_live_artifact_bytes(
        record.job_id,
        "events/missing.jsonl",
        offset=7,
    ) == (b"", 7)


def test_studio_job_manager_rejects_tampered_manifest_artifact(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"original")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)
    (tmp_path / "jobs" / record.job_id / "reports" / "result.txt").write_bytes(b"tampered")

    with pytest.raises(StudioJobArtifactUnavailable, match="integrity"):
        manager.read_artifact(record.job_id, "reports/result.txt")


def test_studio_job_manager_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("../escape.txt", b"bad")
        return {"unreachable": True}

    record = manager.submit(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    assert completed.status == "failed"
    assert completed.error == "Studio job artifact path escapes the job directory."
    assert not (tmp_path / "escape.txt").exists()


def test_studio_job_context_rejects_symlink_artifact_escape(tmp_path: Path) -> None:
    work_dir = tmp_path / "job"
    outside_dir = tmp_path / "outside"
    work_dir.mkdir()
    outside_dir.mkdir()
    (work_dir / "linked").symlink_to(outside_dir, target_is_directory=True)
    context = StudioJobContext(
        job_id="sj_test",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=1024,
    )

    context.check_cancelled()
    with pytest.raises(ValueError, match="escapes"):
        context.write_artifact("linked/escape.txt", b"bad")


def test_studio_job_manager_rejects_disallowed_job_kind(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    with pytest.raises(StudioJobRejected, match="not allowed"):
        manager.submit(
            kind="training",
            owner="operator-1",
            request_id="req-1",
            task=lambda _context: {},
        )


def test_studio_job_manager_rejects_invalid_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allowed job kind"):
        StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset(),
            default_timeout_seconds=1.0,
        )
    with pytest.raises(ValueError, match="timeout"):
        StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset({"synthesis"}),
            default_timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="artifact size"):
        StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset({"synthesis"}),
            default_timeout_seconds=1.0,
            max_artifact_bytes=0,
        )


def test_studio_job_manager_rejects_oversized_artifacts(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
        max_artifact_bytes=4,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"too-large")
        return {"unreachable": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    assert completed.status == "failed"
    assert completed.error == "Studio job artifact exceeds configured size limit."
    assert completed.artifacts == ()
    assert not (tmp_path / "jobs" / record.job_id / "reports" / "result.txt").exists()


def test_studio_job_manager_rejects_non_positive_submit_timeout(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    with pytest.raises(StudioJobRejected, match="timeout"):
        manager.submit(
            kind="synthesis",
            owner="operator-1",
            request_id="req-1",
            task=lambda _context: {},
            timeout_seconds=0,
        )


def test_studio_job_manager_cancel_finished_job_is_noop(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )
    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=lambda _context: {"done": True},
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    cancelled = manager.cancel(record.job_id)

    assert completed.status == "completed"
    assert cancelled.status == "completed"
    assert cancelled.result == {"done": True}


def test_studio_job_manager_cancels_cooperative_job(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=2.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        while not context.cancelled:
            time.sleep(0.01)
        context.check_cancelled()
        return {"unreachable": True}

    record = manager.submit(
        kind="training",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )

    assert manager.cancel(record.job_id).status == "cancelling"
    completed = manager.wait(record.job_id, timeout_seconds=2.0)
    assert completed.status == "cancelled"


def test_studio_job_manager_times_out_cooperative_job(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=0.05,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        while not context.cancelled:
            time.sleep(0.01)
        context.check_cancelled()
        return {"unreachable": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    assert completed.status == "timed_out"
    assert completed.error == "Studio job exceeded its timeout."


def test_studio_job_manager_completes_process_task_with_manifest(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=15.0,
    )

    record = manager.submit_process_task(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task_path="tests.studio_job_tasks:process_echo_task",
        payload={"model": "lif"},
    )
    completed = manager.wait(record.job_id, timeout_seconds=20.0)

    assert completed.status == "completed"
    assert completed.execution_model == "process"
    assert completed.result == {"payload": {"model": "lif"}, "worker_job_id": record.job_id}
    assert completed.artifacts[0].relative_path == "reports/process-result.txt"
    artifact = manager.read_artifact(record.job_id, "reports/process-result.txt")
    assert artifact.payload == b"process ok"
    assert str(tmp_path) not in str(completed.to_public_dict())


def test_studio_process_worker_environment_prepends_source_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process workers can import the repo package without shell PYTHONPATH state."""

    monkeypatch.setenv("PYTHONPATH", "existing")

    environment = jobs_module._process_worker_environment()
    pythonpath = environment["PYTHONPATH"].split(os.pathsep)

    assert pythonpath[0].endswith("/src")
    # Repo root is the parent of the src path — assert structurally rather than by a
    # hard-coded directory name (the checkout dir differs by environment, e.g.
    # "sc-neurocore" in CI vs "SC-NEUROCORE" locally).
    assert pythonpath[1] == str(Path(pythonpath[0]).parent)
    assert "existing" in pythonpath


def test_studio_process_worker_sleep_task_uses_payload_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import-stable worker helper consumes numeric sleep payloads."""

    observed_sleep_seconds: list[float] = []
    context = StudioJobContext(
        job_id="sj_sleep",
        work_dir=tmp_path / "job",
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )
    monkeypatch.setattr("tests.studio_job_tasks.time.sleep", observed_sleep_seconds.append)

    result = studio_job_tasks.process_sleep_task(context, {"seconds": 0.25})

    assert observed_sleep_seconds == [0.25]
    assert result == {"slept": True}


def test_studio_process_worker_failure_task_raises_stable_error(tmp_path: Path) -> None:
    """Import-stable worker helper raises a redacted deterministic error."""

    context = StudioJobContext(
        job_id="sj_failure",
        work_dir=tmp_path / "job",
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="hidden local failure detail"):
        studio_job_tasks.process_failure_task(context, {})


def test_studio_job_manager_fails_process_task_without_error_detail(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=15.0,
    )

    record = manager.submit_process_task(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task_path="tests.studio_job_tasks:process_failure_task",
        payload={},
    )
    completed = manager.wait(record.job_id, timeout_seconds=20.0)

    assert completed.status == "failed"
    assert completed.execution_model == "process"
    assert completed.error == "ValueError"


def test_studio_job_status_counts_execution_models(tmp_path: Path) -> None:
    """Status snapshots expose thread/process coverage without local paths."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=15.0,
    )

    def thread_task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/thread-result.txt", "thread ok")
        return {"thread": True}

    thread_record = manager.submit(
        kind="compiler",
        owner="operator-1",
        request_id="req-thread",
        task=thread_task,
    )
    process_record = manager.submit_process_task(
        kind="compiler",
        owner="operator-1",
        request_id="req-process",
        task_path="tests.studio_job_tasks:process_echo_task",
        payload={"model": "lif"},
    )
    completed_thread = manager.wait(thread_record.job_id, timeout_seconds=2.0)
    completed_process = manager.wait(process_record.job_id, timeout_seconds=20.0)

    payload = manager.status().to_public_dict()

    assert completed_thread.status == "completed"
    assert completed_process.status == "completed"
    assert payload["completed_count"] == 2
    assert payload["process_count"] == 1
    assert payload["thread_count"] == 1
    assert str(tmp_path) not in str(payload)


def test_studio_job_manager_rejects_invalid_process_inputs(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=15.0,
    )

    with pytest.raises(StudioJobRejected, match="module:function"):
        manager.submit_process_task(
            kind="compiler",
            owner="operator-1",
            request_id="req-1",
            task_path="bad",
            payload={},
        )
    with pytest.raises(StudioJobRejected, match="module path"):
        manager.submit_process_task(
            kind="compiler",
            owner="operator-1",
            request_id="req-1",
            task_path="tests..bad:process_echo_task",
            payload={},
        )
    with pytest.raises(StudioJobRejected, match="function name"):
        manager.submit_process_task(
            kind="compiler",
            owner="operator-1",
            request_id="req-1",
            task_path="tests.studio_job_tasks:not-valid",
            payload={},
        )
    with pytest.raises(StudioJobRejected, match="JSON"):
        manager.submit_process_task(
            kind="compiler",
            owner="operator-1",
            request_id="req-1",
            task_path="tests.studio_job_tasks:process_echo_task",
            payload={"bad": object()},
        )
    with pytest.raises(StudioJobRejected, match="not allowed"):
        manager.submit_process_task(
            kind="training",
            owner="operator-1",
            request_id="req-1",
            task_path="tests.studio_job_tasks:process_echo_task",
            payload={},
        )
    with pytest.raises(StudioJobRejected, match="timeout"):
        manager.submit_process_task(
            kind="compiler",
            owner="operator-1",
            request_id="req-1",
            task_path="tests.studio_job_tasks:process_echo_task",
            payload={},
            timeout_seconds=0,
        )


def test_studio_job_manager_cancels_process_task(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=3.0,
    )
    record = manager.submit_process_task(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task_path="tests.studio_job_tasks:process_sleep_task",
        payload={"seconds": 3},
    )

    assert manager.cancel(record.job_id).status == "cancelling"
    completed = manager.wait(record.job_id, timeout_seconds=5.0)

    assert completed.status == "cancelled"


def test_studio_job_manager_times_out_process_task(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=0.05,
    )

    record = manager.submit_process_task(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task_path="tests.studio_job_tasks:process_sleep_task",
        payload={"seconds": 3},
    )
    completed = manager.wait(record.job_id, timeout_seconds=5.0)

    assert completed.status == "timed_out"
    assert completed.error == "Studio job exceeded its timeout."


def test_studio_job_manager_rejects_missing_or_unknown_artifacts(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=3.0,
    )
    record = manager.submit_process_task(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task_path="tests.studio_job_tasks:process_echo_task",
        payload={},
    )
    completed = manager.wait(record.job_id, timeout_seconds=20.0)

    assert completed.status == "completed"
    with pytest.raises(KeyError):
        manager.read_artifact(record.job_id, "../escape.txt")
    with pytest.raises(KeyError):
        manager.read_artifact(record.job_id, "reports/missing.txt")

    (tmp_path / "jobs" / record.job_id / "reports" / "process-result.txt").unlink()
    with pytest.raises(StudioJobArtifactUnavailable, match="unavailable"):
        manager.read_artifact(record.job_id, "reports/process-result.txt")


def test_studio_process_result_loader_handles_invalid_payloads(tmp_path: Path) -> None:
    missing = jobs_module._load_process_result(tmp_path / "missing.json")
    invalid_json_path = tmp_path / "invalid.json"
    invalid_json_path.write_text("{", encoding="utf-8")
    invalid_json = jobs_module._load_process_result(invalid_json_path)
    invalid_shape_path = tmp_path / "invalid-shape.json"
    invalid_shape_path.write_text("[]", encoding="utf-8")
    invalid_shape = jobs_module._load_process_result(invalid_shape_path)
    malformed_artifacts_path = tmp_path / "malformed-artifacts.json"
    malformed_artifacts_path.write_text(
        json.dumps({"artifacts": [{"relative_path": 1}], "status": "completed"}),
        encoding="utf-8",
    )
    malformed = jobs_module._load_process_result(malformed_artifacts_path)
    not_list_path = tmp_path / "not-list-artifacts.json"
    not_list_path.write_text(
        json.dumps({"artifacts": {}, "status": "completed"}),
        encoding="utf-8",
    )
    non_dict_path = tmp_path / "non-dict-artifact.json"
    non_dict_path.write_text(
        json.dumps({"artifacts": [1], "status": "completed"}),
        encoding="utf-8",
    )
    bad_size_path = tmp_path / "bad-size-artifact.json"
    bad_size_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "relative_path": "reports/result.txt",
                        "sha256": "0" * 64,
                        "size_bytes": "bad",
                    }
                ],
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    bad_hash_path = tmp_path / "bad-hash-artifact.json"
    bad_hash_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "relative_path": "reports/result.txt",
                        "sha256": 1,
                        "size_bytes": 1,
                    }
                ],
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )

    assert missing.error == "Studio process worker did not write a result."
    assert invalid_json.error == "Studio process worker wrote an invalid result."
    assert invalid_shape.error == "Studio process worker wrote an invalid result."
    assert malformed.artifacts == ()
    assert jobs_module._load_process_artifacts(malformed_artifacts_path) == ()
    assert jobs_module._load_process_result(not_list_path).artifacts == ()
    assert jobs_module._load_process_result(non_dict_path).artifacts == ()
    assert jobs_module._load_process_result(bad_size_path).artifacts == ()
    assert jobs_module._load_process_result(bad_hash_path).artifacts == ()
    assert jobs_module._load_process_artifacts(tmp_path / "missing.json") == ()


def test_studio_process_worker_environment_bootstraps_missing_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-worker environment sets PYTHONPATH when it is absent."""

    monkeypatch.delenv("PYTHONPATH", raising=False)

    environment = jobs_module._process_worker_environment()

    assert "PYTHONPATH" in environment
    assert str(Path(jobs_module.__file__).resolve().parents[3]) in environment["PYTHONPATH"]


def test_studio_process_terminate_falls_back_to_kill() -> None:
    class BlockingProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.wait_calls = 0

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="worker", timeout=1.0)
            return 0

        def kill(self) -> None:
            self.killed = True

    process = BlockingProcess()

    jobs_module._terminate_process(cast(subprocess.Popen[bytes], process))

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == 2


def test_studio_process_worker_main_writes_result_files(tmp_path: Path) -> None:
    work_dir = tmp_path / "sj_worker"
    work_dir.mkdir()
    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    payload_path.write_text(json.dumps({"model": "lif"}), encoding="utf-8")

    exit_code = process_worker.main(
        [
            "--task",
            "tests.studio_job_tasks:process_echo_task",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
            "--work-dir",
            str(work_dir),
            "--max-artifact-bytes",
            "1024",
        ]
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "completed"
    assert payload["result"] == {"payload": {"model": "lif"}, "worker_job_id": "sj_worker"}
    assert payload["artifacts"][0]["relative_path"] == "reports/process-result.txt"


def test_studio_process_worker_main_records_failure(tmp_path: Path) -> None:
    work_dir = tmp_path / "sj_worker"
    work_dir.mkdir()
    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    payload_path.write_text("[]", encoding="utf-8")

    exit_code = process_worker.main(
        [
            "--task",
            "tests.studio_job_tasks:process_echo_task",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
            "--work-dir",
            str(work_dir),
            "--max-artifact-bytes",
            "1024",
        ]
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "ValueError"


def test_studio_process_worker_main_rejects_non_callable_task(tmp_path: Path) -> None:
    work_dir = tmp_path / "sj_worker"
    work_dir.mkdir()
    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    payload_path.write_text("{}", encoding="utf-8")

    exit_code = process_worker.main(
        [
            "--task",
            "tests.studio_job_tasks:NON_CALLABLE_TASK",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
            "--work-dir",
            str(work_dir),
            "--max-artifact-bytes",
            "1024",
        ]
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "TypeError"


def test_studio_job_status_endpoint_is_path_free(tmp_path: Path) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=3.0,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/studio/jobs/status")

    assert response.status_code == 200
    assert response.json() == {
        "active_count": 0,
        "allowed_kinds": ["compiler", "evidence", "synthesis", "training"],
        "completed_count": 0,
        "configured": True,
        "failed_count": 0,
        "process_count": 0,
        "resource_profiles": [
            {
                "default_timeout_seconds": 3.0,
                "execution_models": ["thread", "process"],
                "kind": "compiler",
                "max_artifact_bytes": 16777216,
            },
            {
                "default_timeout_seconds": 3.0,
                "execution_models": ["thread", "process"],
                "kind": "evidence",
                "max_artifact_bytes": 16777216,
            },
            {
                "default_timeout_seconds": 3.0,
                "execution_models": ["thread", "process"],
                "kind": "synthesis",
                "max_artifact_bytes": 16777216,
            },
            {
                "default_timeout_seconds": 3.0,
                "execution_models": ["thread", "process"],
                "kind": "training",
                "max_artifact_bytes": 16777216,
            },
        ],
        "schema_version": "studio.jobs.status.v1",
        "thread_count": 0,
        "timed_out_count": 0,
    }
    assert str(tmp_path) not in response.text


def test_studio_job_artifact_endpoint_is_admin_gated_and_integrity_checked(
    tmp_path: Path,
) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=3.0,
        )
    )
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)
    client = TestClient(app, base_url="http://127.0.0.1")

    missing_principal = client.get(f"/api/studio/jobs/{record.job_id}/artifacts/reports/result.txt")
    allowed = client.get(
        f"/api/studio/jobs/{record.job_id}/artifacts/reports/result.txt",
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )

    assert missing_principal.status_code == 401
    assert missing_principal.json()["detail"] == "missing_principal"
    assert allowed.status_code == 200
    assert allowed.content == b"artifact body"
    assert allowed.headers["content-type"] == "application/octet-stream"
    assert allowed.headers["x-studio-artifact-size"] == str(len(b"artifact body"))
    assert (
        allowed.headers["x-studio-artifact-sha256"] == hashlib.sha256(b"artifact body").hexdigest()
    )
    assert str(tmp_path) not in str(allowed.headers)
    assert str(tmp_path) not in allowed.text


def test_studio_job_list_and_detail_endpoints_are_admin_gated_and_path_free(
    tmp_path: Path,
) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=3.0,
        )
    )
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)
    client = TestClient(app, base_url="http://127.0.0.1")
    admin_headers = {"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"}

    denied = client.get("/api/studio/jobs")
    listed = client.get("/api/studio/jobs", headers=admin_headers)
    detailed = client.get(f"/api/studio/jobs/{record.job_id}", headers=admin_headers)
    missing = client.get("/api/studio/jobs/sj_missing", headers=admin_headers)

    assert denied.status_code == 401
    assert listed.status_code == 200
    assert listed.json()["schema_version"] == "studio.jobs.list.v1"
    assert listed.json()["jobs"][0]["job_id"] == record.job_id
    assert listed.json()["jobs"][0]["execution_model"] == "thread"
    assert listed.json()["jobs"][0]["artifacts"][0]["relative_path"] == "reports/result.txt"
    assert detailed.status_code == 200
    assert detailed.json()["job_id"] == record.job_id
    assert detailed.json()["execution_model"] == "thread"
    assert detailed.json()["status"] == "completed"
    assert missing.status_code == 404
    assert missing.json()["detail"] == "job_not_found"
    assert str(tmp_path) not in listed.text
    assert str(tmp_path) not in detailed.text


def test_studio_job_artifact_endpoint_uses_generic_integrity_errors(tmp_path: Path) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=3.0,
        )
    )
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"original")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)
    (tmp_path / "jobs" / record.job_id / "reports" / "result.txt").write_bytes(b"tampered")
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        f"/api/studio/jobs/{record.job_id}/artifacts/reports/result.txt",
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "job_artifact_unavailable"
    assert str(tmp_path) not in response.text
