# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


from sc_neurocore.studio.platform.jobs import (
    STUDIO_CONTROL_COMMAND_FILE,
    STUDIO_CONTROL_DIR,
    STUDIO_CONTROL_SEED_DIR,
    StudioJobArtifactUnavailable,
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
)


def test_poll_control_command_consumes_pending_command_once(tmp_path: Path) -> None:
    """A pending control command is decoded and consumed exactly once."""

    work_dir = tmp_path / "ctrl-job"
    (work_dir / STUDIO_CONTROL_DIR).mkdir(parents=True)
    context = StudioJobContext(
        job_id="sj_ctrl",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )
    assert context.poll_control_command() is None

    (work_dir / STUDIO_CONTROL_DIR / STUDIO_CONTROL_COMMAND_FILE).write_text(
        json.dumps({"action": "attach_weights"}), encoding="utf-8"
    )

    assert context.poll_control_command() == {"action": "attach_weights"}
    assert context.poll_control_command() is None


def test_read_control_seed_reports_missing_payload(tmp_path: Path) -> None:
    """Reading an absent control seed fails closed."""

    context = StudioJobContext(
        job_id="sj_ctrl",
        work_dir=tmp_path / "ctrl-job",
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(StudioJobArtifactUnavailable, match="control seed is unavailable"):
        context.read_control_seed("model.bin")


def test_read_control_seed_rejects_escaping_path(tmp_path: Path) -> None:
    """Reading a control seed whose path escapes the directory fails closed."""

    work_dir = tmp_path / "ctrl-job"
    (work_dir / STUDIO_CONTROL_SEED_DIR).mkdir(parents=True)
    context = StudioJobContext(
        job_id="sj_ctrl",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="escapes the control-seed directory"):
        context.read_control_seed("../escape.bin")


def test_read_control_seed_rejects_symlinked_seed_directory(tmp_path: Path) -> None:
    """Control seed reads reject symlinked reserved seed directories."""

    work_dir = tmp_path / "ctrl-job"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    work_dir.mkdir()
    (work_dir / STUDIO_CONTROL_SEED_DIR).symlink_to(outside_dir, target_is_directory=True)
    (outside_dir / "model.bin").write_bytes(b"escape")
    context = StudioJobContext(
        job_id="sj_ctrl",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="escapes the control-seed directory"):
        context.read_control_seed("model.bin")


def test_send_control_command_rejects_terminal_job(tmp_path: Path) -> None:
    """Control commands are rejected for jobs that are not running."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=15.0,
    )
    record = manager.submit_process_task(
        kind="compiler",
        owner="operator",
        request_id="req-done",
        task_path="tests.studio_job_tasks:process_echo_task",
        payload={"model": "lif"},
    )
    completed = manager.wait(record.job_id, timeout_seconds=20.0)
    assert completed.status == "completed"

    with pytest.raises(StudioJobRejected, match="not running"):
        manager.send_control_command(record.job_id, command={"action": "attach_weights"})


def test_send_control_command_delivers_to_running_job(tmp_path: Path) -> None:
    """Control commands and seeds are delivered into a running job's sandbox."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=15.0,
    )
    record = manager.submit_process_task(
        kind="training",
        owner="studio-training",
        request_id="req-run",
        task_path="tests.studio_job_tasks:process_sleep_task",
        payload={"seconds": 3.0},
    )
    deadline = time.monotonic() + 5.0
    while manager.record(record.job_id).status != "running":
        if time.monotonic() >= deadline:
            pytest.fail("process job did not reach running state")
        time.sleep(0.02)

    manager.send_control_command(
        record.job_id,
        command={"action": "attach_weights", "architecture_fingerprint": "a" * 64},
        seed_inputs={"model_state.pt": b"seed weights"},
    )

    work_dir = tmp_path / "jobs" / record.job_id
    command_path = work_dir / STUDIO_CONTROL_DIR / STUDIO_CONTROL_COMMAND_FILE
    seed_path = work_dir / STUDIO_CONTROL_SEED_DIR / "model_state.pt"
    assert json.loads(command_path.read_text())["action"] == "attach_weights"
    assert seed_path.read_bytes() == b"seed weights"


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
