# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import tests.studio_job_tasks as studio_job_tasks

import sc_neurocore.studio.platform.jobs as jobs_module
from sc_neurocore.studio.platform.jobs import (
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
)
from sc_neurocore.studio.training_contract import resolve_training_config


def test_training_snapshot_must_match_worker_payload_before_admission(tmp_path: Path) -> None:
    """A durable config cannot attest to different worker input."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=5.0,
    )
    config = resolve_training_config({"epochs": 1}).to_public_dict()
    other = resolve_training_config({"epochs": 2}).to_public_dict()

    with pytest.raises(StudioJobRejected, match="does not match"):
        manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id=None,
            task_path="tests.studio_job_tasks:process_echo_task",
            payload=other,
            training_config=config,
        )

    assert manager.list_records() == ()


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


def test_studio_process_worker_sleep_task_uses_payload_seconds(tmp_path: Path) -> None:
    """Import-stable worker helper sleeps for the numeric payload seconds."""

    context = StudioJobContext(
        job_id="sj_sleep",
        work_dir=tmp_path / "job",
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )
    started = time.monotonic()
    result = studio_job_tasks.process_sleep_task(context, {"seconds": 0.25})

    assert time.monotonic() - started >= 0.25
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
