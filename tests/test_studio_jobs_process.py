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
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import tests.studio_job_tasks as studio_job_tasks

import sc_neurocore.studio.platform.jobs as jobs_module
from sc_neurocore.studio.platform.jobs import (
    STUDIO_SEED_INPUT_DIR,
    StudioJobArtifactUnavailable,
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
)


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


def test_submit_process_task_delivers_confined_seed_inputs(tmp_path: Path) -> None:
    """Process workers read confined seed inputs written at submission time."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=15.0,
    )

    record = manager.submit_process_task(
        kind="training",
        owner="studio-training-attach",
        request_id="req-seed",
        task_path="tests.studio_job_tasks:process_seed_echo_task",
        payload={"seed_path": "weights/model.bin"},
        seed_inputs={"weights/model.bin": b"seed payload bytes"},
    )
    completed = manager.wait(record.job_id, timeout_seconds=20.0)

    assert completed.status == "completed"
    assert completed.result == {"seed_text": "seed payload bytes", "seed_bytes": 18}
    # Seed inputs are job inputs, not outputs, so they are never published.
    assert completed.artifacts == ()
    assert str(tmp_path) not in str(completed.to_public_dict())


def test_submit_process_task_rejects_oversized_seed_input(tmp_path: Path) -> None:
    """Seed inputs larger than the artifact ceiling are rejected at submission."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=5.0,
        max_artifact_bytes=8,
    )

    with pytest.raises(StudioJobRejected, match="seed input exceeds"):
        manager.submit_process_task(
            kind="training",
            owner="studio-training-attach",
            request_id="req-big",
            task_path="tests.studio_job_tasks:process_seed_echo_task",
            payload={"seed_path": "weights/model.bin"},
            seed_inputs={"weights/model.bin": b"this payload is too large"},
        )


def test_submit_process_task_rejects_escaping_seed_path(tmp_path: Path) -> None:
    """Seed inputs whose path escapes the seed directory are rejected."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=5.0,
    )

    with pytest.raises(StudioJobRejected, match="escapes the seed directory"):
        manager.submit_process_task(
            kind="training",
            owner="studio-training-attach",
            request_id="req-escape",
            task_path="tests.studio_job_tasks:process_seed_echo_task",
            payload={"seed_path": "model.bin"},
            seed_inputs={"../escape.bin": b"escape"},
        )


def test_read_seed_input_reports_missing_payload(tmp_path: Path) -> None:
    """Reading an absent seed input fails closed."""

    context = StudioJobContext(
        job_id="sj_seed",
        work_dir=tmp_path / "seed-job",
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(StudioJobArtifactUnavailable, match="seed input is unavailable"):
        context.read_seed_input("model.bin")


def test_read_seed_input_rejects_escaping_path(tmp_path: Path) -> None:
    """Reading a seed input whose path escapes the seed directory fails closed."""

    work_dir = tmp_path / "seed-job"
    (work_dir / STUDIO_SEED_INPUT_DIR).mkdir(parents=True)
    context = StudioJobContext(
        job_id="sj_seed",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="escapes the seed directory"):
        context.read_seed_input("../escape.bin")


def test_read_seed_input_rejects_symlinked_seed_directory(tmp_path: Path) -> None:
    """Seed reads reject reserved seed directories that resolve outside the job."""

    work_dir = tmp_path / "seed-job"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    work_dir.mkdir()
    (work_dir / STUDIO_SEED_INPUT_DIR).symlink_to(outside_dir, target_is_directory=True)
    (outside_dir / "model.bin").write_bytes(b"escape")
    context = StudioJobContext(
        job_id="sj_seed",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="escapes the seed directory"):
        context.read_seed_input("model.bin")


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
