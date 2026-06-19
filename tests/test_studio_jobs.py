# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import time
import threading
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs import (
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
    assert str(tmp_path) not in str(completed.to_public_dict())
    assert (tmp_path / "jobs" / record.job_id / "reports" / "result.txt").read_bytes() == b"ok"


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
        "allowed_kinds": ["compiler", "synthesis", "training"],
        "completed_count": 0,
        "configured": True,
        "failed_count": 0,
        "schema_version": "studio.jobs.status.v1",
        "timed_out_count": 0,
    }
    assert str(tmp_path) not in response.text
