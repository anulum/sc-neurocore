# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


from sc_neurocore.studio.platform.jobs import (
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
)


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
