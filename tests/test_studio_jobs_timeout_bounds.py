# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio execution timeout boundaries

"""Execution deadlines must be finite before configuration or admission has effects."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager, StudioJobRejected
from sc_neurocore.studio.platform.settings import (
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)

INVALID_TIMEOUTS = [float("nan"), float("inf"), float("-inf"), 0.0, -1.0]


@pytest.mark.parametrize("timeout", INVALID_TIMEOUTS)
@pytest.mark.parametrize("surface", ["settings", "environment", "manager"])
def test_invalid_default_timeout_has_no_storage_effect(
    tmp_path: Path, timeout: float, surface: str
) -> None:
    """Reject invalid defaults without creating a root or opening a ledger."""
    root = tmp_path / "jobs"
    with pytest.raises(ValueError, match="timeout must"):
        if surface == "settings":
            StudioRuntimeSettings(job_root_path=str(root), job_default_timeout_seconds=timeout)
        elif surface == "environment":
            build_default_studio_runtime_settings(
                {
                    "SC_NEUROCORE_STUDIO_JOB_ROOT": str(root),
                    "SC_NEUROCORE_STUDIO_JOB_TIMEOUT_SECONDS": str(timeout),
                }
            )
        else:
            StudioJobManager(
                root=root,
                allowed_kinds=frozenset({"analysis"}),
                default_timeout_seconds=timeout,
            )
    assert not root.exists()


@pytest.mark.parametrize("timeout", INVALID_TIMEOUTS)
@pytest.mark.parametrize("mode", ["thread", "process"])
def test_invalid_override_refuses_before_admission_and_start(
    tmp_path: Path, timeout: float, mode: str
) -> None:
    """No reservation, job row, directory or supervisor may survive a refused override."""
    root = tmp_path / "jobs"
    manager = StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=15.0,
    )
    before = manager._admission.snapshot()
    threads = set(threading.enumerate())
    try:
        with pytest.raises(StudioJobRejected, match="timeout must"):
            if mode == "thread":
                manager.submit(
                    kind="analysis",
                    owner="operator",
                    request_id=None,
                    task=lambda context: {"result": 42},
                    timeout_seconds=timeout,
                )
            else:
                manager.submit_process_task(
                    kind="analysis",
                    owner="operator",
                    request_id=None,
                    task_path="tests.studio_job_tasks:process_echo_task",
                    payload={"result": 42},
                    timeout_seconds=timeout,
                )
        assert manager.list_records() == ()
        assert manager._admission.snapshot() == before
        assert not tuple(root.glob("sj_*"))
        assert not manager._done_events and not manager._cancel_events
        # The refusal is synchronous: no supervisor or worker thread started.
        assert set(threading.enumerate()) == threads
    finally:
        manager._ledger.close()


@pytest.mark.parametrize("mode", ["thread", "process"])
@pytest.mark.parametrize("override", [None, 15.0])
def test_finite_timeout_executes_real_task(
    tmp_path: Path, mode: str, override: float | None
) -> None:
    """Defaults and finite overrides retain actual thread/process execution and results."""
    settings = build_default_studio_runtime_settings(
        {"SC_NEUROCORE_STUDIO_JOB_TIMEOUT_SECONDS": "15.0"}
    )
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=settings.job_default_timeout_seconds,
    )
    try:
        if mode == "thread":
            submitted = manager.submit(
                kind="analysis",
                owner="operator",
                request_id=None,
                task=lambda context: {"result": 42},
                timeout_seconds=override,
            )
            expected: dict[str, object] = {"result": 42}
        else:
            submitted = manager.submit_process_task(
                kind="analysis",
                owner="operator",
                request_id=None,
                task_path="tests.studio_job_tasks:process_echo_task",
                payload={"result": 42},
                timeout_seconds=override,
            )
            expected = {"payload": {"result": 42}, "worker_job_id": submitted.job_id}
        completed = manager.wait(submitted.job_id, 20.0)
        assert completed.status == "completed"
        assert completed.result == expected
        assert manager._done_events[submitted.job_id].wait(2.0)
        assert manager._admission.snapshot().running == 0
    finally:
        manager._ledger.close()
