# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job admission control

"""How many jobs run at once, and what the overflow is told.

The failure these guard against is not a slow Studio: it is a Studio that
accepts everything, starts everything, and then cannot answer anything. A
refusal with a reason is a better answer than an admission that never
completes.

Start-up failures are real: a job root the process cannot write, and a kernel
that refuses new threads or program execution in a child interpreter whose
seccomp filter says so. A live supervisor whose own worker thread cannot be
created needs thread exhaustion under a dedicated identity; the isolated
proof covers it.
"""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from tests.studio_seccomp_support import (
    SECCOMP_AVAILABLE,
    THREAD_CLONE_FLAGS,
    Refusal,
    run_refused,
)

_needs_seccomp = pytest.mark.skipif(
    not SECCOMP_AVAILABLE, reason="seccomp filters need Linux x86_64"
)

_MANAGER = (
    "from pathlib import Path\n"
    "from sc_neurocore.studio.platform.jobs import StudioJobManager\n"
    "manager = StudioJobManager(root=Path(sys.argv[1]), allowed_kinds=frozenset({'analysis'}),\n"
    "    default_timeout_seconds=30.0, max_concurrent_jobs=1, max_queued_jobs=0)\n"
    "def submit(mode):\n"
    "    if mode == 'thread':\n"
    "        return manager.submit(kind='analysis', owner='operator', request_id=None,\n"
    "            task=lambda context: {})\n"
    "    return manager.submit_process_task(kind='analysis', owner='operator',\n"
    "        request_id=None, task_path='tests.studio_job_tasks:process_echo_task', payload={})\n"
)


def _manager(root: Path) -> StudioJobManager:
    return StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=30.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )


@_needs_seccomp
def test_refused_worker_execution_releases_capacity(tmp_path: Path) -> None:
    """A live supervisor records failure when the kernel refuses the worker program."""
    result = run_refused(
        _MANAGER + "install_refusals(REFUSALS)\n"
        "record = submit('process')\n"
        "outcome = manager.wait(record.job_id, 30.0)\n"
        "running = manager._admission.snapshot().running\n"
        "retry = manager.wait(submit('thread').job_id, 30.0).status\n"
        "print(json.dumps({'status': outcome.status, 'error': outcome.error,\n"
        "    'running': running, 'retry': retry}))\n",
        [Refusal("execve", errno.EACCES)],
        arguments=(str(tmp_path / "jobs"),),
    )
    assert result["status"] == "failed"
    assert isinstance(result["error"], str) and "Permission denied" in result["error"]
    assert (result["running"], result["retry"]) == (0, "completed")


@_needs_seccomp
@pytest.mark.parametrize("mode", ["thread", "process"])
def test_refused_supervisor_thread_returns_capacity(tmp_path: Path, mode: str) -> None:
    """A submission whose supervisor thread cannot exist fails and frees its slot."""
    result = run_refused(
        _MANAGER + "install_refusals(REFUSALS)\n"
        "try:\n"
        "    submit(sys.argv[2])\n"
        "    raised = None\n"
        "except RuntimeError as refused:\n"
        "    raised = str(refused)\n"
        "records = [(r.status, r.error) for r in manager.list_records()]\n"
        "print(json.dumps({'raised': raised, 'records': records,\n"
        "    'running': manager._admission.snapshot().running}))\n",
        [
            Refusal("clone3", errno.ENOSYS),
            Refusal("clone", errno.EAGAIN, THREAD_CLONE_FLAGS),
        ],
        arguments=(str(tmp_path / "jobs"), mode),
    )
    assert result == {
        "raised": "can't start new thread",
        "records": [["failed", "Studio job could not start: can't start new thread"]],
        "running": 0,
    }


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_unwritable_job_root_fails_the_start_and_returns_capacity(
    tmp_path: Path, mode: str
) -> None:
    """A job directory that cannot be created leaves a failed record, not an occupied slot."""
    manager = _manager(tmp_path / "jobs")
    manager.root.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            if mode == "thread":
                manager.submit(
                    kind="analysis", owner="operator", request_id=None, task=lambda ctx: {}
                )
            else:
                manager.submit_process_task(
                    kind="analysis",
                    owner="operator",
                    request_id=None,
                    task_path="tests.studio_job_tasks:process_echo_task",
                    payload={},
                )
    finally:
        manager.root.chmod(0o700)
    records = manager.list_records()
    assert len(records) == 1 and records[0].status == "failed"
    assert records[0].error is not None and "Permission denied" in records[0].error
    assert manager._admission.snapshot().running == 0
    healthy = manager.submit(
        kind="analysis", owner="operator", request_id=None, task=lambda ctx: {}
    )
    assert manager.wait(healthy.job_id, 10.0).status == "completed"
