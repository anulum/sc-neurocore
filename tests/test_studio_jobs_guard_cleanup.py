# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Guard start-up failure cleanup

"""A guard that never becomes ready stops task start-up and leaves nothing behind.

The guard is the real module. It is stopped before its ready token, as a guard
that cannot be scheduled is, or killed before it, as by the out-of-memory
killer. A watcher thread signals it as soon as it appears; an attempt in
which the guard became ready first is detected and repeated.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment
from sc_neurocore.studio.platform.jobs_process_state import group_survivors

GUARD_MODULE = "sc_neurocore.studio.platform.jobs_worker_guard"
ATTEMPTS = 20


def _children(pid: int) -> list[int]:
    found: list[int] = []
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except FileNotFoundError:
        return found
    for task in tasks:
        try:
            with open(f"/proc/{pid}/task/{task}/children", encoding="ascii") as handle:
                found.extend(int(item) for item in handle.read().split())
        except FileNotFoundError:
            continue
    return found


def _runs(pid: int, module: str) -> bool:
    """Return whether ``pid`` has executed ``python -m module``.

    A forked child shows its parent's command line until it executes, so only
    the exact ``-m`` argument pair identifies the new interpreter.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return b"\0-m\0" + module.encode() + b"\0" in handle.read()
    except (FileNotFoundError, ProcessLookupError):
        return False


class GuardInterrupter(threading.Thread):
    """Signal the first guard started under ``parent()`` the moment it appears."""

    def __init__(self, parent: Callable[[], int | None], number: signal.Signals) -> None:
        super().__init__(daemon=True)
        self._parent = parent
        self._number = number
        self._halt = threading.Event()
        self.guard: int | None = None

    def run(self) -> None:
        deadline = time.monotonic() + 30.0
        while not self._halt.is_set() and time.monotonic() < deadline:
            owner = self._parent()
            for child in [] if owner is None else _children(owner):
                if _runs(child, GUARD_MODULE):
                    os.kill(child, self._number)
                    self.guard = child
                    return

    def finish(self) -> None:
        self._halt.set()
        self.join(timeout=10.0)


def _guard_gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            return handle.read().rsplit(")", 1)[-1].split()[0] in {"Z", "X"}
    except FileNotFoundError:
        return True


@pytest.mark.parametrize(
    "number,message",
    [
        (signal.SIGSTOP, "Worker lifetime guard readiness timed out."),
        (signal.SIGKILL, "Worker lifetime guard failed to arm."),
    ],
)
def test_worker_refuses_to_start_without_a_ready_guard(
    number: signal.Signals, message: str
) -> None:
    """The worker raises and collects its guard; it never reports itself armed."""
    supervisor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    program = (
        "import sys\n"
        f"from {GUARD_MODULE} import arm_worker_guard\n"
        "try:\n"
        "    arm_worker_guard(sys.argv[1])\n"
        "    print('armed', flush=True)\n"
        "except RuntimeError as refused:\n"
        "    print(refused, flush=True)\n"
    )
    try:
        for _ in range(ATTEMPTS):
            worker = subprocess.Popen(
                [sys.executable, "-c", program, supervisor_identity(supervisor.pid)],
                env=_process_worker_environment(),
                start_new_session=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            interrupter = GuardInterrupter(lambda: worker.pid, number)
            interrupter.start()
            output, _ = worker.communicate(timeout=60.0)
            interrupter.finish()
            with suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGKILL)
            if output.strip() == "armed":
                continue
            assert output.strip() == message
            assert worker.returncode == 0
            assert interrupter.guard is not None and _guard_gone(interrupter.guard)
            return
        pytest.fail("every guard became ready before it could be interrupted")
    finally:
        supervisor.kill()
        supervisor.wait(timeout=10.0)


def test_supervisor_reaps_after_guard_start_failure(tmp_path: Path) -> None:
    """A stalled guard fails the job before task import and frees its capacity."""
    module = tmp_path / "guard_cleanup_probe.py"
    module.write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.imported').write_text('imported')\n"
        "def run(context, payload):\n"
        "    return {'unexpected_execution': True}\n"
    )
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), previous)))
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=30.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )
    try:
        for _ in range(ATTEMPTS):
            workers: list[int] = []

            def worker() -> int | None:
                if not workers:
                    workers.extend(
                        pid
                        for pid in _children(os.getpid())
                        if _runs(pid, "sc_neurocore.studio.platform.process_worker")
                    )
                return workers[0] if workers else None

            interrupter = GuardInterrupter(worker, signal.SIGSTOP)
            interrupter.start()
            job = manager.submit_process_task(
                kind="analysis",
                owner="owner",
                request_id=None,
                task_path="guard_cleanup_probe:run",
                payload={},
            )
            record = manager.wait(job.job_id, 60.0)
            interrupter.finish()
            if record.status == "completed":
                module.with_suffix(".imported").unlink()
                continue
            assert record.status == "failed"
            assert record.error == "RuntimeError"
            assert not module.with_suffix(".imported").exists()
            assert interrupter.guard is not None and _guard_gone(interrupter.guard)
            assert not group_survivors(workers[0])
            assert manager.status().unreaped_workers == ()
            retry = manager.submit_process_task(
                kind="analysis",
                owner="owner",
                request_id=None,
                task_path="tests.studio_job_tasks:process_echo_task",
                payload={"retry": True},
            )
            assert manager.wait(retry.job_id, 60.0).status == "completed"
            return
        pytest.fail("every guard became ready before it could be interrupted")
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous
