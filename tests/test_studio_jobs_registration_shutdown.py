# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Blocked registration and supervisor shutdown

"""A blocked authority write must not prevent the supervisor from terminating compute.

A second, real SQLite connection takes the ledger's write lock as soon as the
job is marked running, before the worker is spawned and registered, as a long
authority transaction does. An attempt in which registration committed first
is detected and repeated. A registration thread that cannot be started needs
thread exhaustion under a dedicated identity; the isolated proof covers it.
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import threading
import time

from sc_neurocore.studio.platform.jobs import StudioJobManager

ATTEMPTS = 10


def _await_running(observer: sqlite3.Connection, job_id: str) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        row = observer.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is not None and row[0] == "running":
            return
    raise AssertionError("job never reached running")


def test_busy_registration_does_not_block_worker_timeout(tmp_path: Path) -> None:
    """Real SQLite contention outlives compute; late registration cannot grant a dead worker."""
    module = tmp_path / "blocked_registration_task.py"
    module.write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.imported').write_text('unexpected import')\n"
        "def run(context,payload): return {}\n"
    )
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), previous)))
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=0.5,
    )
    observer = sqlite3.connect(manager.ledger_path, isolation_level=None, check_same_thread=False)
    try:
        for _ in range(ATTEMPTS):
            job = manager.submit_process_task(
                kind="analysis",
                owner="owner",
                request_id=None,
                task_path="blocked_registration_task:run",
                payload={},
            )
            _await_running(observer, job.job_id)
            observer.execute("BEGIN IMMEDIATE")
            workers = "SELECT COUNT(*) FROM job_workers WHERE job_id = ?"
            if observer.execute(workers, (job.job_id,)).fetchone()[0]:
                observer.execute("ROLLBACK")
                manager.wait(job.job_id, 10.0)
                module.with_suffix(".imported").unlink(missing_ok=True)
                continue
            time.sleep(1.5)
            assert not module.with_suffix(".imported").exists()
            registrations = [
                thread
                for thread in threading.enumerate()
                if thread.name == f"studio-register-{job.job_id}"
            ]
            assert len(registrations) == 1 and registrations[0].is_alive()
            observer.execute("ROLLBACK")
            registrations[0].join(timeout=10.0)
            assert not registrations[0].is_alive()
            assert manager.wait(job.job_id, 10.0).status == "timed_out"
            assert manager._done_events[job.job_id].wait(5.0)
            assert observer.execute(workers, (job.job_id,)).fetchone()[0] == 0
            assert not module.with_suffix(".imported").exists()
            assert manager.status().admission["running"] == 0
            assert manager.status().unreaped_workers == ()
            return
        raise AssertionError("registration committed before every contention attempt")
    finally:
        if observer.in_transaction:
            observer.execute("ROLLBACK")
        observer.close()
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous
