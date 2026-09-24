# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — launched-generation runs shared by the supervision tests

"""Real launcher, authority ledger, jobs and waits for generation supervision tests.

The fixtures start the real launcher service process and a real SQLite
authority ledger; ``runtime`` points the API at them through real sockets.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_finish_protocol import StorageFinishResponse
from sc_neurocore.studio.platform.storage_live_spool import LiveSpools
from sc_neurocore.studio.platform.storage_generation_exchanges import (
    GenerationJob,
    GenerationRuntime,
)
from sc_neurocore.studio.platform.storage_generation_supervisor import (
    supervise_generation,
)
from tests.studio_storage_generation_support import FRAME, Authority
from tests.studio_storage_launcher_support import (
    Launcher,
    launcher_base,
    shutdown,
    start,
)
from tests.studio_storage_supervision_support import JOB, Clock


SIMULATE: dict[str, object] = {
    "analysis": "simulate",
    "payload": {"equations": ["dv/dt = -v"], "duration": 1.0},
    "parameter_order": [],
}
LONG: dict[str, object] = {
    "analysis": "fi_curve",
    "payload": {
        "equations": ["dv/dt = (-v + I)/tau"],
        "params": {"tau": 10.0},
        "init": {"v": 0.0},
        # Far longer than any case waits: every run ends by stop, timeout or kill.
        "duration": 20000.0,
        "dt": 0.1,
        "i_steps": 100,
    },
    "parameter_order": ["tau"],
}
UNREAPED = "The worker processes were not confirmed stopped."


@pytest.fixture
def base() -> Iterator[Path]:
    """Short private base directory so every socket path fits the Unix limit."""
    with launcher_base() as path:
        yield path


@pytest.fixture
def launcher(base: Path) -> Iterator[Launcher]:
    """The real launcher service process."""
    running = start(base)
    try:
        yield running
    finally:
        shutdown(running)


@pytest.fixture
def ledger(base: Path) -> Iterator[StudioJobLedger]:
    """Storage authority ledger; the job is admitted to this API generation."""
    authority = StudioJobLedger(root=base / "authority", supervisor="storage:1:1", clock=Clock())
    (base / "authority").chmod(0o700)
    try:
        yield authority
    finally:
        authority.close()


@pytest.fixture
def authority(ledger: StudioJobLedger) -> Iterator[Authority]:
    served = Authority(ledger)
    yield served
    served.join()


def api_runtime(
    base: Path,
    authority: Authority,
    *,
    spool_root: Path | None = None,
    launcher_socket: Path | None = None,
    grant_timeout_seconds: float = 60.0,
    frame_max_bytes: int = FRAME,
) -> GenerationRuntime:
    return GenerationRuntime(
        workspace="default",
        spool_root=spool_root or base / "spool",
        storage_uid=os.getuid(),
        frame_max_bytes=frame_max_bytes,
        transfer_timeout_seconds=10.0,
        launcher_socket=launcher_socket or base / "sock" / "launcher.sock",
        launcher_uid=os.getuid(),
        worker_uid=os.getuid(),
        worker_gid=os.getgid(),
        max_artifact_bytes=1 << 20,
        artifact_total_bytes=1 << 20,
        artifact_entries=64,
        grant_timeout_seconds=grant_timeout_seconds,
        heartbeat_seconds=0.2,
        poll_seconds=0.05,
        attempts=3,
        connect=authority.connect,
        live=LiveSpools(retain=4, max_seed_bytes=1 << 20),
    )


def analysis_job(
    payload: object, *, timeout: float = 120.0, task: str = "analysis.run"
) -> GenerationJob:
    return GenerationJob(
        job_id=JOB,
        task_name=task,
        authorized_route="/api/analysis/jobs",
        payload=json.dumps(payload).encode(),
        seeds={},
        timeout_seconds=timeout,
    )


def reservations(ledger: StudioJobLedger) -> list[str]:
    rows = ledger.connection().execute("SELECT state FROM admission_reservations").fetchall()
    return [str(row["state"]) for row in rows]


def workers(ledger: StudioJobLedger) -> int:
    row = ledger.connection().execute("SELECT COUNT(*) FROM job_workers").fetchone()
    return int(row[0])


class Background:
    """Run one supervisor in a thread so the test can act while it supervises."""

    def __init__(self, runtime: GenerationRuntime, job: GenerationJob) -> None:
        self.cancel = threading.Event()
        self.result: list[StorageFinishResponse | BaseException] = []
        self._thread = threading.Thread(target=self._run, args=(runtime, job))
        self._thread.start()

    def _run(self, runtime: GenerationRuntime, job: GenerationJob) -> None:
        try:
            self.result.append(supervise_generation(runtime, job, cancel=self.cancel))
        except BaseException as exc:
            self.result.append(exc)

    def wait(self) -> StorageFinishResponse | BaseException:
        self._thread.join(timeout=120.0)
        assert not self._thread.is_alive()
        return self.result[0]


def await_running(ledger: StudioJobLedger) -> None:
    """Wait until the authority registered the launched worker."""
    deadline = time.monotonic() + 90.0
    while workers(ledger) == 0:
        assert time.monotonic() < deadline, "worker was never registered"
        time.sleep(0.05)
    ledger.close()


def sealed(response: object) -> None:
    assert isinstance(response, StorageFinishResponse)
    assert (response.reply, response.reason) == ("sealed", None)
