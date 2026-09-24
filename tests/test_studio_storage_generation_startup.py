# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — launched generations that must not run their task

"""A worker that cannot or must not start never runs its task, and the job still ends.

Each case starts the real launcher service process, which spawns the real
bootstrap and worker running a reviewed named analysis task. The storage
authority is the real supervision and finish handlers over a real SQLite
ledger with delegated admission. Same-identity runs establish function only;
the three-identity proof belongs to the disposable container acceptance.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_process_state import process_exited
from sc_neurocore.studio.platform.storage_generation_supervisor import (
    supervise_generation,
)
from sc_neurocore.studio.platform.storage_launcher_protocol import LauncherResponse
from tests.studio_storage_generation_support import Authority, LauncherRelay
from tests.studio_storage_launcher_support import (
    Launcher,
    shutdown,
    start,
)
from tests.studio_storage_supervision_support import JOB, admit

from tests.studio_storage_generation_runs import *


def test_a_worker_that_exited_before_its_grant_fails_the_unstarted_job(
    base: Path, ledger: StudioJobLedger, authority: Authority
) -> None:
    """The launch reply is lost until the bootstrap has refused an unknown task."""
    running = start(base)
    relay = LauncherRelay(base / "sock" / "relay.sock", running.socket_path)

    def exited(response: LauncherResponse) -> None:
        assert response.pid is not None
        deadline = time.monotonic() + 60.0
        while not process_exited(response.pid):
            assert time.monotonic() < deadline
            time.sleep(0.02)

    try:
        admit(ledger, supervisor=supervisor_identity())
        relay.lose_reply["launch"] = 1
        relay.before_loss = exited
        runtime = api_runtime(base, authority, launcher_socket=relay.path)
        job = analysis_job(SIMULATE, task="analysis.unknown")
        sealed(supervise_generation(runtime, job, cancel=threading.Event()))
    finally:
        relay.close()
        shutdown(running)
    record = ledger.record(JOB)
    assert (record.status, record.error) == ("failed", "Studio process worker exited with 2.")
    assert (record.started_at_utc, workers(ledger), reservations(ledger)) == (None, 0, [])


def test_a_worker_that_never_asks_for_its_grant_fails(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority
) -> None:
    """The grant deadline ends a generation whose bootstrap refused before connecting."""
    admit(ledger, supervisor=supervisor_identity())
    runtime = api_runtime(base, authority, grant_timeout_seconds=3.0)
    job = analysis_job(SIMULATE, task="analysis.unknown")
    sealed(supervise_generation(runtime, job, cancel=threading.Event()))
    record = ledger.record(JOB)
    assert record.status == "failed"
    assert record.error is not None and record.error.startswith("Studio worker grant failed:")
    assert reservations(ledger) == []


@pytest.mark.parametrize(
    "before,reply,status",
    [
        ("cancelling", "sealed", "cancelled"),
        ("foreign", "refused", "pending"),
    ],
)
def test_a_worker_the_authority_does_not_start_never_gets_its_grant(
    base: Path,
    launcher: Launcher,
    ledger: StudioJobLedger,
    authority: Authority,
    before: str,
    reply: str,
    status: str,
) -> None:
    """A cancelled or foreign job registers no worker and runs no task."""
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        if before == "cancelling":
            admit(ledger, supervisor=supervisor_identity())
            ledger.transition(JOB, "cancelling")
        else:
            admit(ledger, supervisor=supervisor_identity(other.pid))
        response = supervise_generation(
            api_runtime(base, authority), analysis_job(SIMULATE), cancel=threading.Event()
        )
    finally:
        other.kill()
        other.wait(timeout=10.0)
    assert response.reply == reply
    record = ledger.record(JOB)
    assert (record.status, record.error) == (status, None)
    assert workers(ledger) == 0
    assert authority.seen[:2] == ["start", "finish"]


def test_unanswered_registration_fails_after_the_worker_is_stopped(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority
) -> None:
    """Every start reply lost: no grant; the registered worker is stopped, then finished."""
    admit(ledger, supervisor=supervisor_identity())
    authority.lose["start"] = 3
    sealed(
        supervise_generation(
            api_runtime(base, authority), analysis_job(SIMULATE), cancel=threading.Event()
        )
    )
    record = ledger.record(JOB)
    assert (record.status, record.error) == (
        "failed",
        "Studio worker could not start: registration unanswered.",
    )
    assert reservations(ledger) == []


def test_refused_launch_fails_the_unstarted_job(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A spool the launcher does not serve is refused; the job fails and frees capacity."""
    admit(ledger, supervisor=supervisor_identity())
    other = base / "other"
    other.mkdir(mode=0o750)
    runtime = api_runtime(base, authority, spool_root=other)
    sealed(supervise_generation(runtime, analysis_job(SIMULATE), cancel=threading.Event()))
    record = ledger.record(JOB)
    assert (record.status, record.error) == (
        "failed",
        "Studio worker could not start: launcher refused (spool).",
    )
    assert (record.started_at_utc, reservations(ledger)) == (None, [])


@pytest.mark.parametrize("fault", ["missing-spool", "long-endpoint"])
def test_staging_and_endpoint_failures_fail_before_any_launch(
    base: Path, ledger: StudioJobLedger, authority: Authority, fault: str
) -> None:
    """Nothing is launched when the spool or the grant endpoint cannot be prepared."""
    admit(ledger, supervisor=supervisor_identity())
    root = base / "absent"
    if fault == "long-endpoint":
        root = base / ("s" * 60)
        root.mkdir(mode=0o750)
    runtime = api_runtime(base, authority, spool_root=root)
    sealed(supervise_generation(runtime, analysis_job(SIMULATE), cancel=threading.Event()))
    record = ledger.record(JOB)
    assert record.status == "failed" and record.error is not None
    expected = "spool:" if fault == "missing-spool" else "grant endpoint:"
    assert record.error.startswith(f"Studio worker could not start: {expected}")
    assert reservations(ledger) == []
