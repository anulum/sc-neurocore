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
import re
import subprocess
import sys
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_process_state import process_exited
from sc_neurocore.studio.platform.storage_generation_supervisor import (
    GenerationSupervisor,
    supervise_generation,
)
from sc_neurocore.studio.platform.storage_launcher_protocol import LauncherResponse
from sc_neurocore.studio.platform.storage_worker_grant import (
    WorkerExitedBeforeGrant,
    WorkerGrantEndpoint,
)
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
        """Wait for the actual bootstrap to exit before dropping its reply."""
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
    assert record.public_error == "Studio process worker exited with 2."
    assert record.to_public_dict()["error"] == record.public_error
    assert (record.started_at_utc, workers(ledger), reservations(ledger)) == (None, 0, [])


def test_a_worker_that_exits_before_asking_for_its_grant_is_reported_as_such(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A bootstrap that refused before connecting ends the wait when the launcher reports it gone.

    The grant deadline is far longer than the run: had the supervisor waited for it,
    the job would carry the grant timeout, not the worker's exit.

    Parameters
    ----------
    base : Path
        The test's namespace.
    launcher : Launcher
        The real launcher service.
    ledger : StudioJobLedger
        The real ledger.
    authority : Authority
        The real storage authority handlers.
    """
    admit(ledger, supervisor=supervisor_identity())
    runtime = api_runtime(base, authority, grant_timeout_seconds=60.0)
    job = analysis_job(SIMULATE, task="analysis.unknown")
    sealed(supervise_generation(runtime, job, cancel=threading.Event()))
    record = ledger.record(JOB)
    expected = "Studio process worker exited with 2 before requesting its grant."
    assert (record.status, record.error, record.public_error) == ("failed", expected, expected)
    assert record.to_public_dict()["error"] == record.public_error
    assert (record.started_at_utc, workers(ledger), reservations(ledger)) == (None, 0, [])


def test_a_worker_that_asked_then_ended_is_reported_as_having_asked(
    base: Path,
    launcher: Launcher,
    ledger: StudioJobLedger,
    authority: Authority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that connected for its grant and ended before it was given is told apart.

    Parameters
    ----------
    base : Path
        The test's namespace.
    launcher : Launcher
        The real launcher service.
    ledger : StudioJobLedger
        The real ledger.
    authority : Authority
        The real storage authority handlers.
    monkeypatch : pytest.MonkeyPatch
        Makes the grant end as the endpoint ends it for a connection left in the backlog.
    """

    def asked_then_ended(self: object, *args: object, **kwargs: object) -> str:
        """End the grant as a worker that connected and then exited with 3 ends it."""
        raise WorkerExitedBeforeGrant(3, requested=True)

    monkeypatch.setattr(WorkerGrantEndpoint, "grant", asked_then_ended)
    admit(ledger, supervisor=supervisor_identity())
    runtime = api_runtime(base, authority)
    sealed(supervise_generation(runtime, analysis_job(SIMULATE), cancel=threading.Event()))
    record = ledger.record(JOB)
    expected = (
        "Studio process worker exited with 3 after requesting its grant, before it was given."
    )
    assert (record.status, record.error, record.public_error) == ("failed", expected, expected)
    assert reservations(ledger) == []


def test_a_worker_whose_exit_is_not_reported_is_waited_for_until_the_grant_deadline(
    base: Path,
    launcher: Launcher,
    ledger: StudioJobLedger,
    authority: Authority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the launcher's word that the worker is gone, the grant deadline still ends the wait.

    Parameters
    ----------
    base : Path
        The test's namespace.
    launcher : Launcher
        The real launcher service.
    ledger : StudioJobLedger
        The real ledger.
    authority : Authority
        The real storage authority handlers.
    monkeypatch : pytest.MonkeyPatch
        Makes the supervisor's question to the launcher answer that the worker still runs.
    """
    monkeypatch.setattr(GenerationSupervisor, "_worker_exit_status", lambda self: None)
    admit(ledger, supervisor=supervisor_identity())
    runtime = api_runtime(base, authority, grant_timeout_seconds=3.0)
    job = analysis_job(SIMULATE, task="analysis.unknown")
    sealed(supervise_generation(runtime, job, cancel=threading.Event()))
    record = ledger.record(JOB)
    assert record.status == "failed"
    assert record.error == (
        "Studio worker grant failed: launched worker did not request its grant in time"
    )
    assert record.public_error == "Studio worker grant failed."
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
    assert record.public_error is None and record.to_public_dict()["error"] is None
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
    assert record.public_error == "Studio worker could not start: registration unanswered."
    assert record.to_public_dict()["error"] == record.public_error
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
    assert record.public_error == "Studio worker could not start: launcher refused (spool)."
    assert record.to_public_dict()["error"] == record.public_error
    assert (record.started_at_utc, reservations(ledger)) == (None, [])


@pytest.mark.parametrize("fault", ["missing-spool", "long-endpoint", "endpoint-refused"])
def test_staging_and_endpoint_failures_fail_before_any_launch(
    base: Path,
    ledger: StudioJobLedger,
    authority: Authority,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing is launched when the spool or the grant endpoint cannot be prepared.

    Parameters
    ----------
    base : Path
        The test's namespace.
    ledger : StudioJobLedger
        The real ledger.
    authority : Authority
        The real storage authority handlers.
    fault : str
        A missing spool, an endpoint path too long for a Unix socket, or an
        endpoint that refuses to open for another reason (an entry already there).
    monkeypatch : pytest.MonkeyPatch
        Makes the endpoint refuse to open, for the last fault.
    """
    admit(ledger, supervisor=supervisor_identity())
    root = base / "absent"
    if fault == "long-endpoint":
        root = base / ("s" * 60)
        root.mkdir(mode=0o750)
    if fault == "endpoint-refused":
        root = base / "spool"

        def refuse(self: object) -> None:
            """Refuse as the endpoint does when an entry with its name is already there."""
            raise FileExistsError("worker grant endpoint already exists")

        monkeypatch.setattr(WorkerGrantEndpoint, "open", refuse)
    runtime = api_runtime(base, authority, spool_root=root)
    sealed(supervise_generation(runtime, analysis_job(SIMULATE), cancel=threading.Event()))
    record = ledger.record(JOB)
    assert record.status == "failed" and record.error is not None
    if fault == "missing-spool":
        assert record.error.startswith("Studio worker could not start: spool:")
        assert record.public_error == "Studio worker could not start: spool."
    elif fault == "endpoint-refused":
        # A fault that is not a refusal written for the caller keeps its fixed public text.
        assert record.error == (
            "Studio worker could not start: grant endpoint: worker grant endpoint already exists"
        )
        assert record.public_error == "Studio worker could not start: grant endpoint."
    else:
        # The operator is told the length to cut, not only that the endpoint failed.
        told = re.fullmatch(
            r"Studio worker could not start: its grant endpoint path is (\d+) bytes; "
            r"a Unix socket path must be shorter than 108 bytes\.",
            record.error,
        )
        assert told is not None, record.error
        assert int(told.group(1)) >= 108
        assert record.public_error == record.error
    assert record.to_public_dict()["error"] == record.public_error
    assert reservations(ledger) == []
