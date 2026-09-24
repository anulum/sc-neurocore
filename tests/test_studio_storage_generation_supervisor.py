# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API supervision of launched worker generations, end to end

"""A launched job ends as the embedded supervisor would end it.

Each case starts the real launcher service process, which spawns the real
bootstrap and worker running a reviewed named analysis task. The storage
authority is the real supervision and finish handlers over a real SQLite
ledger with delegated admission. Same-identity runs establish function only;
the three-identity proof belongs to the disposable container acceptance.
"""

from __future__ import annotations

from pathlib import Path
import stat
import threading

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_finish_protocol import StorageFinishResponse
from sc_neurocore.studio.platform.storage_generation_supervisor import (
    GenerationSupervisor,
    supervise_generation,
)
from tests.studio_storage_generation_support import Authority, LauncherRelay
from tests.studio_storage_launcher_support import (
    Launcher,
    shutdown,
    start,
)
from tests.studio_storage_supervision_support import JOB, admit

from tests.studio_storage_generation_runs import *


def test_completed_job_is_sealed_with_its_result(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority
) -> None:
    """Stage, launch, grant, run and finish: the result and capacity settle once."""
    admit(ledger, supervisor=supervisor_identity())
    cancel = threading.Event()
    supervisor = GenerationSupervisor(
        api_runtime(base, authority), analysis_job(SIMULATE), cancel=cancel
    )
    sealed(supervisor.run())
    record = ledger.record(JOB)
    assert record.status == "completed", record.error
    assert record.result is not None and record.result["evidence_receipt"]
    assert (record.error, reservations(ledger), workers(ledger)) == (None, [], 1)
    spool = base / "spool" / JOB / supervisor.generation
    result = spool / JOB / ".studio_process_result.json"
    assert result.is_file()
    # The worker's umask keeps its output readable by the compute group.
    assert stat.S_IMODE(result.stat().st_mode) & 0o077 == 0o040
    assert not (spool / "grant.sock").exists()
    assert authority.seen[0] == "start" and authority.seen[-1] == "finish"


def test_worker_failure_is_recorded_like_the_embedded_supervisor(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A task error with a non-zero exit fails the job with the worker's error."""
    admit(ledger, supervisor=supervisor_identity())
    sealed(
        supervise_generation(
            api_runtime(base, authority), analysis_job({}), cancel=threading.Event()
        )
    )
    record = ledger.record(JOB)
    assert (record.status, record.error, record.result) == (
        "failed",
        "AnalysisJobValidationError",
        None,
    )
    assert reservations(ledger) == []


@pytest.mark.parametrize("route", ["api", "ledger"])
def test_cancellation_stops_the_worker(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority, route: str
) -> None:
    """The API's own cancel, or one recorded in the ledger, stops and cancels the job."""
    admit(ledger, supervisor=supervisor_identity())
    running = Background(api_runtime(base, authority), analysis_job(LONG))
    await_running(ledger)
    if route == "api":
        running.cancel.set()
    else:
        ledger.transition(JOB, "cancelling")
        ledger.close()
    sealed(running.wait())
    record = ledger.record(JOB)
    assert (record.status, record.error, record.result) == ("cancelled", None, None)
    assert reservations(ledger) == []
    if route == "ledger":
        assert "heartbeat" in authority.seen


def test_deadline_times_out_through_lost_status_and_heartbeat_replies(
    base: Path, ledger: StudioJobLedger, authority: Authority
) -> None:
    """Lost launcher and storage replies are renewed later; the deadline still holds."""
    running = start(base)
    relay = LauncherRelay(base / "sock" / "relay.sock", running.socket_path)
    try:
        admit(ledger, supervisor=supervisor_identity())
        relay.lose_reply["status"] = 2
        authority.lose["heartbeat"] = 2
        runtime = api_runtime(base, authority, launcher_socket=relay.path)
        sealed(
            supervise_generation(runtime, analysis_job(LONG, timeout=3.0), cancel=threading.Event())
        )
    finally:
        relay.close()
        shutdown(running)
    record = ledger.record(JOB)
    assert (record.status, record.error) == ("timed_out", "Studio job exceeded its timeout.")
    assert reservations(ledger) == []
    assert authority.seen.count("heartbeat") > 2 and relay.seen.count("status") > 2


def test_a_job_ended_elsewhere_is_answered_from_its_record(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A refused heartbeat stops the worker; the finish meets the other terminal record."""
    admit(ledger, supervisor=supervisor_identity())
    running = Background(api_runtime(base, authority), analysis_job(LONG))
    await_running(ledger)
    ledger.transition(JOB, "cancelled")
    ledger.close()
    response = running.wait()
    assert isinstance(response, StorageFinishResponse)
    assert (response.reply, response.reason) == ("refused", "conflict")
    assert ledger.record(JOB).status == "cancelled"


def test_output_the_api_cannot_read_fails_a_job_that_ended_by_itself(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A result larger than the API reads is refused; the job fails without artefacts."""
    admit(ledger, supervisor=supervisor_identity())
    runtime = api_runtime(base, authority, frame_max_bytes=1024)
    sealed(supervise_generation(runtime, analysis_job(SIMULATE), cancel=threading.Event()))
    record = ledger.record(JOB)
    assert (record.status, record.result, record.artifacts) == ("failed", None, ())
    assert record.error == (
        "Studio worker output was refused: spool entry is not a bounded regular file"
    )
    assert reservations(ledger) == []


def test_unreadable_output_keeps_the_api_verdict(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A torn result planted in the worker directory does not turn a cancel into a failure."""
    admit(ledger, supervisor=supervisor_identity())
    running = Background(api_runtime(base, authority), analysis_job(LONG))
    await_running(ledger)
    (generation,) = (base / "spool" / JOB).iterdir()
    (generation / JOB / ".studio_process_result.json").write_text('{"status": "comp')
    running.cancel.set()
    sealed(running.wait())
    record = ledger.record(JOB)
    assert record.status == "cancelled"
    assert record.error is not None
    assert record.error.startswith("Studio worker output was refused: Unterminated string")
    assert reservations(ledger) == []
