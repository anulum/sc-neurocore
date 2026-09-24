# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — lost replies and unconfirmed stops of launched generations

"""Lost messages resolve to one worker; unconfirmed stops keep custody and capacity.

Each case starts the real launcher service process, which spawns the real
bootstrap and worker running a reviewed named analysis task. The storage
authority is the real supervision and finish handlers over a real SQLite
ledger with delegated admission. Same-identity runs establish function only;
the three-identity proof belongs to the disposable container acceptance.
Lost messages are dropped on real connections by relays; survivors come
from a launcher whose kernel refuses ``pidfd_send_signal``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment
from sc_neurocore.studio.platform.storage_finish_protocol import StorageFinishResponse
from sc_neurocore.studio.platform.storage_generation_supervisor import (
    supervise_generation,
)
from tests.studio_seccomp_support import REPOSITORY, SECCOMP_AVAILABLE
from tests.studio_storage_generation_support import Authority, LauncherRelay
from tests.studio_storage_launcher_support import (
    Launcher,
    configuration,
    shutdown,
    start,
)
from tests.studio_storage_supervision_support import JOB, admit

from tests.studio_storage_generation_runs import *


def test_lost_launch_and_start_replies_are_resolved(
    base: Path, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A lost launch request, a lost launch reply and a lost start reply leave one worker."""
    running = start(base)
    relay = LauncherRelay(base / "sock" / "relay.sock", running.socket_path)
    try:
        admit(ledger, supervisor=supervisor_identity())
        relay.lose_request["launch"] = 1
        relay.lose_reply["launch"] = 1
        authority.lose["start"] = 1
        runtime = api_runtime(base, authority, launcher_socket=relay.path)
        sealed(supervise_generation(runtime, analysis_job(SIMULATE), cancel=threading.Event()))
    finally:
        relay.close()
        shutdown(running)
    assert ledger.record(JOB).status == "completed"
    assert relay.seen[:5] == ["launch", "status", "launch", "status", "status"]
    assert authority.seen[:2] == ["start", "start"]
    assert workers(ledger) == 1


@pytest.mark.parametrize("lost", [1, 3])
def test_lost_finish_replies_are_repeated_identically(
    base: Path, launcher: Launcher, ledger: StudioJobLedger, authority: Authority, lost: int
) -> None:
    """The retried finish is answered from the sealed record; unanswered ones raise."""
    admit(ledger, supervisor=supervisor_identity())
    authority.lose["finish"] = lost
    running = Background(api_runtime(base, authority), analysis_job(SIMULATE))
    response = running.wait()
    if lost == 1:
        assert isinstance(response, StorageFinishResponse)
        assert (response.reply, response.reason) == ("already_sealed", None)
    else:
        assert isinstance(response, TimeoutError)
    assert ledger.record(JOB).status == "completed"
    assert authority.seen.count("finish") == min(lost + 1, 3)


def test_unreachable_launcher_keeps_capacity_unreaped(
    base: Path, ledger: StudioJobLedger, authority: Authority
) -> None:
    """Without any launcher answer the job fails but its capacity is not released."""
    admit(ledger, supervisor=supervisor_identity())
    runtime = api_runtime(base, authority, launcher_socket=base / "sock" / "missing.sock")
    sealed(supervise_generation(runtime, analysis_job(SIMULATE), cancel=threading.Event()))
    record = ledger.record(JOB)
    assert (record.status, record.error) == (
        "failed",
        "Studio worker could not start: launcher unanswered.",
    )
    assert reservations(ledger) == ["unreaped"]


def test_a_restarted_launcher_leaves_the_generation_unreaped(
    base: Path, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A launcher without a record of the generation never confirms its stop."""
    first = start(base)
    admit(ledger, supervisor=supervisor_identity())
    running = Background(api_runtime(base, authority), analysis_job(LONG))
    await_running(ledger)
    shutdown(first)
    second = start(base)
    try:
        sealed(running.wait())
    finally:
        shutdown(second)
    record = ledger.record(JOB)
    assert (record.status, record.error) == (
        "failed",
        f"Studio worker generation is unknown to its launcher. {UNREAPED}",
    )
    assert reservations(ledger) == ["unreaped"]


REFUSING_LAUNCHER = (
    "import errno, sys\n"
    "from tests.studio_seccomp_support import Refusal, install_refusals\n"
    "from sc_neurocore.studio.platform.storage_launcher_service import main\n"
    "install_refusals([Refusal('pidfd_send_signal', errno.EPERM)])\n"
    "raise SystemExit(main(['--configuration', sys.argv[1]]))\n"
)


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="seccomp filters need Linux x86_64")
def test_undeliverable_stop_keeps_the_live_worker_in_custody(
    base: Path, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A stop the kernel refuses is never reported as a finished job."""
    config_path = base / "launcher.json"
    config_path.write_text(json.dumps(configuration(base)))
    environment = {
        **_process_worker_environment(),
        "PYTHONPATH": os.pathsep.join((str(REPOSITORY / "src"), str(REPOSITORY))),
    }
    process = subprocess.Popen(
        [sys.executable, "-c", REFUSING_LAUNCHER, str(config_path)],
        cwd=REPOSITORY,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stdout.readline() == b"ready\n"
    refusing = Launcher(process, base / "sock" / "launcher.sock", base / "spool")
    admit(ledger, supervisor=supervisor_identity())
    try:
        running = Background(api_runtime(base, authority), analysis_job(LONG))
        await_running(ledger)
        running.cancel.set()
        response = running.wait()
        assert isinstance(response, TimeoutError), ledger.record(JOB)
        assert ledger.record(JOB).status == "running"
        assert reservations(ledger) == ["running"]
        assert authority.seen.count("finish") == 3
    finally:
        worker = ledger.connection().execute("SELECT group_id FROM job_workers").fetchone()
        if worker is not None:
            os.killpg(int(worker[0]), signal.SIGKILL)
        ledger.close()
        shutdown(refusing)
