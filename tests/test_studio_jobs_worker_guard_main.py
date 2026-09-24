# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Worker lifetime guard entry point

"""Reject malformed guard invocations and stop the group when custody is lost.

Every guard here is the real module in its own interpreter. Guards that may
signal a group always run inside a dedicated session created for the test, so
they can never reach the test runner's own process group.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment

GUARD = "sc_neurocore.studio.platform.jobs_worker_guard"


def _run(code: str, *, new_session: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=_process_worker_environment(),
        start_new_session=new_session,
        text=True,
        timeout=60.0,
        check=False,
    )


def _dead_identity() -> str:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = supervisor_identity(child.pid)
    child.wait(timeout=10.0)
    return identity


def test_malformed_invocations_refuse_before_any_signal() -> None:
    """Missing arguments, a foreign group or a group-leading guard exit with 2."""
    program = (
        "import os, runpy, sys\n"
        "cases = [[], ['supervisor', '-1'], ['supervisor', str(os.getpgrp() + 1)],\n"
        "         ['supervisor', str(os.getpid())]]\n"
        "codes = []\n"
        "for arguments in cases:\n"
        "    sys.argv = ['guard', *arguments]\n"
        "    try:\n"
        f"        runpy.run_module({GUARD!r}, run_name='__main__')\n"
        "    except SystemExit as exited:\n"
        "        codes.append(exited.code)\n"
        "print(codes)\n"
    )
    completed = _run(program, new_session=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[2, 2, 2, 2]"


def test_arming_requires_a_dedicated_session_leader() -> None:
    """A worker that does not lead its own group refuses before spawning a guard."""
    program = (
        f"from {GUARD} import arm_worker_guard\n"
        "try:\n"
        "    arm_worker_guard('supervisor')\n"
        "except ValueError as refused:\n"
        "    print(refused)\n"
    )
    completed = _run(program, new_session=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "Worker guard requires a dedicated session leader."


def test_guard_stops_its_group_when_the_supervisor_is_already_gone() -> None:
    """A guard started for a dead supervisor kills the whole group at once."""
    program = (
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-m', {GUARD!r}, sys.argv[1], str(os.getpgrp())])\n"
        "time.sleep(60)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", program, _dead_identity()],
        env=_process_worker_environment(),
        start_new_session=True,
    )
    try:
        assert leader.wait(timeout=30.0) == -signal.SIGKILL
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=10.0)


def test_armed_guard_stops_its_group_when_a_live_supervisor_dies() -> None:
    """After a ready handshake, supervisor death kills the worker's group."""
    supervisor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    program = (
        "import sys, time\n"
        f"from {GUARD} import arm_worker_guard\n"
        "arm_worker_guard(sys.argv[1])\n"
        "print('armed', flush=True)\n"
        "time.sleep(60)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", program, supervisor_identity(supervisor.pid)],
        env=_process_worker_environment(),
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert leader.stdout is not None
        assert leader.stdout.readline().strip() == "armed"
        supervisor.kill()
        supervisor.wait(timeout=10.0)
        stopped_at = time.monotonic()
        assert leader.wait(timeout=30.0) == -signal.SIGKILL
        assert time.monotonic() - stopped_at < 10.0
    finally:
        for process in (leader, supervisor):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10.0)
        if leader.stdout is not None:
            leader.stdout.close()
