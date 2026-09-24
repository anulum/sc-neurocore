# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Worker recovery evidence boundaries

"""Never confuse missing process visibility with proof that a group stopped.

Hidden or refused process listings and a group vanishing mid-scan need a
separate proc mount or a race; the isolated and adversarial proofs cover them.
"""

from __future__ import annotations

import socket
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

import pytest

from sc_neurocore.studio.platform.jobs_reaper import reap_process_group
from sc_neurocore.studio.platform.jobs_worker_recovery import worker_group_stopped


@pytest.mark.parametrize("evidence", ["current-boot", "old-boot"])
def test_live_group_is_retained_unless_evidence_is_from_an_earlier_boot(evidence: str) -> None:
    """A real live group is retained, except evidence explicitly from an earlier boot."""
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        start_new_session=True,
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if evidence == "old-boot":
        boot = str(UUID(int=UUID(boot).int ^ 1))
    # A deliberately different start token models a reused PID, not its actual
    # reuse by the kernel. The real live group must nevertheless remain held.
    identity = f"{socket.gethostname()}:{worker.pid}:1"
    try:
        assert worker_group_stopped(identity, boot, worker.pid) is (evidence == "old-boot")
        assert worker.poll() is None
    finally:
        assert reap_process_group(worker, owned_group_id=worker.pid).reaped


def test_zombie_leader_with_running_threads_is_not_stopped() -> None:
    """A leader thread that exits while other threads compute leaves the group live."""
    program = (
        "import ctypes, threading, time\n"
        "threading.Thread(target=lambda: time.sleep(60)).start()\n"
        "print('ready', flush=True)\n"
        "ctypes.CDLL(None).syscall(60, 0)\n"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", program],
        start_new_session=True,
        stdout=subprocess.PIPE,
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    identity = f"{socket.gethostname()}:{worker.pid}:1"
    try:
        assert worker.stdout is not None and worker.stdout.readline() == b"ready\n"
        stat = Path(f"/proc/{worker.pid}/stat")
        deadline = time.monotonic() + 10.0
        while stat.read_text().rsplit(")", 1)[-1].split()[0] != "Z":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert worker.poll() is None
        assert worker_group_stopped(identity, boot, worker.pid) is False
    finally:
        assert reap_process_group(worker, owned_group_id=worker.pid).reaped
        if worker.stdout is not None:
            worker.stdout.close()


@pytest.mark.parametrize("identity", ["bad", "foreign:1:1", "host:-1:1", "host:1:0"])
def test_malformed_or_foreign_identity_is_not_stop_evidence(identity: str) -> None:
    """Unqualified identity never releases capacity."""
    assert not worker_group_stopped(identity, "not-a-boot", 1)


@pytest.mark.parametrize("group_id", [2**31, 2**63 - 1])
def test_unrepresentable_group_id_is_not_stop_evidence(group_id: int) -> None:
    """Corrupt group IDs fitting SQLite integers must not crash recovery."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    identity = f"{socket.gethostname()}:{group_id}:1"
    assert worker_group_stopped(identity, boot, group_id) is False


def test_real_zombie_group_is_stopped_before_reaping() -> None:
    """An exited but unreaped session leader cannot execute further work."""
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    identity = f"{socket.gethostname()}:{child.pid}:1"
    try:
        observed = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        assert observed is not None and observed.si_pid == child.pid
        assert Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[-1].split()[0] == "Z"
        assert worker_group_stopped(identity, boot, child.pid) is True
        assert child.returncode is None
    finally:
        child.wait(timeout=3.0)
