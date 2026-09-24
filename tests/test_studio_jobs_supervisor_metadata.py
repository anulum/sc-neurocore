# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Supervisor metadata uncertainty

"""Probe failures do not prove death; an actually reaped process does.

Every probe here is real. The init process belongs to another identity, so
the kernel refuses the signal probe while its proc metadata stays readable,
as for a storage service observing a distinct API identity. Hidden proc
entries and a process vanishing between probe and read need a separate proc
mount or a race; they are exercised by the isolated three-identity proof.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from sc_neurocore.studio.platform import jobs_ledger_supervisor as supervisors

_foreign_init = pytest.mark.skipif(
    os.geteuid() == 0 or os.stat("/proc/1").st_uid == os.geteuid(),
    reason="needs an init process owned by another identity",
)


@pytest.mark.parametrize("pid_text", ["invalid", "0", "-1", str(2**64)])
def test_invalid_pid_cannot_identify_a_dead_supervisor(pid_text: str) -> None:
    """Malformed or unrepresentable process IDs are unknown, not reclaimable."""
    host, _, token = supervisors.supervisor_identity().split(":", 2)
    assert supervisors.supervisor_is_alive(f"{host}:{pid_text}:{token}") is None


@_foreign_init
def test_denied_probe_uses_the_matching_proc_generation() -> None:
    """Another identity's process is proven alive by its readable start token."""
    identity = supervisors.supervisor_identity(1)
    with pytest.raises(PermissionError):
        os.kill(1, 0)
    assert supervisors.supervisor_is_alive(identity) is True


@_foreign_init
def test_denied_probe_does_not_accept_a_reused_pid_generation() -> None:
    """Readable metadata still rejects a start token from another generation."""
    host, pid_text, token = supervisors.supervisor_identity(1).split(":", 2)
    assert supervisors.supervisor_is_alive(f"{host}:{pid_text}:{int(token) + 1}") is False


def test_reaped_child_has_an_unknown_identity_and_is_dead() -> None:
    """A reaped process yields start token 0 and its recorded identity is dead."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    live_identity = supervisors.supervisor_identity(child.pid)
    child.wait(timeout=10.0)
    assert supervisors.supervisor_identity(child.pid).endswith(":0")
    assert supervisors.supervisor_is_alive(supervisors.supervisor_identity(child.pid)) is None
    assert supervisors.supervisor_is_alive(live_identity) is False


def test_unreaped_child_is_dead_while_its_metadata_remains() -> None:
    """An exited, not yet collected child is not a live supervisor."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = supervisors.supervisor_identity(child.pid)
    descriptor = os.pidfd_open(child.pid)
    try:
        os.waitid(os.P_PIDFD, descriptor, os.WEXITED | os.WNOWAIT)
        assert supervisors.supervisor_is_alive(identity) is False
    finally:
        os.close(descriptor)
        child.wait(timeout=10.0)
