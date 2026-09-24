# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Read-only worker termination evidence

"""Prove group termination without signalling an orphan or a reused process ID."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from uuid import UUID

from sc_neurocore.studio.platform.jobs_process_state import process_exited


def worker_group_stopped(identity: str, boot_id: str, group_id: int) -> bool:
    """Return true only for a validated old boot or a locally stopped group.

    Missing metadata, foreign hosts, malformed identities and inaccessible
    process metadata retain capacity. A live group with a reused ID also stays
    occupied; this probe never signals it. A member counts as stopped only when
    every one of its threads has exited: zombies cannot execute or spawn work,
    but a zombie thread-group leader can still have running threads.
    """
    try:
        host, pid, token = identity.split(":", 2)
        if host != socket.gethostname() or int(pid) != group_id or group_id <= 0 or int(token) <= 0:
            return False
        previous_boot = UUID(boot_id)
        current_boot = UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        if previous_boot != current_boot:
            return True
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return True
        observed_member = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if os.getpgid(int(entry.name)) != group_id:
                    continue
                fields = (entry / "stat").read_text().rsplit(")", 1)[-1].split()
                if int(fields[2]) == group_id:
                    observed_member = True
                    if not process_exited(int(entry.name)):
                        return False
            except (ProcessLookupError, FileNotFoundError):
                continue
        if observed_member:
            return True
        # An existing group with no visible members is not proof of emptiness.
        # It may have vanished during enumeration, so recheck that exact group.
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return True
        return False
    except (OSError, ValueError, IndexError, TypeError, OverflowError):
        return False
