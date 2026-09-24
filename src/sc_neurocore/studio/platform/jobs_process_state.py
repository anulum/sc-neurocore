# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio worker process-state observation

"""Observe whether worker processes and their groups still execute.

Only ``/proc`` and signal-0 probes are used; nothing here signals or reaps.
"""

from __future__ import annotations

import os


def process_exited(pid: int) -> bool:
    """Return whether every thread of a process has exited.

    A zombie still belongs to its process group, so ``killpg(group, 0)`` keeps
    succeeding for it; treating it as running would report every ordinary reap
    as a failure. A thread-group leader that exits while its other threads run
    is also shown as a zombie, yet its process still executes, so each thread
    is checked. A process or thread that vanished during the check has exited.
    """
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return True
    for task in tasks:
        try:
            with open(f"/proc/{pid}/task/{task}/stat", encoding="utf-8", errors="replace") as file:
                state = file.read().rsplit(")", 1)[-1].split()[0]
        except OSError:
            continue
        if state not in {"Z", "X"}:
            return False
    return True


def group_survivors(group_id: int) -> tuple[int, ...]:
    """Return the process ids still running in one group, zombies excluded."""
    survivors: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) != group_id:
                continue
        except (ProcessLookupError, PermissionError, OSError):
            continue
        if not process_exited(pid):
            survivors.append(pid)
    return tuple(survivors)


def group_is_gone(group_id: int) -> bool:
    """Return whether nothing in the group is still running."""
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        # A group with members this identity may not signal is not gone.
        return False
    # The group still exists as far as the kernel is concerned; that is only
    # meaningful if something in it is actually running.
    return not group_survivors(group_id)
