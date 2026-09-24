# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — launcher-held worker process tree custody

"""Track and stop one launched worker and every descendant the launcher observes.

The launcher holds a pidfd for the worker leader and for each descendant it
discovers through ``/proc/<pid>/task/<tid>/children``. A pidfd refers to one
process generation, so a reused PID can never be signalled or reported as the
worker. The launcher is a child subreaper: a descendant orphaned by a fast
double fork or ``setsid`` is reparented to the worker leader (itself a
subreaper) or to the launcher, never to an unrelated init. The launcher kills
adopted processes that no tree has attributed, because jobs sharing a compute
identity may not leave daemons behind.

Limits: this is process-tree custody, not cgroup accounting. Discovery is by
polling, termination is by repeated walk-and-kill bounded by an explicit round
budget, and a process that is created faster than the walk can observe it is
only caught when it is reparented to a subreaper. Jobs sharing one compute UID
are not isolated from each other.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
import select
import signal

_PR_SET_CHILD_SUBREAPER = 36


@dataclass(frozen=True, slots=True)
class TrackedProcess:
    """One observed process generation held by an open pidfd.

    ``start_token`` is the ``/proc/<pid>/stat`` start time read after the
    pidfd was opened and while the parent relation was confirmed.
    """

    pid: int
    pidfd: int
    start_token: str


def process_control(option: int, value: int) -> None:
    """Apply one Linux ``prctl`` option with a single integer argument.

    Parameters
    ----------
    option : int
        ``PR_*`` option number from ``linux/prctl.h``.
    value : int
        First option argument; the remaining arguments are zero.

    Raises
    ------
    OSError
        The kernel refuses the call; the kernel errno is preserved.
    """
    # The interpreter's own symbol scope already contains the C library.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def become_child_subreaper() -> None:
    """Mark the calling process as a child subreaper.

    Raises
    ------
    OSError
        The kernel refuses the ``prctl`` call.
    """
    process_control(_PR_SET_CHILD_SUBREAPER, 1)


def _stat_fields(pid: int) -> list[str]:
    with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as handle:
        return handle.read().rsplit(")", 1)[-1].split()


def process_start_token(pid: int) -> str:
    """Return the ``/proc`` start time of ``pid`` as a decimal string.

    Raises
    ------
    OSError
        The process metadata is unavailable.
    """
    return _stat_fields(pid)[19]


def _children(pid: int) -> set[int]:
    """Return direct children of every thread of ``pid``; a vanished task is skipped."""
    found: set[int] = set()
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return found
    for task in tasks:
        try:
            with open(f"/proc/{pid}/task/{task}/children", encoding="ascii") as handle:
                text = handle.read()
        except OSError:
            continue
        found.update(int(item) for item in text.split())
    return found


def _exited(pidfd: int) -> bool:
    return bool(select.select([pidfd], [], [], 0)[0])


def _track_child(parent: TrackedProcess, pid: int) -> TrackedProcess | None:
    """Open a pidfd for ``pid`` only while it is still a child of ``parent``."""
    try:
        pidfd = os.pidfd_open(pid)
    except OSError:
        return None
    try:
        fields = _stat_fields(pid)
        if int(fields[1]) != parent.pid or _exited(parent.pidfd):
            os.close(pidfd)
            return None
        return TrackedProcess(pid=pid, pidfd=pidfd, start_token=fields[19])
    except OSError:
        os.close(pidfd)
        return None


class WorkerTree:
    """Launcher custody of one worker leader and its observed descendants.

    The tree owns every pidfd it opens and closes them in :meth:`close`.
    ``leader`` must be a direct child of the launcher so the launcher can reap
    it; descendants are reaped by their parents or by :func:`reap_adopted`.
    """

    def __init__(self, leader: TrackedProcess) -> None:
        """Take ownership of the leader pidfd."""
        self._leader = leader
        self._members: dict[int, TrackedProcess] = {leader.pid: leader}

    @property
    def leader(self) -> TrackedProcess:
        """Return the launched worker leader."""
        return self._leader

    def owns(self, pid: int) -> bool:
        """Return whether ``pid`` is an observed member of this tree."""
        return pid in self._members

    def observe(self) -> None:
        """Walk from every live member and track newly visible children."""
        frontier = [member for member in self._members.values() if not _exited(member.pidfd)]
        while frontier:
            parent = frontier.pop()
            for pid in _children(parent.pid) - self._members.keys():
                tracked = _track_child(parent, pid)
                if tracked is not None:
                    self._members[pid] = tracked
                    frontier.append(tracked)

    def live(self) -> tuple[TrackedProcess, ...]:
        """Return tracked members whose pidfd does not yet report exit."""
        return tuple(member for member in self._members.values() if not _exited(member.pidfd))

    def kill(self, *, rounds: int) -> bool:
        """SIGKILL every observed member until none is live or rounds run out.

        Parameters
        ----------
        rounds : int
            Positive number of observe-and-kill passes.

        Returns
        -------
        bool
            ``True`` when no tracked member remains live after the last pass.

        Notes
        -----
        A pass observes first so children forked since the previous pass are
        signalled too. Exit is observed through each pidfd; zombies count as
        exited because they cannot execute. A signal the kernel refuses to
        deliver, for example under a security module, leaves that member live,
        so the stop is reported unconfirmed rather than raised.
        """
        if type(rounds) is not int or rounds <= 0:
            raise ValueError("worker tree kill rounds must be positive")
        for _ in range(rounds):
            self.observe()
            live = self.live()
            if not live:
                return True
            for member in live:
                try:
                    signal.pidfd_send_signal(member.pidfd, signal.SIGKILL)
                except OSError:
                    # Already exited, or delivery refused: the next check decides.
                    continue
            for member in live:
                select.select([member.pidfd], [], [], 0.05)
        self.observe()
        return not self.live()

    def close(self) -> None:
        """Close every pidfd held by this tree."""
        members = self._members
        self._members = {}
        for member in members.values():
            os.close(member.pidfd)


def reap_adopted(trees: list[WorkerTree], *, leaders: set[int]) -> tuple[int, ...]:
    """Reap adopted zombies and kill adopted processes no tree has attributed.

    Parameters
    ----------
    trees : list[WorkerTree]
        Every tree the launcher currently holds.
    leaders : set[int]
        Worker leader PIDs whose status the launcher collects through their
        own process handles; they are never reaped here.

    Returns
    -------
    tuple[int, ...]
        PIDs of unattributed adopted processes that were sent SIGKILL.

    Notes
    -----
    Every listed process is a child of the calling launcher, which collects
    children only on its single serving thread. No other process can reap or
    reparent it first, so its PID cannot be reused while this call runs. A
    kill the kernel refuses to deliver leaves that process unreported.
    """
    killed: list[int] = []
    for pid in sorted(_children(os.getpid()) - leaders):
        if _stat_fields(pid)[0] == "Z":
            os.waitpid(pid, os.WNOHANG)
            continue
        if any(tree.owns(pid) for tree in trees):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
        killed.append(pid)
    return tuple(killed)
