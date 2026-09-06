# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio worker process-group reaping

"""Stopping a worker means stopping everything it started.

Terminating the direct child is not stopping the job. A worker that spawned a
subprocess leaves it running, still holding the job's files and its CPU, while
the record says the job timed out — the Studio reporting an end that did not
happen. So a worker runs in its own process group, and the group is what gets
signalled, escalated and then *checked*.

The check is the point. A reap reports what it actually achieved: whether the
group is gone, how it went (exited, terminated, killed) and how long it took.
Nothing here raises: a supervisor that crashes while cleaning up leaves a job
with no terminal record at all, which is worse than a job whose reap is
recorded as incomplete.
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Literal

#: How long the group gets to exit after SIGTERM before SIGKILL follows.
DEFAULT_TERMINATE_GRACE_SECONDS = 2.0
#: How long the group gets to disappear after SIGKILL before the reap is
#: reported as incomplete.
DEFAULT_KILL_GRACE_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.02

ReapOutcome = Literal["exited", "terminated", "killed", "unreaped"]


@dataclass(frozen=True, slots=True)
class ReapReport:
    """What stopping one worker process group actually achieved.

    Attributes
    ----------
    outcome : {"exited", "terminated", "killed", "unreaped"}
        ``exited`` when the worker had already finished, ``terminated`` when it
        stopped on SIGTERM, ``killed`` when SIGKILL was needed, and
        ``unreaped`` when the group was still there afterwards.
    group_id : int or None
        The process group signalled, when one could be resolved.
    returncode : int or None
        The direct worker's exit status, when it was collected.
    duration_seconds : float
        Wall-clock time the reap took.
    survivors : tuple of int
        Process ids still alive in the group when the reap gave up. Empty
        unless ``outcome`` is ``unreaped``.
    """

    outcome: ReapOutcome
    group_id: int | None
    returncode: int | None
    duration_seconds: float
    survivors: tuple[int, ...] = ()

    @property
    def reaped(self) -> bool:
        """Return whether nothing from the worker is still running."""
        return self.outcome != "unreaped"

    def to_public_dict(self) -> dict[str, object]:
        """Return a path-free JSON representation of this reap."""
        return {
            "duration_seconds": round(self.duration_seconds, 3),
            "group_id": self.group_id,
            "outcome": self.outcome,
            "reaped": self.reaped,
            "returncode": self.returncode,
            "survivor_count": len(self.survivors),
        }


def process_group_of(process: subprocess.Popen[bytes]) -> int | None:
    """Return the worker's process group, or ``None`` when it has none.

    A worker started with ``start_new_session=True`` leads its own group, so
    the group id equals its pid. Reading it from the operating system rather
    than assuming it keeps the reap honest when the process has already gone.
    """
    try:
        return os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def _is_zombie(pid: int) -> bool:
    """Return whether a process has exited and is only awaiting collection.

    A zombie still belongs to its process group, so ``killpg(group, 0)`` keeps
    succeeding for it. Treating that as "still running" would report every
    ordinary reap as a failure.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
            state = handle.read().rsplit(")", 1)[-1].split()[0]
    except (OSError, IndexError):
        return True
    return state == "Z"


def _group_survivors(group_id: int) -> tuple[int, ...]:
    """Return the process ids still running in one group, zombies excluded."""
    survivors: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:  # pragma: no cover - non-Linux fallback
        return ()
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) != group_id:
                continue
        except (ProcessLookupError, PermissionError, OSError):
            continue
        if not _is_zombie(pid):
            survivors.append(pid)
    return tuple(survivors)


def _group_is_gone(group_id: int) -> bool:
    """Return whether nothing in the group is still running."""
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return True
    except PermissionError:  # pragma: no cover - a foreign group we cannot see
        return False
    except OSError as exc:  # pragma: no cover - defensive
        return exc.errno == errno.ESRCH
    # The group still exists as far as the kernel is concerned; that is only
    # meaningful if something in it is actually running.
    return not _group_survivors(group_id)


def _signal_group(group_id: int, number: int) -> None:
    """Signal a whole process group, ignoring one that has already gone."""
    try:
        os.killpg(group_id, number)
    except (ProcessLookupError, PermissionError, OSError):
        return


def _wait_for_group(
    group_id: int, *, deadline: float, process: subprocess.Popen[bytes] | None = None
) -> bool:
    """Wait for a group to stop running, collecting the direct child as it goes."""
    while time.monotonic() < deadline:
        if process is not None:
            process.poll()
        if _group_is_gone(group_id):
            return True
        time.sleep(_POLL_INTERVAL_SECONDS)
    if process is not None:
        process.poll()
    return _group_is_gone(group_id)


def reap_process_group(
    process: subprocess.Popen[bytes],
    *,
    terminate_grace_seconds: float = DEFAULT_TERMINATE_GRACE_SECONDS,
    kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS,
) -> ReapReport:
    """Stop a worker and everything it started, and report what happened.

    SIGTERM to the group first, so a worker that handles it can seal its own
    files; SIGKILL to the group if the grace period passes; then a check that
    the group is actually gone.

    Parameters
    ----------
    process : subprocess.Popen
        The worker. It must have been started with ``start_new_session=True``,
        or it shares the supervisor's group and only the direct child is
        signalled.
    terminate_grace_seconds : float
        How long the group may take to exit on SIGTERM.
    kill_grace_seconds : float
        How long the group may take to disappear after SIGKILL.

    Returns
    -------
    ReapReport
        The outcome, never an exception.
    """
    started = time.monotonic()
    group_id = process_group_of(process)
    if process.poll() is not None and (group_id is None or _group_is_gone(group_id)):
        return ReapReport(
            outcome="exited",
            group_id=group_id,
            returncode=process.returncode,
            duration_seconds=time.monotonic() - started,
        )
    if group_id is None or group_id == os.getpgrp():
        # No separate group: signalling the group would hit this very process,
        # so only the direct child can be stopped. The caller is expected to
        # start workers in their own session; this branch keeps a misconfigured
        # worker from taking the Studio down with it.
        _terminate_direct_child(process, terminate_grace_seconds, kill_grace_seconds)
        return ReapReport(
            outcome="killed" if process.returncode not in (0, None) else "terminated",
            group_id=None,
            returncode=process.returncode,
            duration_seconds=time.monotonic() - started,
        )

    _signal_group(group_id, signal.SIGTERM)
    if _wait_for_group(
        group_id, deadline=time.monotonic() + terminate_grace_seconds, process=process
    ):
        _collect(process)
        return ReapReport(
            outcome="terminated",
            group_id=group_id,
            returncode=process.returncode,
            duration_seconds=time.monotonic() - started,
        )

    _signal_group(group_id, signal.SIGKILL)
    if _wait_for_group(group_id, deadline=time.monotonic() + kill_grace_seconds, process=process):
        _collect(process)
        return ReapReport(
            outcome="killed",
            group_id=group_id,
            returncode=process.returncode,
            duration_seconds=time.monotonic() - started,
        )

    _collect(process)
    return ReapReport(
        outcome="unreaped",
        group_id=group_id,
        returncode=process.returncode,
        duration_seconds=time.monotonic() - started,
        survivors=_group_survivors(group_id),
    )


def _collect(process: subprocess.Popen[bytes]) -> None:
    """Reap the direct child's exit status without blocking indefinitely."""
    try:
        process.wait(timeout=_POLL_INTERVAL_SECONDS * 10)
    except subprocess.TimeoutExpired:
        return


def _terminate_direct_child(
    process: subprocess.Popen[bytes], terminate_grace: float, kill_grace: float
) -> None:
    """Stop one worker that does not lead its own group."""
    try:
        process.terminate()
        process.wait(timeout=terminate_grace)
        return
    except subprocess.TimeoutExpired:
        pass
    except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover - defensive
        return
    try:
        process.kill()
        process.wait(timeout=kill_grace)
    except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError, OSError):
        return


__all__ = [
    "DEFAULT_KILL_GRACE_SECONDS",
    "_terminate_direct_child",
    "DEFAULT_TERMINATE_GRACE_SECONDS",
    "ReapOutcome",
    "ReapReport",
    "process_group_of",
    "reap_process_group",
]
