# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio worker process-group reaping

"""Stop and verify a worker's entire process group, including descendants.

Signal, escalate and check the group: direct-child exit alone is insufficient.
Reports preserve cleanup outcome, elapsed time and surviving process IDs so
the supervisor can retain custody instead of claiming an unverified stop.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Literal

from sc_neurocore.studio.platform.jobs_process_state import group_is_gone, group_survivors

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


def _signal_group(group_id: int, number: int) -> None:
    """Signal a whole process group, ignoring one that has already gone.

    ``group_id`` is always a worker's own session group (its PID, above 1) and
    never the caller's group, so ``killpg(1)``, which is ``kill(-1)``, and
    ``killpg(0)`` cannot occur here.
    """
    try:
        os.killpg(group_id, number)
    except (ProcessLookupError, PermissionError, OSError):
        return


def _wait_for_group(group_id: int, *, deadline: float, process: subprocess.Popen[bytes]) -> bool:
    """Wait for a group to stop running, collecting the direct child as it goes."""
    while time.monotonic() < deadline:
        process.poll()
        if group_is_gone(group_id):
            return True
        time.sleep(_POLL_INTERVAL_SECONDS)
    process.poll()
    return group_is_gone(group_id)


def reap_process_group(
    process: subprocess.Popen[bytes],
    *,
    owned_group_id: int | None = None,
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
    owned_group_id : int, optional
        Group captured by the caller for a worker it started in its own session.
        Retains descendant custody after the direct child has been collected.
        Must equal that worker's PID and must not be the caller's group.

    Returns
    -------
    ReapReport
        The outcome, never an exception.
    """
    started = time.monotonic()
    group_id = process_group_of(process)
    if owned_group_id is not None:
        if owned_group_id != process.pid or owned_group_id == os.getpgrp():
            return ReapReport(
                outcome="unreaped",
                group_id=None,
                returncode=process.returncode,
                duration_seconds=time.monotonic() - started,
            )
        group_id = owned_group_id
    if process.poll() is not None and (group_id is None or group_is_gone(group_id)):
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
        if process.returncode is None:
            # Both signals were refused or ignored: the worker still runs.
            return ReapReport(
                outcome="unreaped",
                group_id=None,
                returncode=None,
                duration_seconds=time.monotonic() - started,
                survivors=(process.pid,),
            )
        return ReapReport(
            outcome="killed" if process.returncode != 0 else "terminated",
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
        survivors=group_survivors(group_id),
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
    except OSError:
        # Delivery refused (for example by a security policy): report, not retry.
        return
    try:
        process.kill()
        process.wait(timeout=kill_grace)
    except (subprocess.TimeoutExpired, OSError):
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
