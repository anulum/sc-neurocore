# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The time budget of the recorder's real-process tests

"""The one rule by which the recorder's real-process tests bound a command.

A test of the recorder starts real processes. It times a baseline command on the
host it runs on and derives from that time how long a bounded command may take
and how often a waiting thread looks. The rule lives here so that it has one
definition and its own tests.

The budget is the baseline times a fixed factor, widened by the host's load. The
factor does not depend on the number of processors: an earlier form multiplied
by one plus the processor count, which gave a host with four processors a fifth
of the margin of the 24-processor host the tests were qualified on, and a hosted
runner timed out on a command that had not hung. The factor is the one that
earlier form gave on that reference host.
"""

from __future__ import annotations

import math
import os

BUDGET_FACTOR = 25
BUDGET_CEILING_SECONDS = 4 * 60 * 60


def usable_processors() -> int:
    """Return the number of processors this process may run on.

    Returns
    -------
    int
        The size of the scheduling affinity where the platform reports one,
        else the processor count, and one when that is unknown.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def host_pressure() -> float:
    """Return how many runnable tasks stand per usable processor, at least one.

    Returns
    -------
    float
        The one-minute load average divided by the usable processors, never
        below one. One where the platform has no load average.
    """
    if not hasattr(os, "getloadavg"):
        return 1.0
    return max(1.0, os.getloadavg()[0] / usable_processors())


def _require_time(value: float, what: str) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"the {what} must be a positive, finite time in seconds, got {value!r}")


def command_budget(baseline: float, *, ceiling: float = BUDGET_CEILING_SECONDS) -> float:
    """Return how long a bounded command may take on this host.

    Parameters
    ----------
    baseline : float
        The measured time of the test's baseline command, in seconds.
    ceiling : float
        The most the budget may be, in seconds.

    Returns
    -------
    float
        The baseline times the fixed factor times the host's pressure, and at
        most the ceiling.

    Raises
    ------
    ValueError
        If the baseline or the ceiling is not a positive, finite time.
    """
    _require_time(baseline, "measured baseline")
    _require_time(ceiling, "budget ceiling")
    return min(ceiling, baseline * BUDGET_FACTOR * host_pressure())


def poll_interval(baseline: float) -> float:
    """Return how long a waiting thread sleeps between two looks.

    Parameters
    ----------
    baseline : float
        The measured time of the test's baseline command, in seconds.

    Returns
    -------
    float
        The baseline divided by the fixed factor.

    Raises
    ------
    ValueError
        If the baseline is not a positive, finite time.
    """
    _require_time(baseline, "measured baseline")
    return baseline / BUDGET_FACTOR


__all__ = [
    "BUDGET_CEILING_SECONDS",
    "BUDGET_FACTOR",
    "command_budget",
    "host_pressure",
    "poll_interval",
    "usable_processors",
]
