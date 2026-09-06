# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job admission control

"""How many jobs may run at once, and what happens to the rest.

Without a ceiling every submitted job starts immediately, so a caller decides
how much of the machine the Studio uses. With a ceiling and no queue, the
overflow is simply lost. This module gives both: a bounded number of jobs
running at once, a bounded queue behind it, and a refusal with a stated reason
once the queue is full — predictable rather than silent.

Admission is separate from supervision on purpose. A job is admitted, then
supervised; the ledger records it either way, and a refused job never reaches
the ledger at all, because it never happened.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from sc_neurocore.studio.platform.jobs_models import StudioJobRejected

DEFAULT_MAX_CONCURRENT_JOBS = 4
DEFAULT_MAX_QUEUED_JOBS = 32


class StudioJobQueueFull(StudioJobRejected):
    """Raised when both the running slots and the queue behind them are full.

    Attributes
    ----------
    running : int
        Jobs occupying a slot when the request arrived.
    queued : int
        Jobs already waiting.
    limit : int
        The queue ceiling that was reached.
    """

    def __init__(self, *, running: int, queued: int, limit: int) -> None:
        super().__init__(
            f"Studio job queue is full: {running} running, {queued} queued, limit {limit}."
        )
        self.running = running
        self.queued = queued
        self.limit = limit

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "error": "job_queue_full",
            "limit": self.limit,
            "queued": self.queued,
            "reason": str(self),
            "running": self.running,
        }


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    """What admission control is doing right now.

    Attributes
    ----------
    running : int
        Jobs holding a slot.
    queued : int
        Jobs waiting for one.
    max_concurrent : int
        How many may run at once.
    max_queued : int
        How many may wait.
    admitted : int
        Jobs admitted since this controller started.
    refused : int
        Submissions refused because the queue was full.
    """

    running: int
    queued: int
    max_concurrent: int
    max_queued: int
    admitted: int
    refused: int

    def to_public_dict(self) -> dict[str, int]:
        """Return a JSON-serializable, path-free admission snapshot."""
        return {
            "admitted": self.admitted,
            "max_concurrent": self.max_concurrent,
            "max_queued": self.max_queued,
            "queued": self.queued,
            "refused": self.refused,
            "running": self.running,
        }


class StudioJobAdmission:
    """A bounded number of running jobs, with a bounded queue behind them.

    Parameters
    ----------
    max_concurrent : int
        Jobs allowed to run at once.
    max_queued : int
        Jobs allowed to wait for a slot. A submission that arrives when both
        are full is refused with :class:`StudioJobQueueFull`.
    """

    def __init__(
        self,
        *,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_JOBS,
        max_queued: int = DEFAULT_MAX_QUEUED_JOBS,
    ) -> None:
        if max_concurrent <= 0:
            raise ValueError("Studio job concurrency ceiling must be positive.")
        if max_queued < 0:
            raise ValueError("Studio job queue ceiling cannot be negative.")
        self._max_concurrent = max_concurrent
        self._max_queued = max_queued
        self._condition = threading.Condition()
        self._running = 0
        self._queued = 0
        self._admitted = 0
        self._refused = 0

    def reserve(self, *, timeout_seconds: float | None = None) -> None:
        """Take a slot, waiting in the queue when they are all occupied.

        Parameters
        ----------
        timeout_seconds : float, optional
            How long to wait for a slot. ``None`` waits indefinitely; a
            submission that times out releases its queue place and is refused.

        Raises
        ------
        StudioJobQueueFull
            Every slot is occupied and the queue is full, or the wait timed out
            without a slot becoming free.
        """
        with self._condition:
            if self._running < self._max_concurrent:
                self._running += 1
                self._admitted += 1
                return
            if self._queued >= self._max_queued:
                self._refused += 1
                raise StudioJobQueueFull(
                    running=self._running, queued=self._queued, limit=self._max_queued
                )
            self._queued += 1
            try:
                admitted = self._condition.wait_for(
                    lambda: self._running < self._max_concurrent, timeout=timeout_seconds
                )
            finally:
                self._queued -= 1
            if not admitted:
                self._refused += 1
                raise StudioJobQueueFull(
                    running=self._running, queued=self._queued, limit=self._max_queued
                )
            self._running += 1
            self._admitted += 1

    def release(self) -> None:
        """Give a slot back and wake one waiting submission."""
        with self._condition:
            if self._running > 0:
                self._running -= 1
            self._condition.notify()

    def snapshot(self) -> AdmissionSnapshot:
        """Return the current admission state."""
        with self._condition:
            return AdmissionSnapshot(
                running=self._running,
                queued=self._queued,
                max_concurrent=self._max_concurrent,
                max_queued=self._max_queued,
                admitted=self._admitted,
                refused=self._refused,
            )


__all__ = [
    "DEFAULT_MAX_CONCURRENT_JOBS",
    "DEFAULT_MAX_QUEUED_JOBS",
    "AdmissionSnapshot",
    "StudioJobAdmission",
    "StudioJobQueueFull",
]
