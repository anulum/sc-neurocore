# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - AER priority queue reference contract.

"""Reference contract for strict-priority AER queue scheduling.

Lower numeric priority is more urgent. Priority ties are served FIFO. The
model mirrors the hardware contract: finite capacity, explicit backpressure,
sticky drop/deadline flags, and cycle-stepped output readiness.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AERPriorityEvent:
    """Address-event payload emitted by the hardware priority queue."""

    target_id: int
    weight: int
    timestamp: int
    priority: int


@dataclass(frozen=True, slots=True)
class AERPriorityStep:
    """One cycle of queue input/output activity."""

    output: AERPriorityEvent | None
    accepted: bool
    ready: bool
    occupancy: int
    dropped_event: bool
    critical_deadline_violation: bool


class AERPriorityQueueReference:
    """Finite AER strict-priority queue with hardware-equivalent traps."""

    def __init__(self, *, capacity: int, max_latency_cycles: int = 16) -> None:
        if capacity <= 0:
            msg = "capacity must be positive"
            raise ValueError(msg)
        if max_latency_cycles < 0:
            msg = "max_latency_cycles must be non-negative"
            raise ValueError(msg)
        self.capacity = capacity
        self.max_latency_cycles = max_latency_cycles
        self._queue: list[AERPriorityEvent] = []
        self.dropped_event = False
        self.critical_deadline_violation = False
        self._critical_wait_cycles = 0

    @property
    def occupancy(self) -> int:
        return len(self._queue)

    @property
    def ready(self) -> bool:
        return len(self._queue) < self.capacity

    @property
    def empty(self) -> bool:
        return not self._queue

    @property
    def full(self) -> bool:
        return len(self._queue) >= self.capacity

    def enqueue(self, event: AERPriorityEvent) -> bool:
        """Accept an event if capacity is available, otherwise latch a drop."""

        if self.full:
            self.dropped_event = True
            return False
        self._queue.append(event)
        return True

    def peek(self) -> AERPriorityEvent | None:
        """Return the next event without consuming it."""

        if not self._queue:
            return None
        best_index = self._best_index()
        return self._queue[best_index]

    def dequeue(self) -> AERPriorityEvent | None:
        """Consume and return the highest-priority event."""

        if not self._queue:
            return None
        best_index = self._best_index()
        return self._queue.pop(best_index)

    def step(
        self,
        event: AERPriorityEvent | None,
        *,
        out_ready: bool,
    ) -> AERPriorityStep:
        """Advance one hardware cycle with optional input and output ready."""

        output = self.peek()
        ready = self.ready or (output is not None and out_ready)
        accepted = False

        if output is not None and output.priority == 0 and not out_ready:
            if self._critical_wait_cycles >= self.max_latency_cycles:
                self.critical_deadline_violation = True
            else:
                self._critical_wait_cycles += 1
        else:
            self._critical_wait_cycles = 0

        if output is not None and out_ready:
            output = self.dequeue()
        else:
            output = None

        if event is not None:
            if ready:
                accepted = self.enqueue(event)
            else:
                self.dropped_event = True

        return AERPriorityStep(
            output=output,
            accepted=accepted,
            ready=ready,
            occupancy=self.occupancy,
            dropped_event=self.dropped_event,
            critical_deadline_violation=self.critical_deadline_violation,
        )

    def drain(self) -> list[AERPriorityEvent]:
        """Drain the queue according to the strict-priority contract."""

        outputs: list[AERPriorityEvent] = []
        while True:
            event = self.dequeue()
            if event is None:
                return outputs
            outputs.append(event)

    def extend(self, events: Iterable[AERPriorityEvent]) -> int:
        """Enqueue all events that fit and return the accepted count."""

        accepted = 0
        for event in events:
            if self.enqueue(event):
                accepted += 1
        return accepted

    def _best_index(self) -> int:
        best_index = 0
        best_priority = self._queue[0].priority
        for index, event in enumerate(self._queue[1:], start=1):
            if event.priority < best_priority:
                best_index = index
                best_priority = event.priority
        return best_index


__all__ = [
    "AERPriorityEvent",
    "AERPriorityQueueReference",
    "AERPriorityStep",
]
