# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor temporal scheduler

"""Allocate bounded cycle windows under supported tenant scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from sc_neurocore.hypervisor.tenant import Tenant, TenantPriority


class SchedulingPolicy(Enum):
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    FAIR_SHARE = "fair_share"
    EDF = "earliest_deadline_first"


@dataclass
class ScheduleSlot:
    """One time slot in the schedule."""

    tenant_id: str
    region_id: int
    start_cycle: int
    duration_cycles: int

    @property
    def end_cycle(self) -> int:
        return self.start_cycle + self.duration_cycles


class Scheduler:
    """Multi-tenant temporal scheduler with preemption.

    Supports priority-based, round-robin, fair-share, and EDF scheduling.
    """

    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.PRIORITY) -> None:
        self.policy = policy
        self.time_quantum_cycles: int = 10000
        self.schedule: List[ScheduleSlot] = []
        self.current_cycle: int = 0

    def generate_schedule(self, tenants: List[Tenant], num_cycles: int) -> List[ScheduleSlot]:
        """Generate a schedule for the given tenants."""
        if not tenants:
            return []
        active = [t for t in tenants if t.active and t.region_id is not None]
        if not active:
            return []

        if self.policy == SchedulingPolicy.ROUND_ROBIN:
            return self._round_robin(active, num_cycles)
        elif self.policy == SchedulingPolicy.PRIORITY:
            return self._priority(active, num_cycles)
        elif self.policy == SchedulingPolicy.FAIR_SHARE:
            return self._fair_share(active, num_cycles)
        elif self.policy == SchedulingPolicy.EDF:
            return self._edf(active, num_cycles)
        return []

    def _round_robin(self, tenants: List[Tenant], total: int) -> List[ScheduleSlot]:
        slots = []
        cycle = 0
        idx = 0
        while cycle < total:
            t = tenants[idx % len(tenants)]
            dur = min(self.time_quantum_cycles, total - cycle)
            slots.append(ScheduleSlot(t.tenant_id, t.region_id or 0, cycle, dur))
            cycle += dur
            idx += 1
        self.schedule = slots
        return slots

    def _priority(self, tenants: List[Tenant], total: int) -> List[ScheduleSlot]:
        sorted_t = sorted(tenants, key=lambda t: t.priority.value)
        slots = []
        cycle = 0
        for t in sorted_t:
            share = max(1, total // len(tenants))
            if t.priority == TenantPriority.REALTIME:
                share = total // 2  # Realtime gets 50%
            dur = min(share, total - cycle)
            if dur > 0:
                slots.append(ScheduleSlot(t.tenant_id, t.region_id or 0, cycle, dur))
                cycle += dur
        self.schedule = slots
        return slots

    def _fair_share(self, tenants: List[Tenant], total: int) -> List[ScheduleSlot]:
        total_share = sum(t.qos.min_compute_share for t in tenants)
        slots = []
        cycle = 0
        for t in tenants:
            frac = t.qos.min_compute_share / total_share if total_share > 0 else 1.0 / len(tenants)
            dur = int(total * frac)
            dur = min(dur, total - cycle)
            if dur > 0:
                slots.append(ScheduleSlot(t.tenant_id, t.region_id or 0, cycle, dur))
                cycle += dur
        self.schedule = slots
        return slots

    def _edf(self, tenants: List[Tenant], total: int) -> List[ScheduleSlot]:
        sorted_t = sorted(tenants, key=lambda t: t.qos.max_latency_us)
        return self._round_robin(sorted_t, total)
