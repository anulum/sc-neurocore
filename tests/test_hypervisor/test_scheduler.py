# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor scheduler contracts

"""Verify temporal scheduling policies and historical object identity."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, cast

from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import scheduler as scheduler_owner
from sc_neurocore.hypervisor.scheduler import Scheduler, SchedulingPolicy
from sc_neurocore.hypervisor.tenant import Tenant, TenantPriority


def _tenant(
    tid: str = "t0",
    name: str = "test",
    prio: TenantPriority = TenantPriority.NORMAL,
) -> Tenant:
    return Tenant(tenant_id=tid, name=name, priority=prio)


def _active_tenant(
    tenant_id: str,
    *,
    priority: TenantPriority = TenantPriority.NORMAL,
    region_id: int = 0,
) -> Tenant:
    tenant = _tenant(tenant_id, prio=priority)
    tenant.active = True
    tenant.region_id = region_id
    return tenant


class TestScheduler:
    def _tenants(self) -> List[Tenant]:
        t0 = _tenant("t0", prio=TenantPriority.NORMAL)
        t0.active = True
        t0.region_id = 0
        t1 = _tenant("t1", prio=TenantPriority.HIGH)
        t1.active = True
        t1.region_id = 1
        return [t0, t1]

    def test_round_robin(self) -> None:
        sched = Scheduler(SchedulingPolicy.ROUND_ROBIN)
        sched.time_quantum_cycles = 100
        slots = sched.generate_schedule(self._tenants(), 400)
        assert len(slots) == 4
        assert slots[0].tenant_id != slots[1].tenant_id

    def test_priority(self) -> None:
        sched = Scheduler(SchedulingPolicy.PRIORITY)
        slots = sched.generate_schedule(self._tenants(), 1000)
        assert len(slots) > 0
        # Higher priority tenant should get scheduled first
        assert slots[0].tenant_id == "t1"

    def test_fair_share(self) -> None:
        tenants = self._tenants()
        tenants[0].qos.min_compute_share = 0.3
        tenants[1].qos.min_compute_share = 0.7
        sched = Scheduler(SchedulingPolicy.FAIR_SHARE)
        slots = sched.generate_schedule(tenants, 1000)
        total_t1 = sum(s.duration_cycles for s in slots if s.tenant_id == "t1")
        assert total_t1 > 500

    def test_edf(self) -> None:
        tenants = self._tenants()
        tenants[0].qos.max_latency_us = 1000
        tenants[1].qos.max_latency_us = 100
        sched = Scheduler(SchedulingPolicy.EDF)
        slots = sched.generate_schedule(tenants, 1000)
        assert slots[0].tenant_id == "t1"  # Lower latency first

    def test_empty_tenants(self) -> None:
        sched = Scheduler()
        assert sched.generate_schedule([], 1000) == []

    def test_priority_realtime_gets_half(self) -> None:
        # A realtime tenant is granted a fixed 50% slice under priority
        # scheduling rather than the even per-tenant share.
        rt = _tenant("rt", prio=TenantPriority.REALTIME)
        rt.active = True
        rt.region_id = 0
        normal = _tenant("nm", prio=TenantPriority.NORMAL)
        normal.active = True
        normal.region_id = 1
        sched = Scheduler(SchedulingPolicy.PRIORITY)
        slots = sched.generate_schedule([rt, normal], 1000)
        rt_slot = next(s for s in slots if s.tenant_id == "rt")
        assert rt_slot.duration_cycles == 500

    def test_slot_continuity(self) -> None:
        sched = Scheduler(SchedulingPolicy.ROUND_ROBIN)
        sched.time_quantum_cycles = 100
        slots = sched.generate_schedule(self._tenants(), 400)
        for i in range(1, len(slots)):
            assert slots[i].start_cycle == slots[i - 1].end_cycle


def test_inactive_tenants_are_not_scheduled() -> None:
    tenant = Tenant("inactive", "Inactive")

    assert Scheduler().generate_schedule([tenant], 1000) == []


def test_zero_cycle_budget_produces_no_slots() -> None:
    tenants = [_active_tenant("t0"), _active_tenant("t1", region_id=1)]

    assert Scheduler(SchedulingPolicy.PRIORITY).generate_schedule(tenants, 0) == []
    assert Scheduler(SchedulingPolicy.FAIR_SHARE).generate_schedule(tenants, 0) == []


def test_zero_fair_share_weights_fall_back_to_even_allocation() -> None:
    tenants = [_active_tenant("t0"), _active_tenant("t1", region_id=1)]
    for tenant in tenants:
        tenant.qos.min_compute_share = 0.0

    slots = Scheduler(SchedulingPolicy.FAIR_SHARE).generate_schedule(tenants, 100)

    assert [slot.duration_cycles for slot in slots] == [50, 50]


def test_unknown_policy_fails_closed() -> None:
    scheduler = Scheduler()
    scheduler.policy = cast(SchedulingPolicy, "corrupt-policy")

    assert scheduler.generate_schedule([_active_tenant("t0")], 100) == []


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.SchedulingPolicy is scheduler_owner.SchedulingPolicy
    assert compatibility_surface.ScheduleSlot is scheduler_owner.ScheduleSlot
    assert compatibility_surface.Scheduler is scheduler_owner.Scheduler


def test_scheduler_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(scheduler_owner.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}

    owned_names = {"SchedulingPolicy", "ScheduleSlot", "Scheduler"}
    assert facade_classes.isdisjoint(owned_names)
    assert owner_classes == owned_names
    assert len(Path(scheduler_owner.__file__).read_text(encoding="utf-8").splitlines()) <= 125
    assert len(Path(compatibility_surface.__file__).read_text(encoding="utf-8").splitlines()) <= 689
