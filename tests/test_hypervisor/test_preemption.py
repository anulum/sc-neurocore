# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor preemption contracts

"""Verify victim capture, region handoff, restore, and definition ownership."""

from __future__ import annotations

import ast
from pathlib import Path

from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import preemption as preemption_owner
from sc_neurocore.hypervisor.preemption import PreemptionManager
from sc_neurocore.hypervisor.region import HWRegion
from sc_neurocore.hypervisor.tenant import Tenant, TenantPriority, TenantState


def _region(rid: int = 0, neurons: int = 1024, base: int = 0x4000_0000) -> HWRegion:
    return HWRegion(
        region_id=rid,
        num_neurons=neurons,
        num_synapses=neurons * 16,
        axi_base_addr=base,
        axi_size=0x1000,
        die_id=0,
    )


def _tenant(
    tid: str = "t0", name: str = "test", prio: TenantPriority = TenantPriority.NORMAL
) -> Tenant:
    return Tenant(tenant_id=tid, name=name, priority=prio)


class TestPreemptionManager:
    def test_preempt(self) -> None:
        pm = PreemptionManager()
        victim = _tenant("v")
        victim.active = True
        victim.region_id = 0
        victim.state = TenantState(lfsr_state=42)
        preemptor = _tenant("p")
        region = _region(0)
        region.tenant_id = "v"
        evt = pm.preempt(victim, preemptor, region, cycle=1000)
        assert evt.state_saved is True
        assert victim.active is False
        assert preemptor.region_id == 0

    def test_restore_preempted(self) -> None:
        pm = PreemptionManager()
        victim = _tenant("v")
        victim.state = TenantState(lfsr_state=99)
        preemptor = _tenant("p")
        region = _region(0)
        pm.preempt(victim, preemptor, region, cycle=0)
        assert pm.restore_preempted(victim) is True
        assert victim.state.lfsr_state == 99

    def test_restore_missing(self) -> None:
        pm = PreemptionManager()
        t = _tenant("x")
        assert pm.restore_preempted(t) is False


def test_preempt_without_runtime_state_records_unsaved_handoff() -> None:
    manager = PreemptionManager()
    victim = _tenant("victim")
    victim.active = True
    victim.region_id = 3
    preemptor = _tenant("preemptor")
    region = _region(3)
    region.tenant_id = victim.tenant_id

    event = manager.preempt(victim, preemptor, region, cycle=77)

    assert event.state_saved is False
    assert manager.saved_states == {}
    assert manager.events == [event]
    assert victim.active is False
    assert victim.region_id is None
    assert region.tenant_id == preemptor.tenant_id
    assert preemptor.region_id == region.region_id
    assert preemptor.active is True


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.PreemptionEvent is preemption_owner.PreemptionEvent
    assert compatibility_surface.PreemptionManager is preemption_owner.PreemptionManager


def test_preemption_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(preemption_owner.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}
    owned_names = {"PreemptionEvent", "PreemptionManager"}

    assert facade_classes.isdisjoint(owned_names)
    assert owner_classes == owned_names
