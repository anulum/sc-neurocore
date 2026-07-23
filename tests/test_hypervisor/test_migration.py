# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor live-migration contracts

"""Verify sealed checkpoints, transactional transfer, and migration ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import migration as migration_owner
from sc_neurocore.hypervisor.isolation import BitstreamFirewall, FirewallRule
from sc_neurocore.hypervisor.migration import (
    MigrationEngine,
    MigrationRequest,
)
from sc_neurocore.hypervisor.region import HWRegion, RegionState
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


class TestMigrationEngine:
    def test_checkpoint(self) -> None:
        me = MigrationEngine()
        t = _tenant()
        t.state = TenantState(
            neuron_voltages=np.array([1.0]),
            lfsr_state=42,
            timestep=10,
        )
        state = me.checkpoint(t)
        assert state.checksum != ""

    def test_restore(self) -> None:
        me = MigrationEngine()
        t = _tenant()
        state = TenantState(lfsr_state=42, timestep=10)
        state.compute_checksum()
        assert me.restore(t, state) is True
        assert t.state is not None

    def test_checkpoint_initialises_missing_state(self) -> None:
        # A tenant that has never run has no state; checkpointing one must
        # materialise a fresh TenantState rather than dereference None.
        me = MigrationEngine()
        t = _tenant()
        assert t.state is None
        state = me.checkpoint(t)
        assert isinstance(state, TenantState)
        assert t.state is state

    def test_restore_rejects_tampered_state(self) -> None:
        # If the stored checksum no longer matches the recomputed one (the state
        # was altered after checkpointing), restore must refuse it.
        me = MigrationEngine()
        t = _tenant()
        state = TenantState(lfsr_state=42, timestep=10)
        state.compute_checksum()
        state.timestep = 99  # tamper after the checksum was sealed
        assert me.restore(t, state) is False

    def test_migrate_success(self) -> None:
        me = MigrationEngine()
        fw = BitstreamFirewall()
        t = _tenant()
        t.state = TenantState(lfsr_state=42, timestep=10)
        t.region_id = 0
        src = _region(rid=0)
        src.state = RegionState.ALLOCATED
        src.tenant_id = "t0"
        dst = _region(rid=1, base=0x5000_0000)
        result = me.migrate(t, src, dst, fw)
        assert result.success is True
        assert t.region_id == 1
        assert src.is_free
        assert dst.tenant_id == "t0"

    def test_migrate_target_busy(self) -> None:
        me = MigrationEngine()
        fw = BitstreamFirewall()
        t = _tenant()
        t.state = TenantState()
        t.region_id = 0
        src = _region(rid=0)
        dst = _region(rid=1)
        dst.state = RegionState.ALLOCATED
        result = me.migrate(t, src, dst, fw)
        assert result.success is False
        assert result.reason == "target_not_free"

    def test_migration_history(self) -> None:
        me = MigrationEngine()
        fw = BitstreamFirewall()
        t = _tenant()
        t.state = TenantState()
        t.region_id = 0
        src = _region(rid=0)
        dst = _region(rid=1, base=0x5000_0000)
        me.migrate(t, src, dst, fw)
        assert len(me.history) == 1


def test_restore_rejects_unsealed_state_without_mutation() -> None:
    tenant = _tenant()
    state = TenantState(lfsr_state=42, timestep=10)

    assert MigrationEngine().restore(tenant, state) is False
    assert tenant.state is None
    assert state.checksum == ""


def test_busy_target_failure_is_mutation_free() -> None:
    engine = MigrationEngine()
    tenant = _tenant()
    tenant.state = TenantState(lfsr_state=42, timestep=10)
    tenant.region_id = 0
    tenant.active = True
    source = _region(rid=0)
    source.state = RegionState.ALLOCATED
    source.tenant_id = tenant.tenant_id
    target = _region(rid=1, base=0x5000_0000)
    target.state = RegionState.ALLOCATED
    target.tenant_id = "other"
    firewall = BitstreamFirewall()
    source_rule = FirewallRule(tenant.tenant_id, source.axi_base_addr, source.axi_size)
    other_rule = FirewallRule("other", target.axi_base_addr, target.axi_size)
    firewall.add_rule(source_rule)
    firewall.add_rule(other_rule)
    original_state = tenant.state

    result = engine.migrate(tenant, source, target, firewall)

    assert result.success is False
    assert result.reason == "target_not_free"
    assert source.state == RegionState.ALLOCATED
    assert source.tenant_id == tenant.tenant_id
    assert target.state == RegionState.ALLOCATED
    assert target.tenant_id == "other"
    assert tenant.region_id == source.region_id
    assert tenant.active is True
    assert tenant.state is original_state
    assert original_state.checksum == ""
    assert firewall.rules == [source_rule, other_rule]


def test_request_default_reason() -> None:
    request = MigrationRequest("tenant", 0, 1)

    assert request.reason == "load_balance"


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.MigrationRequest is migration_owner.MigrationRequest
    assert compatibility_surface.MigrationResult is migration_owner.MigrationResult
    assert compatibility_surface.MigrationEngine is migration_owner.MigrationEngine


def test_migration_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(migration_owner.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}
    owned_names = {"MigrationRequest", "MigrationResult", "MigrationEngine"}

    assert facade_classes.isdisjoint(owned_names)
    assert owner_classes == owned_names
