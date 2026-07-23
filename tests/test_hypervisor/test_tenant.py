# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor tenant-domain contracts

"""Verify tenant defaults, checkpoint integrity, and historical object identity."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import tenant as tenant_domain
from sc_neurocore.hypervisor.hypervisor import MigrationEngine
from sc_neurocore.hypervisor.tenant import (
    QoSPolicy,
    Tenant,
    TenantPriority,
    TenantState,
)


class TestTenantState:
    def test_checksum(self) -> None:
        ts = TenantState(
            neuron_voltages=np.array([1.0, 2.0]),
            synapse_weights=np.array([0.5, 0.3]),
            lfsr_state=42,
            timestep=100,
        )
        cs = ts.compute_checksum()
        assert len(cs) == 16

    def test_checksum_deterministic(self) -> None:
        ts = TenantState(lfsr_state=42, timestep=100)
        assert ts.compute_checksum() == ts.compute_checksum()

    def test_checksum_differs(self) -> None:
        ts1 = TenantState(lfsr_state=42, timestep=100)
        ts2 = TenantState(lfsr_state=43, timestep=100)
        assert ts1.compute_checksum() != ts2.compute_checksum()


def test_checkpoint_checksum_covers_spike_queue() -> None:
    first = TenantState(spike_queues=np.array([1, 0, 1], dtype=np.uint8))
    second = TenantState(spike_queues=np.array([1, 1, 1], dtype=np.uint8))

    assert first.compute_checksum() != second.compute_checksum()


def test_restore_rejects_tampered_spike_queue() -> None:
    state = TenantState(spike_queues=np.array([1, 0, 1], dtype=np.uint8))
    state.compute_checksum()
    assert state.spike_queues is not None
    state.spike_queues[1] = 1

    assert MigrationEngine().restore(Tenant("tenant", "Tenant"), state) is False


def test_tenant_defaults_use_independent_qos_policies() -> None:
    first = Tenant("first", "First")
    second = Tenant("second", "Second")

    assert first.priority is TenantPriority.NORMAL
    assert first.qos == QoSPolicy()
    assert first.qos is not second.qos
    assert first.region_id is None
    assert first.state is None
    assert first.active is False


def test_priority_values_preserve_scheduler_order() -> None:
    assert [priority.value for priority in TenantPriority] == [0, 1, 2, 3]


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.TenantPriority is tenant_domain.TenantPriority
    assert compatibility_surface.QoSPolicy is tenant_domain.QoSPolicy
    assert compatibility_surface.TenantState is tenant_domain.TenantState
    assert compatibility_surface.Tenant is tenant_domain.Tenant


def test_tenant_domain_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(tenant_domain.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}

    owned_names = {"TenantPriority", "QoSPolicy", "TenantState", "Tenant"}
    assert facade_classes.isdisjoint(owned_names)
    assert owner_classes == owned_names
    assert len(Path(tenant_domain.__file__).read_text(encoding="utf-8").splitlines()) <= 90
    assert len(Path(compatibility_surface.__file__).read_text(encoding="utf-8").splitlines()) <= 785
