# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_hypervisor.py

from __future__ import annotations

from sc_neurocore.hypervisor.hypervisor import (
    HWRegion,
    Hypervisor,
    HypervisorConfig,
    MigrationThrottle,
    RegionState,
    Tenant,
    TenantPriority,
    TenantState,
    admission_check,
)


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


__all__ = [
    "HWRegion",
    "Hypervisor",
    "HypervisorConfig",
    "MigrationThrottle",
    "RegionState",
    "Tenant",
    "TenantPriority",
    "TenantState",
    "admission_check",
    "_region",
    "_tenant",
]
