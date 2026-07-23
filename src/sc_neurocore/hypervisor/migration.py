# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor live migration

"""Checkpoint, verify, and transfer tenants between hardware regions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

from sc_neurocore.hypervisor.isolation import BitstreamFirewall, FirewallRule
from sc_neurocore.hypervisor.region import HWRegion, RegionState
from sc_neurocore.hypervisor.tenant import Tenant, TenantState


@dataclass
class MigrationRequest:
    """Request to migrate a tenant between regions."""

    tenant_id: str
    source_region: int
    target_region: int
    reason: str = "load_balance"


@dataclass
class MigrationResult:
    """Result of a migration attempt."""

    success: bool
    tenant_id: str
    source_region: int
    target_region: int
    state_checksum: str = ""
    duration_ns: int = 0
    reason: str = ""


class MigrationEngine:
    """Live migration of tenants between hardware regions.

    Migration steps:
    1. Pause tenant on source region
    2. Checkpoint state (voltages, weights, LFSR, spike queues)
    3. Verify checkpoint integrity (SHA-256)
    4. Restore state on target region
    5. Update firewall rules
    6. Resume tenant
    """

    def __init__(self) -> None:
        self.history: List[MigrationResult] = []

    def checkpoint(self, tenant: Tenant) -> TenantState:
        """Checkpoint tenant state for migration."""
        if tenant.state is None:
            tenant.state = TenantState()
        tenant.state.compute_checksum()
        return tenant.state

    def restore(self, tenant: Tenant, state: TenantState) -> bool:
        """Restore a sealed checkpointed state to a tenant."""
        verify = state.checksum
        if not verify:
            return False
        recomputed = state.compute_checksum()
        if verify != recomputed:
            return False
        tenant.state = state
        return True

    def migrate(
        self,
        tenant: Tenant,
        source: HWRegion,
        target: HWRegion,
        firewall: BitstreamFirewall,
    ) -> MigrationResult:
        """Execute live migration."""
        start = time.time_ns()

        # Reject before mutating source, tenant checkpoint, or firewall.
        if not target.is_free:
            result = MigrationResult(
                False,
                tenant.tenant_id,
                source.region_id,
                target.region_id,
                reason="target_not_free",
            )
            self.history.append(result)
            return result

        # 1. Checkpoint
        state = self.checkpoint(tenant)
        checksum = state.checksum

        # 2. Free source
        source.state = RegionState.FREE
        source.tenant_id = None

        # 3. Allocate target
        target.state = RegionState.ALLOCATED
        target.tenant_id = tenant.tenant_id

        # 4. Update firewall
        firewall.remove_tenant_rules(tenant.tenant_id)
        firewall.add_rule(
            FirewallRule(
                tenant.tenant_id,
                target.axi_base_addr,
                target.axi_size,
            )
        )

        # 5. Restore state
        success = self.restore(tenant, state)
        tenant.region_id = target.region_id

        elapsed = time.time_ns() - start
        result = MigrationResult(
            success,
            tenant.tenant_id,
            source.region_id,
            target.region_id,
            checksum,
            elapsed,
        )
        self.history.append(result)
        return result
