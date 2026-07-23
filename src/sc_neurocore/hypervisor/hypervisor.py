# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuromorphic Hypervisor (Multi-Tenant FPGA/ASIC Scheduler)

"""Multi-tenant neuromorphic hypervisor with hard isolation.

Enables multiple SC networks to share the same FPGA/ASIC fabric with:

- **Spatial partitioning**: Non-overlapping hardware regions per tenant
- **Temporal partitioning**: Time-sliced scheduling with preemption
- **Bitstream firewalls**: AXI address-range isolation preventing
  cross-tenant memory/register access
- **Dynamic migration**: Live migration of tenants between dies/regions
  with state checkpoint/restore
- **QoS enforcement**: Per-tenant bandwidth, latency, and compute quotas

Architecture:

    ┌──────────────────────────────────────┐
    │          Hypervisor Scheduler         │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐│
    │  │ Tenant A │ │ Tenant B │ │ Tenant C ││
    │  │ (BCI)   │ │ (Vision) │ │ (Audio) ││
    │  └────┬────┘ └────┬────┘ └────┬────┘│
    │       │           │           │      │
    │  ┌────▼───────────▼───────────▼────┐ │
    │  │     Bitstream Firewall Layer     │ │
    │  └────┬───────────┬───────────┬────┘ │
    │       │           │           │      │
    │  ┌────▼────┐ ┌────▼────┐ ┌────▼────┐│
    │  │Region 0 │ │Region 1 │ │Region 2 ││
    │  │AXI+AER  │ │AXI+AER  │ │AXI+AER  ││
    │  └─────────┘ └─────────┘ └─────────┘│
    └──────────────────────────────────────┘

Compatible with:
- ``hdl/sc_axis_interface.v`` — AXI-Stream wrappers
- ``hdl/sc_aer_router.v`` — AER spike routing
- ``chiplet_gen/`` — multi-die topology
- ``dynamic_adaptation/`` — runtime adaptation hooks
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sc_neurocore.hypervisor.accounting import (
    ResourceAccounting as ResourceAccounting,
    UsageRecord as UsageRecord,
)
from sc_neurocore.hypervisor.audit import (
    AuditEntry as AuditEntry,
    AuditEventType as AuditEventType,
    SecurityAuditLog as SecurityAuditLog,
)
from sc_neurocore.hypervisor.isolation import (
    BitstreamFirewall as BitstreamFirewall,
    FirewallRule as FirewallRule,
    verify_isolation as verify_isolation,
)
from sc_neurocore.hypervisor.migration import (
    MigrationEngine as MigrationEngine,
    MigrationRequest as MigrationRequest,
    MigrationResult as MigrationResult,
)
from sc_neurocore.hypervisor.preemption import (
    PreemptionEvent as PreemptionEvent,
    PreemptionManager as PreemptionManager,
)
from sc_neurocore.hypervisor.qos_monitor import (
    BandwidthMeter as BandwidthMeter,
    SLAMonitor as SLAMonitor,
    SLAViolation as SLAViolation,
)
from sc_neurocore.hypervisor.region import (
    HWRegion as HWRegion,
    RegionHealth as RegionHealth,
    RegionState as RegionState,
    select_region_multi_die as select_region_multi_die,
)
from sc_neurocore.hypervisor.scheduler import (
    ScheduleSlot as ScheduleSlot,
    Scheduler as Scheduler,
    SchedulingPolicy as SchedulingPolicy,
)
from sc_neurocore.hypervisor.tenant import (
    QoSPolicy as QoSPolicy,
    Tenant as Tenant,
    TenantPriority as TenantPriority,
    TenantState as TenantState,
)

# ── Hypervisor ──────────────────────────────────────────────────────


@dataclass
class HypervisorConfig:
    """Hypervisor configuration."""

    max_tenants: int = 16
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY
    time_quantum_cycles: int = 10000
    migration_cooldown_ns: int = 1_000_000_000  # 1s
    enable_firewall: bool = True


class Hypervisor:
    """Multi-tenant neuromorphic hypervisor.

    Manages tenant lifecycle, hardware allocation, scheduling,
    firewall enforcement, and live migration.
    """

    def __init__(self, config: Optional[HypervisorConfig] = None) -> None:
        self.config = config or HypervisorConfig()
        self.regions: Dict[int, HWRegion] = {}
        self.tenants: Dict[str, Tenant] = {}
        self.scheduler = Scheduler(self.config.scheduling_policy)
        self.scheduler.time_quantum_cycles = self.config.time_quantum_cycles
        self.firewall = BitstreamFirewall()
        self.migration_engine = MigrationEngine()
        self.uptime_ns: int = 0

    def add_region(self, region: HWRegion) -> None:
        """Register a hardware region."""
        self.regions[region.region_id] = region

    def register_tenant(self, tenant: Tenant) -> bool:
        """Register a new tenant."""
        if len(self.tenants) >= self.config.max_tenants:
            return False
        if tenant.tenant_id in self.tenants:
            return False
        tenant.created_ns = time.time_ns()
        self.tenants[tenant.tenant_id] = tenant
        return True

    def allocate(self, tenant_id: str) -> Optional[int]:
        """Allocate a free region to a tenant."""
        tenant = self.tenants.get(tenant_id)
        if tenant is None:
            return None

        # Find a free region that fits the QoS
        for rid, region in self.regions.items():
            if not region.is_free:
                continue
            if region.num_neurons < tenant.qos.max_neurons:
                continue
            # Allocate
            region.state = RegionState.ALLOCATED
            region.tenant_id = tenant_id
            tenant.region_id = rid
            tenant.active = True

            # Set up firewall
            if self.config.enable_firewall:
                self.firewall.add_rule(
                    FirewallRule(
                        tenant_id,
                        region.axi_base_addr,
                        region.axi_size,
                    )
                )
            return rid
        return None

    def deallocate(self, tenant_id: str) -> bool:
        """Release a tenant's hardware region."""
        tenant = self.tenants.get(tenant_id)
        if tenant is None or tenant.region_id is None:
            return False

        region = self.regions.get(tenant.region_id)
        if region is not None:
            region.state = RegionState.FREE
            region.tenant_id = None

        self.firewall.remove_tenant_rules(tenant_id)
        tenant.region_id = None
        tenant.active = False
        return True

    def remove_tenant(self, tenant_id: str) -> bool:
        """Remove a tenant entirely."""
        self.deallocate(tenant_id)
        return self.tenants.pop(tenant_id, None) is not None

    def schedule(self, num_cycles: int) -> List[ScheduleSlot]:
        """Generate a schedule for active tenants."""
        active = [t for t in self.tenants.values() if t.active]
        return self.scheduler.generate_schedule(active, num_cycles)

    def migrate(self, tenant_id: str, target_region_id: int) -> MigrationResult:
        """Migrate a tenant to a different region."""
        tenant = self.tenants.get(tenant_id)
        if tenant is None or tenant.region_id is None:
            return MigrationResult(False, tenant_id or "", -1, target_region_id, reason="not_found")

        source = self.regions.get(tenant.region_id)
        target = self.regions.get(target_region_id)
        if source is None or target is None:
            return MigrationResult(False, tenant_id, -1, target_region_id, reason="invalid_region")

        return self.migration_engine.migrate(tenant, source, target, self.firewall)

    def check_access(self, tenant_id: str, addr: int, is_write: bool = False) -> bool:
        """Check if a tenant can access an address (firewall)."""
        if not self.config.enable_firewall:
            return True
        return self.firewall.check_access(tenant_id, addr, is_write)

    def status(self) -> Dict[str, Any]:
        """Get hypervisor status."""
        free_regions = sum(1 for r in self.regions.values() if r.is_free)
        active_tenants = sum(1 for t in self.tenants.values() if t.active)
        return {
            "total_regions": len(self.regions),
            "free_regions": free_regions,
            "total_tenants": len(self.tenants),
            "active_tenants": active_tenants,
            "firewall_violations": self.firewall.violation_count,
            "migrations": len(self.migration_engine.history),
            "scheduling_policy": self.config.scheduling_policy.value,
        }

    def tenant_report(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get a report for one tenant."""
        t = self.tenants.get(tenant_id)
        if t is None:
            return None
        return {
            "tenant_id": t.tenant_id,
            "name": t.name,
            "priority": t.priority.value,
            "region_id": t.region_id,
            "active": t.active,
            "total_spikes": t.total_spikes,
            "total_cycles": t.total_cycles,
            "qos_bandwidth_mbps": t.qos.max_bandwidth_mbps,
            "qos_latency_us": t.qos.max_latency_us,
        }

    def compute_utilisation(self) -> Dict[int, float]:
        """Compute utilisation fraction per region."""
        result = {}
        for rid, region in self.regions.items():
            if region.is_free:
                result[rid] = 0.0
            elif region.tenant_id:
                tenant = self.tenants.get(region.tenant_id)
                if tenant and tenant.qos:
                    result[rid] = min(1.0, tenant.qos.max_neurons / max(region.num_neurons, 1))
                else:
                    result[rid] = 1.0
            else:
                result[rid] = 0.0
        return result

    def check_overcommit(self) -> bool:
        """Check if total tenant QoS exceeds fabric capacity."""
        total_neurons_needed = sum(t.qos.max_neurons for t in self.tenants.values() if t.active)
        total_neurons_available = sum(r.num_neurons for r in self.regions.values())
        return total_neurons_needed > total_neurons_available

    def get_faulted_regions(self) -> List[int]:
        """List regions in FAULTED state."""
        return [rid for rid, r in self.regions.items() if r.state == RegionState.FAULTED]

    def mark_region_faulted(self, region_id: int) -> bool:
        """Mark a region as faulted and evict its tenant."""
        region = self.regions.get(region_id)
        if region is None:
            return False
        if region.tenant_id:
            self.deallocate(region.tenant_id)
        region.state = RegionState.FAULTED
        return True


# ── Admission Control (Gap 6) ───────────────────────────────────────


def admission_check(
    tenant: Tenant,
    regions: Dict[int, HWRegion],
    existing_tenants: Dict[str, Tenant],
) -> Tuple[bool, str]:
    """Check if a new tenant can be admitted without overcommitting."""
    required = tenant.qos.max_neurons
    free_capacity = sum(r.num_neurons for r in regions.values() if r.is_free)

    if required > free_capacity:
        return False, f"insufficient_neurons: need={required}, free={free_capacity}"

    if any(r.num_neurons >= required for r in regions.values() if r.is_free):
        return True, "admitted"

    return False, "no_single_region_large_enough"


# ── Migration Throttle (Gap 10) ─────────────────────────────────────


@dataclass
class MigrationThrottle:
    """Rate-limits migration requests to prevent storms."""

    max_per_window: int = 5
    window_ns: int = 10_000_000_000  # 10s
    _timestamps: List[int] = field(default_factory=list)

    def allow(self) -> bool:
        """Check if a migration is allowed under the rate limit."""
        now = time.time_ns()
        cutoff = now - self.window_ns
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        return len(self._timestamps) < self.max_per_window

    def record(self) -> None:
        self._timestamps.append(time.time_ns())

    @property
    def recent_count(self) -> int:
        now = time.time_ns()
        cutoff = now - self.window_ns
        return sum(1 for t in self._timestamps if t > cutoff)
