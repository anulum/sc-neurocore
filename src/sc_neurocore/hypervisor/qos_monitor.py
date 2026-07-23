# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor runtime QoS monitoring

"""Measure tenant throughput and record runtime SLA violations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sc_neurocore.hypervisor.tenant import Tenant


@dataclass
class BandwidthMeter:
    """Per-tenant throughput metering (spikes/cycles per window)."""

    window_cycles: int = 100_000
    _counters: Dict[str, List[int]] = field(default_factory=dict)
    _timestamps: Dict[str, List[int]] = field(default_factory=dict)

    def record(self, tenant_id: str, spike_count: int, cycle: int) -> None:
        if tenant_id not in self._counters:
            self._counters[tenant_id] = []
            self._timestamps[tenant_id] = []
        self._counters[tenant_id].append(spike_count)
        self._timestamps[tenant_id].append(cycle)

    def throughput(self, tenant_id: str) -> float:
        """Spikes per cycle (averaged over window)."""
        if tenant_id not in self._counters or not self._counters[tenant_id]:
            return 0.0
        entries = self._counters[tenant_id]
        total_spikes = sum(entries[-100:])
        if len(entries) < 2:
            return float(total_spikes)
        ts = self._timestamps[tenant_id]
        span = max(1, ts[-1] - ts[max(0, len(ts) - 100)])
        return total_spikes / span

    def exceeds_quota(self, tenant_id: str, max_mbps: float) -> bool:
        return self.throughput(tenant_id) > max_mbps


@dataclass
class SLAViolation:
    """One SLA violation."""

    tenant_id: str
    metric: str  # "latency", "bandwidth", "compute_share"
    measured: float
    limit: float
    cycle: int


class SLAMonitor:
    """Monitors per-tenant QoS compliance and detects violations."""

    def __init__(self) -> None:
        self.violations: List[SLAViolation] = []

    def check_latency(
        self, tenant: Tenant, measured_us: float, cycle: int
    ) -> Optional[SLAViolation]:
        if measured_us > tenant.qos.max_latency_us:
            v = SLAViolation(
                tenant.tenant_id, "latency", measured_us, tenant.qos.max_latency_us, cycle
            )
            self.violations.append(v)
            return v
        return None

    def check_bandwidth(
        self, tenant: Tenant, measured_mbps: float, cycle: int
    ) -> Optional[SLAViolation]:
        if measured_mbps > tenant.qos.max_bandwidth_mbps:
            v = SLAViolation(
                tenant.tenant_id, "bandwidth", measured_mbps, tenant.qos.max_bandwidth_mbps, cycle
            )
            self.violations.append(v)
            return v
        return None

    @property
    def total_violations(self) -> int:
        return len(self.violations)

    def violations_for(self, tenant_id: str) -> List[SLAViolation]:
        return [v for v in self.violations if v.tenant_id == tenant_id]
