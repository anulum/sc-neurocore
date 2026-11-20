# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor runtime QoS monitoring

"""Measure tenant throughput and record runtime SLA violations."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sc_neurocore.hypervisor.tenant import Tenant


@dataclass
class BandwidthMeter:
    """Per-tenant spike throughput over a window of cycles.

    A sample is the spikes counted in the interval that ends at its cycle. The
    meter knows spikes and cycles only; Mbps need the caller's bits per spike
    and clock.
    """

    window_cycles: int = 100_000
    _counters: Dict[str, List[int]] = field(default_factory=dict)
    _timestamps: Dict[str, List[int]] = field(default_factory=dict)

    def record(self, tenant_id: str, spike_count: int, cycle: int) -> None:
        """Record the spikes a tenant produced in the interval that ends at ``cycle``.

        A tenant's cycles must increase: the window and the span of
        :meth:`throughput` are measured along them. Because they increase, each
        later window's base is this one's or a later sample, so the samples
        before this window's base can never be read again; they are dropped
        here, and a tenant's history stays within one window and its base.
        ``window_cycles`` is therefore set before recording, not enlarged later.

        Raises
        ------
        ValueError
            If ``cycle`` is not later than the tenant's last recorded cycle; the
            history is left unchanged.
        """
        if tenant_id not in self._counters:
            self._counters[tenant_id] = []
            self._timestamps[tenant_id] = []
        stamps = self._timestamps[tenant_id]
        if stamps and cycle <= stamps[-1]:
            raise ValueError(
                f"cycle {cycle} is not later than tenant {tenant_id!r}'s last cycle {stamps[-1]}"
            )
        counts = self._counters[tenant_id]
        counts.append(spike_count)
        stamps.append(cycle)
        base = self._base(stamps)
        if base:
            del counts[:base]
            del stamps[:base]

    def _base(self, stamps: List[int]) -> int:
        """Return the index of the latest sample at or before ``last − window_cycles``, else 0."""
        return max(bisect.bisect_right(stamps, stamps[-1] - self.window_cycles) - 1, 0)

    def throughput(self, tenant_id: str) -> float:
        """Return spikes per cycle over the window.

        The base is the latest sample at or before ``last − window_cycles``,
        or the first sample. The rate is the spikes of the samples after the
        base divided by the cycles from the base to the last sample. With fewer
        than two samples, or no cycles between the base and the last sample,
        no interval is known and the rate is 0.0.
        """
        entries = self._counters.get(tenant_id, [])
        if len(entries) < 2:
            return 0.0
        ts = self._timestamps[tenant_id]
        base = self._base(ts)
        span = ts[-1] - ts[base]
        if span <= 0:
            return 0.0
        return sum(entries[base + 1 :]) / span

    def throughput_mbps(self, tenant_id: str, *, bits_per_spike: int, clock_hz: float) -> float:
        """Return the window's throughput in Mbps.

        Parameters
        ----------
        tenant_id : str
            Metered tenant.
        bits_per_spike : int
            Bits one spike event occupies on the link; a positive integer.
        clock_hz : float
            Clock of the cycles the samples were recorded in; finite and positive, not a bool.

        Returns
        -------
        float
            Spikes per cycle × ``bits_per_spike`` × ``clock_hz`` / 1e6.

        Raises
        ------
        ValueError
            If ``bits_per_spike`` is not a positive integer, or ``clock_hz`` is a
            bool or not finite and positive.
        """
        if isinstance(bits_per_spike, bool) or not isinstance(bits_per_spike, int):
            raise ValueError(f"bits_per_spike must be a positive integer, got {bits_per_spike!r}")
        if bits_per_spike < 1:
            raise ValueError(f"bits_per_spike must be a positive integer, got {bits_per_spike!r}")
        if isinstance(clock_hz, bool) or not (math.isfinite(clock_hz) and clock_hz > 0.0):
            raise ValueError(f"clock_hz must be finite and positive, got {clock_hz!r}")
        return self.throughput(tenant_id) * bits_per_spike * clock_hz / 1e6

    def exceeds_quota(
        self, tenant_id: str, max_mbps: float, *, bits_per_spike: int, clock_hz: float
    ) -> bool:
        """Return whether the window's throughput is above a quota in Mbps.

        Parameters
        ----------
        tenant_id : str
            Metered tenant.
        max_mbps : float
            Quota; finite and zero or more, not a bool.
        bits_per_spike, clock_hz
            As for :meth:`throughput_mbps`.

        Raises
        ------
        ValueError
            If ``max_mbps`` is a bool, not finite or negative, or a unit is
            refused by :meth:`throughput_mbps`.
        """
        if isinstance(max_mbps, bool) or not (math.isfinite(max_mbps) and max_mbps >= 0.0):
            raise ValueError(f"max_mbps must be finite and zero or more, got {max_mbps!r}")
        measured = self.throughput_mbps(tenant_id, bits_per_spike=bits_per_spike, clock_hz=clock_hz)
        return measured > max_mbps


@dataclass
class SLAViolation:
    """One recorded SLA violation: tenant, metric, measured value, limit and cycle."""

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
        """Record and return a violation when a latency exceeds the tenant's QoS limit."""
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
        """Record and return a violation when a bandwidth in Mbps exceeds the tenant's QoS limit.

        :meth:`BandwidthMeter.throughput_mbps` gives the measured Mbps.
        """
        if measured_mbps > tenant.qos.max_bandwidth_mbps:
            v = SLAViolation(
                tenant.tenant_id, "bandwidth", measured_mbps, tenant.qos.max_bandwidth_mbps, cycle
            )
            self.violations.append(v)
            return v
        return None

    @property
    def total_violations(self) -> int:
        """Return the number of violations recorded."""
        return len(self.violations)

    def violations_for(self, tenant_id: str) -> List[SLAViolation]:
        """Return the violations recorded for one tenant."""
        return [v for v in self.violations if v.tenant_id == tenant_id]
