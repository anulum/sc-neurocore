# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor resource accounting

"""Record per-tenant accelerator usage and calculate metered invoices."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class UsageRecord:
    """One billing record."""

    tenant_id: str
    cycles_used: int
    spikes_processed: int
    timestamp_ns: int


class ResourceAccounting:
    """Tracks per-tenant resource usage for metered billing."""

    def __init__(self) -> None:
        self.records: List[UsageRecord] = []
        self._totals: Dict[str, Dict[str, int]] = {}

    def record(self, tenant_id: str, cycles: int, spikes: int) -> None:
        r = UsageRecord(tenant_id, cycles, spikes, time.time_ns())
        self.records.append(r)
        if tenant_id not in self._totals:
            self._totals[tenant_id] = {"cycles": 0, "spikes": 0}
        self._totals[tenant_id]["cycles"] += cycles
        self._totals[tenant_id]["spikes"] += spikes

    def total_cycles(self, tenant_id: str) -> int:
        return self._totals.get(tenant_id, {}).get("cycles", 0)

    def total_spikes(self, tenant_id: str) -> int:
        return self._totals.get(tenant_id, {}).get("spikes", 0)

    def invoice(self, tenant_id: str, cost_per_cycle: float = 1e-6) -> float:
        """Compute billing amount."""
        return self.total_cycles(tenant_id) * cost_per_cycle
