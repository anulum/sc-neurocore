# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor hardware-region domain

"""Model physical accelerator regions, their health, and placement selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class RegionState(Enum):
    FREE = "free"
    ALLOCATED = "allocated"
    MIGRATING = "migrating"
    FAULTED = "faulted"


@dataclass
class HWRegion:
    """One isolated hardware region on the fabric."""

    region_id: int
    num_neurons: int
    num_synapses: int
    axi_base_addr: int
    axi_size: int
    die_id: int = 0
    state: RegionState = RegionState.FREE
    tenant_id: Optional[str] = None
    utilisation: float = 0.0

    @property
    def axi_end_addr(self) -> int:
        return self.axi_base_addr + self.axi_size

    @property
    def is_free(self) -> bool:
        return self.state == RegionState.FREE

    def contains_addr(self, addr: int) -> bool:
        return self.axi_base_addr <= addr < self.axi_end_addr


def select_region_multi_die(
    regions: Dict[int, HWRegion],
    min_neurons: int,
    preferred_die: Optional[int] = None,
) -> Optional[int]:
    """Select best free region, preferring a specific die."""
    candidates = [
        (rid, r) for rid, r in regions.items() if r.is_free and r.num_neurons >= min_neurons
    ]
    if not candidates:
        return None

    if preferred_die is not None:
        on_die = [(rid, r) for rid, r in candidates if r.die_id == preferred_die]
        if on_die:
            return min(on_die, key=lambda x: x[1].num_neurons)[0]

    return min(candidates, key=lambda x: x[1].num_neurons)[0]


@dataclass
class RegionHealth:
    """Health score with degradation model."""

    region_id: int
    error_count: int = 0
    temperature_c: float = 25.0
    age_hours: float = 0.0

    @property
    def health_score(self) -> float:
        """0.0 = dead, 1.0 = perfect."""
        temp_pen = max(0, (self.temperature_c - 85)) * 0.01
        age_pen = self.age_hours / 100_000 * 0.1
        err_pen = min(self.error_count * 0.05, 0.5)
        return max(0.0, 1.0 - temp_pen - age_pen - err_pen)

    @property
    def is_degraded(self) -> bool:
        return self.health_score < 0.8

    def record_error(self) -> None:
        self.error_count += 1
