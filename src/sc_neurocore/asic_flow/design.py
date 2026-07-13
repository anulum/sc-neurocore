# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC design and stochastic-computing synthesis parameters

"""Define physical and stochastic-computing parameters for ASIC decks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class SCASICOptimisationConfig:
    """SC-specific synthesis settings for stochastic neuromorphic datapaths."""

    share_stochastic_counters: bool = True
    reduce_constant_widths: bool = True
    preserve_lfsr_hierarchy: bool = True
    max_fanout: int = 16
    abc_delay_margin: float = 0.90

    def yosys_passes(self) -> List[str]:
        """Return the ordered Yosys passes selected for the SC datapath."""
        passes: List[str] = []
        if self.reduce_constant_widths:
            passes.append("wreduce")
        if self.share_stochastic_counters:
            passes.extend(["share", "opt_share"])
        passes.append("opt_clean -purge")
        return passes


@dataclass
class DesignParams:
    """ASIC design parameters."""

    top_module: str = "sc_neurocore_top"
    clock_name: str = "clk"
    reset_name: str = "rst_n"
    reset_active_low: bool = True
    target_frequency_mhz: float = 100.0
    die_area_um: Tuple[float, float, float, float] = (0, 0, 500, 500)
    core_area_um: Tuple[float, float, float, float] = (20, 20, 480, 480)
    utilisation: float = 0.5
    aspect_ratio: float = 1.0
    io_margin_um: float = 20.0
    power_nets: List[str] = field(default_factory=lambda: ["VDD", "VSS"])
    rtl_files: List[str] = field(default_factory=list)
    sc_optimisation: SCASICOptimisationConfig = field(default_factory=SCASICOptimisationConfig)

    @property
    def clock_period_ns(self) -> float:
        """Return the target clock period in nanoseconds."""
        return 1000.0 / self.target_frequency_mhz

    @property
    def die_width_um(self) -> float:
        """Return the die width in micrometres."""
        return self.die_area_um[2] - self.die_area_um[0]

    @property
    def die_height_um(self) -> float:
        """Return the die height in micrometres."""
        return self.die_area_um[3] - self.die_area_um[1]

    @property
    def core_area_mm2(self) -> float:
        """Return the rectangular core area in square millimetres."""
        w = self.core_area_um[2] - self.core_area_um[0]
        h = self.core_area_um[3] - self.core_area_um[1]
        return (w * h) / 1e6
