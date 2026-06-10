# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SLR placement constraints

"""SLR placement utilities for multi-die FPGA deployment.

Generates Vivado XDC PBLOCK constraints to pin modules to specific
Super Logic Regions (SLRs) in large FPGAs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SLRPlacement:
    """SLR (Super Logic Region) placement for multi-die FPGAs.

    Attributes
    ----------
    module_name : str
        Module or instance name.
    slr : int
        Target SLR index (0-based).
    pblock_name : str
        Vivado PBLOCK name (auto-generated if empty).
    """

    module_name: str
    slr: int
    pblock_name: str = ""

    def __post_init__(self) -> None:
        """Auto-generate pblock name if not set."""
        if not self.pblock_name:
            self.pblock_name = f"pblock_slr{self.slr}"


def generate_slr_constraints(
    placements: list[SLRPlacement],
    *,
    insert_pipeline_regs: bool = True,
    target_freq_mhz: float = 500.0,
) -> str:
    """Generate Vivado XDC for multi-die SLR placement.

    Emits PBLOCK constraints that pin modules to specific SLRs and
    optionally adds inter-SLR pipeline register directives.

    Parameters
    ----------
    placements : list[SLRPlacement]
        Module-to-SLR assignments.
    insert_pipeline_regs : bool
        Add register duplication directives for SLR crossings.
    target_freq_mhz : float
        Target frequency for SLR crossing timing.

    Returns
    -------
    str
        Complete XDC constraint block.
    """
    period_ns = 1000.0 / target_freq_mhz
    lines = [
        "# Auto-generated SLR placement constraints",
        "# SC-NeuroCore multi-die deployment",
        f"# Target: {target_freq_mhz:.0f} MHz",
        "",
    ]

    slrs_used: set[int] = set()
    for p in placements:
        slrs_used.add(p.slr)
        lines.extend(
            [
                f"create_pblock {p.pblock_name}",
                f"add_cells_to_pblock [get_pblocks {p.pblock_name}] "
                f"[get_cells -hier -filter {{NAME =~ *{p.module_name}*}}]",
                f"resize_pblock [get_pblocks {p.pblock_name}] -add SLR{p.slr}",
                "",
            ]
        )

    if insert_pipeline_regs and len(slrs_used) > 1:
        lines.extend(
            [
                "# Inter-SLR pipeline register directives",
                "set_property REGISTER_DUPLICATION true [get_cells -hier -filter {IS_SEQUENTIAL}]",
                f"set_max_delay {period_ns / 2:.3f} "
                "-datapath_only -from [get_clocks *] -to [get_clocks *]",
                "",
            ]
        )

    return "\n".join(lines)
