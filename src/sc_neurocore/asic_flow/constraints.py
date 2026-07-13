# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC CDC, power-grid, IO, and equivalence constraints

"""Generate auxiliary ASIC CDC, IR-drop, IO-placement, and LEC decks."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import List, Optional

from sc_neurocore.asic_flow.design import DesignParams
from sc_neurocore.asic_flow.pdk import PDKConfig


class CDCCheckGenerator:
    """Generates clock-domain crossing lint scripts."""

    @staticmethod
    def generate(design: DesignParams, clock_domains: Optional[List[str]] = None) -> str:
        """Render CDC checks for explicit domains or the design clock."""
        if clock_domains is None:
            clock_domains = [design.clock_name]
        domain_defs = "\n".join(f"create_clock -name {c} [get_ports {c}]" for c in clock_domains)
        return textwrap.dedent(f"""\
# SC-NeuroCore CDC Check
# Domains: {", ".join(clock_domains)}

{domain_defs}

# Report all clock-domain crossings
report_cdc -from [all_clocks] -to [all_clocks]
report_cdc -type async_reset
report_cdc -type reconvergence

# Check for missing synchronisers
check_cdc -severity error
""")


class IRDropGenerator:
    """Generates IR drop analysis scripts for OpenROAD."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams, toggle_rate: float = 0.1) -> str:
        """Render OpenROAD power-grid analysis at an input toggle fraction."""
        return textwrap.dedent(f"""\
# SC-NeuroCore IR Drop Analysis — OpenROAD
# Toggle rate: {toggle_rate:.2f}

# Read design
read_lef {pdk.tech_lef}
read_lef {pdk.lef_file}
read_def {design.top_module}_final.def
read_liberty {pdk.liberty_file}
read_sdc constraints_{design.top_module}.sdc

# Set activity
set_power_activity -input -activity {toggle_rate:.3f}

# Analyze IR drop
analyze_power_grid -net {design.power_nets[0]}
analyze_power_grid -net {design.power_nets[1]}

# Report
report_power_grid -net {design.power_nets[0]} -corner tt
""")


@dataclass
class IOPin:
    """Specification for one IO pad."""

    name: str
    direction: str  # "input", "output", "inout"
    side: str = "N"  # N, S, E, W
    offset_um: float = 0.0
    layer: str = "met3"


@dataclass
class IOConstraintGenerator:
    """Generates IO placement constraint files."""

    @staticmethod
    def generate(pins: List[IOPin], design: DesignParams) -> str:
        """Render one OpenROAD ``place_pin`` command per supplied IO pin."""
        lines = [f"# IO Constraints for {design.top_module}"]
        for pin in pins:
            lines.append(
                f"place_pin -pin_name {pin.name} -layer {pin.layer} "
                f"-location {{{pin.offset_um} 0}} -side {pin.side}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def auto_assign(signal_names: List[str], sides: str = "NSEW") -> List[IOPin]:
        """Auto-assign pins to die edges round-robin."""
        pins = []
        for i, name in enumerate(signal_names):
            side = sides[i % len(sides)]
            pins.append(IOPin(name=name, direction="input", side=side, offset_um=float(i * 10)))
        return pins


class LECGenerator:
    """Generates Logic Equivalence Checking scripts."""

    @staticmethod
    def generate(design: DesignParams) -> str:
        """Render a Yosys equivalence proof between synthesis and routed RTL."""
        return textwrap.dedent(f"""\
# SC-NeuroCore LEC — Yosys equivalence check
# Golden: synth_{design.top_module}.v
# Revised: {design.top_module}_final.v

read_verilog synth_{design.top_module}.v
prep -top {design.top_module}
design -stash golden

read_verilog {design.top_module}_final.v
prep -top {design.top_module}
design -stash revised

design -copy-from golden -as golden {design.top_module}
design -copy-from revised -as revised {design.top_module}

equiv_make golden revised equiv
equiv_simple
equiv_induct
equiv_status -assert
""")
