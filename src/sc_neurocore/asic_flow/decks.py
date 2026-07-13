# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Yosys, OpenROAD, SDC, and GDSII deck generation

"""Generate synthesis, physical-design, timing, and stream-out decks."""

from __future__ import annotations

import textwrap

from sc_neurocore.asic_flow.design import DesignParams
from sc_neurocore.asic_flow.pdk import PDKConfig


class SynthesisGenerator:
    """Generates Yosys synthesis TCL scripts."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        """Render the Yosys synthesis script for ``design`` and ``pdk``."""
        rtl_reads = "\n".join(f"read_verilog {f}" for f in design.rtl_files)
        if not rtl_reads:
            rtl_reads = f"read_verilog {design.top_module}.v"
        sc_passes = "\n".join(design.sc_optimisation.yosys_passes())
        if design.sc_optimisation.preserve_lfsr_hierarchy:
            sc_passes = (
                "# Preserve deterministic SC seed generators for gate-level debug\n"
                "setattr -mod -pattern *lfsr* keep_hierarchy 1\n"
                f"{sc_passes}"
            )
        abc_delay_ps = design.clock_period_ns * 1000.0 * design.sc_optimisation.abc_delay_margin

        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Synthesis — Yosys Script
# PDK: {pdk.pdk_type.value}
# Target: {design.top_module} @ {design.target_frequency_mhz} MHz

# Read RTL
{rtl_reads}

# Hierarchy check
hierarchy -check -top {design.top_module}

# High-level synthesis
proc; opt; fsm; opt; memory; opt

# Technology mapping
synth -top {design.top_module}

# SC-aware optimisation
{sc_passes}

# Map to standard cells
dfflibmap -liberty {pdk.liberty_file}
abc -liberty {pdk.liberty_file} -D {abc_delay_ps:.0f}

# Clean up
opt_clean -purge

# Write outputs
write_verilog -noattr synth_{design.top_module}.v
write_json synth_{design.top_module}.json

# Statistics
stat -liberty {pdk.liberty_file}
""")


class FloorplanGenerator:
    """Generates OpenROAD floorplan TCL scripts."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        """Render the OpenROAD floorplan and optional two-net power grid."""
        die = design.die_area_um
        core = design.core_area_um
        power = design.power_nets

        power_ring = ""
        if len(power) >= 2:
            power_ring = textwrap.dedent(f"""\
# Power grid
add_global_connection -net {power[0]} -pin_pattern "VPWR|VDD|vdd" -power
add_global_connection -net {power[1]} -pin_pattern "VGND|VSS|vss" -ground

set_voltage_domain -name CORE -power {power[0]} -ground {power[1]}

define_pdn_grid -name core_grid -pins {{{power[0]} {power[1]}}} \\
    -voltage_domains CORE

add_pdn_stripe -grid core_grid -layer met1 -width 0.48 -pitch 5.44 \\
    -offset 0 -followpins
add_pdn_stripe -grid core_grid -layer met4 -width 1.6 -pitch 27.14 \\
    -offset 13.57
add_pdn_stripe -grid core_grid -layer met5 -width 1.6 -pitch 27.2 \\
    -offset 13.6

add_pdn_connect -grid core_grid -layers {{met1 met4}}
add_pdn_connect -grid core_grid -layers {{met4 met5}}
""")

        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Floorplan — OpenROAD
# Die: {die[2] - die[0]}×{die[3] - die[1]} µm

# Read technology
read_lef {pdk.tech_lef}
read_lef {pdk.lef_file}

# Read synthesized netlist
read_verilog synth_{design.top_module}.v
link_design {design.top_module}

# Read timing constraints
read_sdc constraints_{design.top_module}.sdc

# Initialize floorplan
initialize_floorplan \\
    -die_area {{{die[0]} {die[1]} {die[2]} {die[3]}}} \\
    -core_area {{{core[0]} {core[1]} {core[2]} {core[3]}}} \\
    -site unithd

# IO placement
place_pins -hor_layers met3 -ver_layers met2

{power_ring}
""")


class PlaceRouteGenerator:
    """Generates OpenROAD place-and-route TCL scripts."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        """Render the OpenROAD placement, clock-tree, and routing script."""
        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Place & Route — OpenROAD
# Target: {design.top_module}

# Global placement
global_placement -density {design.utilisation:.2f} -pad_left 2 -pad_right 2

# Clock tree synthesis
clock_tree_synthesis -root_buf {pdk.cell_prefix}buf_2 \\
    -buf_list {{{pdk.cell_prefix}buf_4 {pdk.cell_prefix}buf_8}} \\
    -wire_unit 10

# Repair hold violations
estimate_parasitics -placement
repair_timing -hold

# Detailed placement
detailed_placement

# Check placement
check_placement

# Filler cell insertion
filler_placement {pdk.cell_prefix}fill_1 {pdk.cell_prefix}fill_2

# Global routing
global_route -guide_file route_{design.top_module}.guide \\
    -congestion_iterations 30

# Detailed routing
detailed_route -output_drc route_drc_{design.top_module}.rpt \\
    -output_maze route_maze_{design.top_module}.log

# Write outputs
write_def {design.top_module}_final.def
write_verilog {design.top_module}_final.v
""")


class SDCGenerator:
    """Generates Synopsys Design Constraints (SDC) for STA."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        """Render clock, IO-delay, reset, fanout, and load constraints."""
        period = design.clock_period_ns
        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Constraints — SDC
# Clock: {design.clock_name} @ {design.target_frequency_mhz} MHz

create_clock [get_ports {design.clock_name}] \\
    -name {design.clock_name} \\
    -period {period:.3f}

# Clock uncertainty (10% of period)
set_clock_uncertainty {period * 0.1:.3f} [get_clocks {design.clock_name}]

# Input/output delays (25% of period)
set_input_delay {period * 0.25:.3f} -clock {design.clock_name} [all_inputs]
set_output_delay {period * 0.25:.3f} -clock {design.clock_name} [all_outputs]

# Reset is constant during operation
set_false_path -from [get_ports {design.reset_name}]

# Don't touch clock/reset nets
set_dont_touch_network [get_ports {design.clock_name}]

# Max transition / fanout
set_max_transition {period * 0.15:.3f} [current_design]
set_max_fanout {design.sc_optimisation.max_fanout} [current_design]

# Driving cell
set_driving_cell -lib_cell {pdk.cell_prefix}buf_2 [all_inputs]

# Load
set_load 0.05 [all_outputs]
""")


class GDSIIExporter:
    """Generates GDSII stream-out scripts."""

    @staticmethod
    def generate(pdk: PDKConfig, design: DesignParams) -> str:
        """Render open-PDK stream-out commands or a vendor-tool boundary."""
        if pdk.is_open_source:
            return textwrap.dedent(f"""\
# SC-NeuroCore GDSII Export — KLayout/Magic
# Merge standard cell GDS with routed DEF

# Option 1: KLayout
klayout -zz -rd design_name={design.top_module} \\
    -rd in_def={design.top_module}_final.def \\
    -rd in_gds="$PDK_ROOT/{pdk.pdk_type.value}/libs.ref/*/gds/*.gds" \\
    -rd seal_gds="" \\
    -rd out_gds={design.top_module}.gds \\
    -rm $OPENLANE_ROOT/scripts/klayout/def2gds.py

# Option 2: Magic
magic -dnull -noconsole << EOF
lef read {pdk.tech_lef}
lef read {pdk.lef_file}
def read {design.top_module}_final.def
load {design.top_module}
select top cell
expand
gds write {design.top_module}.gds
quit
EOF
""")
        return f"# GDSII export for {pdk.pdk_type.value}: use vendor stream-out\n"
