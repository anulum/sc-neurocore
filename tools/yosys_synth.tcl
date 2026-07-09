# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Yosys synthesis script for SC-NeuroCore HDL modules
# Targets Xilinx 7-series (Artix-7) via synth_xilinx
#
# Usage: yosys -s tools/yosys_synth.tcl -D MODULE=sc_lif_neuron
#
# The MODULE variable selects which top-level to synthesize.
# Resource utilization is printed via 'stat' at the end.

if {![info exists ::env(MODULE)]} {
    set mod "sc_lif_neuron"
} else {
    set mod $::env(MODULE)
}

set hdl_dir [file join [file dirname [info script]] .. hdl]

# Read all HDL sources (modules may instantiate each other)
foreach f [glob -directory $hdl_dir *.v] {
    set basename [file tail $f]
    # Skip testbenches
    if {[string match "tb_*" $basename]} continue
    read_verilog $f
}

# Synthesize for Xilinx 7-series
synth_xilinx -top $mod -flatten

# Print resource utilization
stat
