# SPDX-License-Identifier: AGPL-3.0-or-later
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
