# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# vivado_synth.tcl
#
# Tcl script to synthesize sc_neurocore for PYNQ-Z2 (Zynq-7000)
# Usage: vivado -mode batch -source vivado_synth.tcl

# 1. Project Settings
set project_name "sc_neurocore_project"
set part_name "xc7z020clg400-1"
set output_dir "vivado_out"

file mkdir $output_dir

# 2. Create Project
create_project -force $project_name $output_dir -part $part_name

# 3. Add Sources
# Note: Paths relative to where script is run (project root)
add_files [glob hdl/*.v]

# 4. Set Top Module
set_property top sc_neurocore_top [current_fileset]

# 5. Define Clock Constraint (100 MHz for AXI)
# create_clock -period 10.000 -name clk [get_ports S_AXI_ACLK]
# (Typically done in XDC, but inline here for OOC synthesis)
create_clock -period 10.0 [get_ports S_AXI_ACLK]

# 6. Run Synthesis
# Out-of-context mode is typical for IP generation
synth_design -top sc_neurocore_top -part $part_name -mode out_of_context

# 7. Report Utilization & Timing
report_utilization -file $output_dir/post_synth_util.rpt
report_timing_summary -file $output_dir/post_synth_timing.rpt

# 8. Write Checkpoint and Netlist
write_checkpoint -force $output_dir/post_synth.dcp
write_verilog -force $output_dir/post_synth_netlist.v

# Done
exit
