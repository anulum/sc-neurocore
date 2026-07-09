# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Standalone Vivado synthesis script for SC-NeuroCore.
#
# Usage:
#   vivado -mode batch -source deploy/fpga/vivado_synth.tcl
#
# Override part with:
#   vivado -mode batch -source deploy/fpga/vivado_synth.tcl -tclargs xc7z020clg400-1

set part "xc7a35tcpg236-1"
if {$argc > 0} {
    set part [lindex $argv 0]
}

set project_dir "build/vivado_project"
set hdl_dir "hdl"
set xdc_file "deploy/fpga/constraints.xdc"

# Create project
create_project sc_neurocore $project_dir -part $part -force

# Add HDL sources
foreach f [glob -directory $hdl_dir *.v] {
    if {[string match "*tb_*" $f]} { continue }
    add_files $f
}
set_property top sc_neurocore_top [current_fileset]

# Add constraints
if {[file exists $xdc_file]} {
    add_files -fileset constrs_1 $xdc_file
}

# Synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Reports
open_run synth_1
report_utilization -file $project_dir/utilization.rpt
report_timing_summary -file $project_dir/timing.rpt
report_power -file $project_dir/power.rpt

puts "============================================"
puts "  Synthesis complete for part: $part"
puts "  Utilization: $project_dir/utilization.rpt"
puts "  Timing:      $project_dir/timing.rpt"
puts "  Power:       $project_dir/power.rpt"
puts "============================================"

# Implementation + bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

puts "Bitstream generated."
