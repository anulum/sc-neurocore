# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Vivado Build Script for SC-NeuroCore
# Usage: vivado -mode batch -source build_bitstream.tcl

# 1. Project Configuration
set project_name "sc_neurocore_proj"
set part_name "xc7z020clg400-1" ;# PYNQ-Z1/Z2 standard part. Change if using ZCU104 etc.
set output_dir "output"

# 2. Setup Project
file mkdir $output_dir
create_project -force $project_name $output_dir -part $part_name

# 3. Add Sources
# Assuming script is run from 03_CODE/sc-neurocore/scripts, so hdl is at ../hdl
set hdl_dir "../hdl"
add_files [glob $hdl_dir/*.v]

# 4. Set Top Module
set_property top sc_neurocore_top [current_fileset]
update_compile_order -fileset sources_1

# 5. Run Synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed"
    exit 1
}

# 6. Run Implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed"
    exit 1
}

# 7. Generate Bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# 8. Export
set bitfile_path "$output_dir/$project_name.runs/impl_1/sc_neurocore_top.bit"
if {[file exists $bitfile_path]} {
    file copy -force $bitfile_path "sc_neurocore.bit"
    puts "SUCCESS: Bitstream generated at sc_neurocore.bit"
} else {
    puts "ERROR: Bitstream not found."
    exit 1
}

close_project
