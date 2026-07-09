# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# tools/vivado_impl.tcl
#
# Vivado non-project synthesis + implementation for SC-NeuroCore.
# Produces timing, utilization, and power reports.
#
# Usage:
#   vivado -mode batch -source tools/vivado_impl.tcl
#   vivado -mode batch -source tools/vivado_impl.tcl -tclargs -top sc_dense_matrix_layer -part xc7z020clg484-1
#
# Defaults: sc_neurocore_top on xc7a100tcsg324-1 (Artix-7 100T)

# ---- Parse arguments ----
set top_module "sc_neurocore_top"
set part       "xc7a100tcsg324-1"
set clk_mhz    250
set out_dir    "vivado_reports"

for {set i 0} {$i < [llength $argv]} {incr i} {
    set arg [lindex $argv $i]
    switch -- $arg {
        -top  { incr i; set top_module [lindex $argv $i] }
        -part { incr i; set part       [lindex $argv $i] }
        -clk  { incr i; set clk_mhz    [lindex $argv $i] }
        -out  { incr i; set out_dir    [lindex $argv $i] }
    }
}

set clk_period [expr {1000.0 / $clk_mhz}]
puts "=== SC-NeuroCore Vivado Implementation ==="
puts "  Top:    $top_module"
puts "  Part:   $part"
puts "  Clock:  ${clk_mhz} MHz (${clk_period} ns)"
puts "  Output: $out_dir"

# ---- Read sources ----
read_verilog [glob hdl/*.v]

# ---- Synthesis ----
synth_design -top $top_module -part $part -flatten_hierarchy rebuilt
file mkdir $out_dir
report_utilization -file ${out_dir}/utilization_synth.rpt

# ---- Clock constraint ----
create_clock -period $clk_period -name sys_clk [get_ports clk]

# ---- Implementation ----
opt_design
place_design
phys_opt_design
route_design

# ---- Reports ----
report_timing_summary -file ${out_dir}/timing_summary.rpt
report_utilization    -file ${out_dir}/utilization_impl.rpt
report_power          -file ${out_dir}/power.rpt
report_drc            -file ${out_dir}/drc.rpt

# ---- Write checkpoint ----
write_checkpoint -force ${out_dir}/${top_module}_impl.dcp

# ---- Summary ----
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1]]
set fmax [expr {1000.0 / ($clk_period - $wns)}]
puts ""
puts "=== Results ==="
puts "  WNS:  ${wns} ns"
puts "  Fmax: [format %.1f $fmax] MHz"
puts "  Reports in: $out_dir/"
puts "  Parse with: python tools/vivado_report.py $out_dir"
