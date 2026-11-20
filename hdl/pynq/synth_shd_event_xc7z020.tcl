# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Out-of-context synthesis and implementation of sc_shd_event_axi (XC7Z020, 100 MHz)
#
# Reproduces the committed out-of-context synthesis and placed-and-routed
# utilisation, timing, and DRC reports for the event-driven SHD core and
# AXI4-Lite integrity-checked loader (`hdl/shd_event/sc_shd_event_axi.v`) with
# the full trained integer SHD network geometry (`hdl/fixtures/shd_int_network.json`:
# P=8, FRAC_BITS=0, THETA1=72, THETA2=51, V1_WIDTH=22, V2_WIDTH=17, CAP1=62,
# CAP2=39, WSHAPE=6, QSHAPE=12) targeting Zynq-7000 `xc7z020clg400-1` at 100 MHz.
#
# Run from the repository root:
#   vivado -mode batch -nojournal -nolog -source hdl/pynq/synth_shd_event_xc7z020.tcl -tclargs <out_dir>

set script_dir [file dirname [info script]]
set repo_root [expr {$script_dir eq "." ? "../.." : "${script_dir}/../.."}]
set out_dir [expr {[llength $argv] >= 1 ? [lindex $argv 0] : "${repo_root}/hdl/reports"}]
file mkdir $out_dir

set_param general.maxThreads 8

read_verilog [list \
    ${repo_root}/hdl/shd_event/sc_cg.v \
    ${repo_root}/hdl/shd_event/sc_lanes.v \
    ${repo_root}/hdl/shd_event/tech/generic/cells_clkgate.v \
    ${repo_root}/hdl/shd_event/tech/xilinx_7series/sc_mem.v \
    ${repo_root}/hdl/shd_event/sc_shd_event_p.v \
    ${repo_root}/hdl/shd_event/sc_shd_event_axi.v \
]

read_xdc ${repo_root}/hdl/constraints/sc_shd_event_axi_ooc_100mhz.xdc

synth_design -top sc_shd_event_axi -part xc7z020clg400-1 -mode out_of_context \
    -include_dirs [list ${repo_root}/hdl/shd_event] \
    -generic P=8 \
    -generic FRAC_BITS=0 \
    -generic THETA1=72 \
    -generic THETA2=51 \
    -generic V1_WIDTH=22 \
    -generic V2_WIDTH=17 \
    -generic CAP1=62 \
    -generic CAP2=39 \
    -generic WSHAPE=6 \
    -generic QSHAPE=12

report_utilization -file ${out_dir}/vivado_sc_shd_event_axi_synth_util_xc7z020_100mhz.rpt
report_utilization -hierarchical -file ${out_dir}/vivado_sc_shd_event_axi_synth_hier_util_xc7z020_100mhz.rpt
report_timing_summary -file ${out_dir}/vivado_sc_shd_event_axi_synth_timing_xc7z020_100mhz.rpt

set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports S_AXI_ACLK]
opt_design
place_design
phys_opt_design
route_design

report_utilization -file ${out_dir}/vivado_sc_shd_event_axi_routed_util_xc7z020_100mhz.rpt
report_utilization -hierarchical -file ${out_dir}/vivado_sc_shd_event_axi_routed_hier_util_xc7z020_100mhz.rpt
report_timing_summary -file ${out_dir}/vivado_sc_shd_event_axi_routed_timing_xc7z020_100mhz.rpt
report_drc -file ${out_dir}/vivado_sc_shd_event_axi_routed_drc_xc7z020_100mhz.rpt
