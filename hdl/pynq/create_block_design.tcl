# SC-NeuroCore — Vivado block design for PYNQ-Z2
# Creates: Zynq PS + AXI interconnect + sc_shd_axi_wrapper
#
# Run from Vivado Tcl console or batch mode:
#   vivado -mode batch -source create_block_design.tcl

set project_name sc_shd_pynq
set part xc7z020clg400-1
set hdl_dir [file normalize [file dirname [info script]]/../../hdl]

# Create project
create_project ${project_name} ./vivado_project -part ${part} -force

# Add HDL sources
add_files -norecurse [list \
    ${hdl_dir}/sc_vmin_lif_neuron.v \
    ${hdl_dir}/sc_axonal_delay.v \
    ${hdl_dir}/sc_dense_int8_sparse.v \
    ${hdl_dir}/sc_shd_top.v \
    ${hdl_dir}/sc_shd_axi_wrapper.v \
]
add_files -fileset constrs_1 -norecurse ${hdl_dir}/constraints/pynq_z2.xdc

# Create block design
create_bd_design "system"

# Add Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0
# Apply PYNQ-Z2 preset (100 MHz FCLK, DDR3, etc.)
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
] [get_bd_cells ps7_0]

# Add AXI interconnect
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ic_0
set_property CONFIG.NUM_MI {1} [get_bd_cells axi_ic_0]

# Add sc_shd_axi_wrapper as RTL module
create_bd_cell -type module -reference sc_shd_axi_wrapper sc_shd_0

# Connect clocks and resets
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] \
    [get_bd_pins sc_shd_0/S_AXI_ACLK] \
    [get_bd_pins axi_ic_0/ACLK] \
    [get_bd_pins axi_ic_0/S00_ACLK] \
    [get_bd_pins axi_ic_0/M00_ACLK]

connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] \
    [get_bd_pins sc_shd_0/S_AXI_ARESETN] \
    [get_bd_pins axi_ic_0/ARESETN] \
    [get_bd_pins axi_ic_0/S00_ARESETN] \
    [get_bd_pins axi_ic_0/M00_ARESETN]

# Connect AXI interfaces
connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0] \
    [get_bd_intf_pins axi_ic_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ic_0/M00_AXI] \
    [get_bd_intf_pins sc_shd_0/S_AXI]

# Assign address — 256 bytes at 0x43C0_0000
assign_bd_address -target_address_space /ps7_0/Data \
    [get_bd_addr_segs sc_shd_0/S_AXI/reg0] \
    -range 256 -offset 0x43C00000

# Validate and save
validate_bd_design
save_bd_design

# Generate wrapper
make_wrapper -files [get_files system.bd] -top
add_files -norecurse vivado_project/${project_name}.gen/sources_1/bd/system/hdl/system_wrapper.v
update_compile_order -fileset sources_1

# Run synthesis + implementation + bitstream
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Export hardware for PYNQ
write_hw_platform -fixed -include_bit -force \
    vivado_project/${project_name}.xsa

puts "=== BUILD COMPLETE ==="
puts "Bitstream: vivado_project/${project_name}.runs/impl_1/system_wrapper.bit"
puts "HW platform: vivado_project/${project_name}.xsa"
puts ""
puts "For PYNQ overlay, copy .bit and .hwh to the board:"
puts "  scp system_wrapper.bit xilinx@pynq:/home/xilinx/sc_shd.bit"
puts "  scp system_wrapper.hwh xilinx@pynq:/home/xilinx/sc_shd.hwh"
