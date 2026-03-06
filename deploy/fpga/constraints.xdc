# SPDX-License-Identifier: AGPL-3.0-or-later
# SC-NeuroCore FPGA constraints for Xilinx Artix-7 (Arty A7-35T)
#
# Adapt clock pin and period to your board.  The default assumes the
# 100 MHz oscillator on the Digilent Arty A7.

# ---------- Clock ----------
set_property -dict {PACKAGE_PIN E3 IOSTANDARD LVCMOS33} [get_ports clk]
create_clock -period 10.000 -name sys_clk [get_ports clk]

# ---------- Reset (active-high button BTN0) ----------
set_property -dict {PACKAGE_PIN D9 IOSTANDARD LVCMOS33} [get_ports rst]

# ---------- AXI-Lite (directly on Pmod or MicroBlaze/ZYNQ) ----------
# Uncomment and adjust for your board's AXI interface pins.
# set_property -dict {PACKAGE_PIN ... IOSTANDARD LVCMOS33} [get_ports s_axil_*]

# ---------- Timing ----------
# Allow 2 ns of input delay on the AXI bus (conservative)
# set_input_delay -clock sys_clk -max 2.0 [get_ports s_axil_*]

# ---------- Configuration ----------
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
