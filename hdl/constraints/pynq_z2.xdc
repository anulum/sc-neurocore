# SC-NeuroCore — PYNQ-Z2 timing constraints for sc_shd_top
# Target: Zynq XC7Z020-1CLG400C (PYNQ-Z2 board)
# Clock: 100 MHz from PS FCLK_CLK0 (via AXI interconnect)

# Primary clock — 100 MHz (10 ns period)
# In a block design, this is auto-created by the PS. For OOC synthesis:
create_clock -period 10.000 -name clk [get_ports clk]

# Input delay — assume 2 ns setup from AXI interconnect
set_input_delay -clock clk -max 2.0 [get_ports -filter {DIRECTION == IN && NAME != "clk"}]
set_input_delay -clock clk -min 0.5 [get_ports -filter {DIRECTION == IN && NAME != "clk"}]

# Output delay — assume 2 ns hold to AXI interconnect
set_output_delay -clock clk -max 2.0 [get_ports -filter {DIRECTION == OUT}]
set_output_delay -clock clk -min 0.5 [get_ports -filter {DIRECTION == OUT}]

# Multicycle paths — none needed for single-clock synchronous design

# False paths — none

# IO standard — LVCMOS33 for PYNQ-Z2 PMOD/GPIO (if any external IO used)
# set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]
