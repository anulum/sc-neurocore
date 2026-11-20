# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Out-of-context timing constraints for sc_shd_event_axi (XC7Z020, 100 MHz)

# Primary AXI4-Lite clock — 100 MHz (10.000 ns period) from Zynq PS FCLK_CLK0
create_clock -period 10.000 -name clk [get_ports S_AXI_ACLK]

# AXI interconnect setup/hold budgets matching hdl/constraints/pynq_z2.xdc
set_input_delay -clock clk -max 2.0 [get_ports -filter {DIRECTION == IN && NAME != "S_AXI_ACLK"}]
set_input_delay -clock clk -min 0.5 [get_ports -filter {DIRECTION == IN && NAME != "S_AXI_ACLK"}]
set_output_delay -clock clk -max 2.0 [get_ports -filter {DIRECTION == OUT}]
set_output_delay -clock clk -min 0.5 [get_ports -filter {DIRECTION == OUT}]
