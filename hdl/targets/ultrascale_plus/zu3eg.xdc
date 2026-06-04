# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - ZU3EG timing baseline constraints

# Clock-only baseline for xczu3eg-sbva484-1-e.
# Board-specific LOC constraints must come from a verified board-revision
# manifest. This file intentionally avoids fabricated pin assignments.
create_clock -name sc_neurocore_clk -period 4.000 [get_ports clk]
set_property IOSTANDARD LVCMOS18 [get_ports rst_n]
