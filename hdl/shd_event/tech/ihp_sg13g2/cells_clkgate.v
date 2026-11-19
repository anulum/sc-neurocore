// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — IHP SG13G2 integrated clock-gating cell wrapper
//
// Provenance: IHP SG13G2 latch branch of cells_clkgate.v from the group's earlier design study.
module OPENROAD_CLKGATE (CK, E, GCK);
  input CK;
  input E;
  output GCK;

sg13g2_lgcp_1 latch (.CLK (CK), .GATE(E), .GCLK(GCK));

endmodule
