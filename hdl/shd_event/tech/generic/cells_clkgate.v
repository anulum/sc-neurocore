// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pass-through clock-gate leaf for generic and FPGA targets
//
// Provenance: pass-through branch of cells_clkgate.v from the group's earlier design study.
module OPENROAD_CLKGATE (CK, E, GCK);
  input CK;
  input E;
  output GCK;

assign GCK = CK;

endmodule
