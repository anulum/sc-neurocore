// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parameterised clock-gate wrapper for the event-driven SHD core
//
// Provenance: extracted code-identically from the group's earlier design study.
// Delegates to target leaf OPENROAD_CLKGATE when GATING != 0, else passes clk through.
`timescale 1ns / 1ps

module sc_cg #(parameter integer GATING = 0) (input wire clk, input wire en, output wire gclk);
    generate
        if (GATING) begin : g
            OPENROAD_CLKGATE icg (.CK(clk), .E(en), .GCK(gclk));
        end else begin : n
            assign gclk = clk;
        end
    endgenerate
endmodule
