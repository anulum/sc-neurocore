// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal wrapper for the Cazelles source map

`default_nettype none

module sc_cazelles_map_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [15:0] I_t
);
    wire spike_out;
    wire signed [15:0] x_out;

    sc_cazelles_map uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .x_out(x_out)
    );

`ifdef FORMAL
    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end
`endif
endmodule
