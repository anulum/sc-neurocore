// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_ermentroutkopellpopulation

`default_nettype none

// Formal wrapper for equation-compiler RTL of a dual-axis perfect model.
// Properties use only public ports so default_nettype none stays clean.
module sc_ermentroutkopellpopulation_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] I_t
);

    wire spike_out;
    wire signed [63:0] r_out;
    wire signed [63:0] v_out;

    sc_ermentroutkopellpopulation uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .r_out(r_out),
        .v_out(v_out)
    );

`ifdef FORMAL
    // Minimal safety: async reset clears the spike flag.
    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end

    reg past_valid = 1'b0;
    always @(posedge clk) begin
        past_valid <= 1'b1;
        if (past_valid && rst_n)
            assert (spike_out == 1'b0);
    end

`endif

endmodule
