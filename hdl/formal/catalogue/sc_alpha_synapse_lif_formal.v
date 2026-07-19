// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_alpha_synapse_lif

`default_nettype none

// Formal wrapper for equation-compiler RTL of a dual-axis perfect model.
// Properties use only public ports so default_nettype none stays clean.
module sc_alpha_synapse_lif_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] I_t
);

    wire spike_out;
    wire signed [63:0] a_exc_out;
    wire signed [63:0] i_exc_out;
    wire signed [63:0] a_inh_out;
    wire signed [63:0] i_inh_out;
    wire signed [63:0] v_out;

    sc_alpha_synapse_lif uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .a_exc_out(a_exc_out),
        .i_exc_out(i_exc_out),
        .a_inh_out(a_inh_out),
        .i_inh_out(i_inh_out),
        .v_out(v_out)
    );

`ifdef FORMAL
    // Minimal safety: async reset clears the spike flag.
    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end
`endif

endmodule
