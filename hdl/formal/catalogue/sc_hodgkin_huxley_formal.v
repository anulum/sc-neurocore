// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_hodgkin_huxley

`default_nettype none

// Formal wrapper for equation-compiler RTL of a dual-axis perfect model.
// Properties use only public ports so default_nettype none stays clean.
module sc_hodgkin_huxley_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t
);

    wire spike_out;
    wire signed [31:0] v_out;
    wire signed [31:0] m_out;
    wire signed [31:0] h_out;
    wire signed [31:0] n_out;

    sc_hodgkin_huxley uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .v_out(v_out),
        .m_out(m_out),
        .h_out(h_out),
        .n_out(n_out)
    );

`ifdef FORMAL

    // Bounded receipt protocol: initialise through reset and hold the exact
    // enrolled fixed-point drive while checking the public reset property.
    reg protocol_started = 1'b0;
    always @(posedge clk) begin
        if (!protocol_started)
            assume (!rst_n);
        else
            assume (rst_n);
        assume ($signed(I_t) == 32'sd983040);
        protocol_started <= 1'b1;
    end

    // Minimal safety: async reset clears the spike flag.
    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end
`endif

endmodule
