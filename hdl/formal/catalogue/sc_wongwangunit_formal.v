// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_wongwangunit

`default_nettype none

// Formal wrapper for equation-compiler RTL of a dual-axis perfect model.
// Properties use only public ports so default_nettype none stays clean.
module sc_wongwangunit_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] I_t
);

    wire spike_out;
    wire signed [63:0] s1_out;
    wire signed [63:0] s2_out;
    wire signed [63:0] noise1_out;
    wire signed [63:0] noise2_out;
    wire signed [63:0] r1_out;
    wire signed [63:0] r2_out;
    wire signed [63:0] x1_out;
    wire signed [63:0] x2_out;
    wire signed [63:0] phase_out;
    wire signed [63:0] stim1_latched_out;
    wire signed [63:0] stim2_latched_out;
    wire signed [63:0] xi1_latched_out;
    wire signed [63:0] xi2_latched_out;

    sc_wongwangunit uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .s1_out(s1_out),
        .s2_out(s2_out),
        .noise1_out(noise1_out),
        .noise2_out(noise2_out),
        .r1_out(r1_out),
        .r2_out(r2_out),
        .x1_out(x1_out),
        .x2_out(x2_out),
        .phase_out(phase_out),
        .stim1_latched_out(stim1_latched_out),
        .stim2_latched_out(stim2_latched_out),
        .xi1_latched_out(xi1_latched_out),
        .xi2_latched_out(xi2_latched_out)
    );

`ifdef FORMAL
    // Minimal safety: async reset clears the spike flag.
    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end
`endif

endmodule
