// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_dpineuron

`default_nettype none

// Formal wrapper for equation-compiler RTL of a dual-axis perfect model.
// Properties use only public ports so default_nettype none stays clean.
module sc_dpineuron_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t
);

    wire spike_out;
    wire signed [31:0] i_mem_out;
    wire signed [31:0] i_ahp_out;
    wire signed [31:0] refractory_time_out;

    sc_dpineuron uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .i_mem_out(i_mem_out),
        .i_ahp_out(i_ahp_out),
        .refractory_time_out(refractory_time_out)
    );

`ifdef FORMAL
    // Minimal safety: async reset clears the spike flag.
    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end
`endif

endmodule
