// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_cobalifneuron

`default_nettype none

// Formal wrapper for equation-compiler RTL of a dual-axis perfect model.
// Properties use only public ports so default_nettype none stays clean.
module sc_cobalifneuron_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [47:0] I_t
);

    wire spike_out;
    wire signed [47:0] v_out;
    wire signed [47:0] g_e_out;
    wire signed [47:0] g_i_out;
    wire signed [47:0] refractory_time_out;
    wire signed [47:0] spike_flag_out;
    wire signed [47:0] phase_out;
    wire signed [47:0] base_v_out;
    wire signed [47:0] base_ge_out;
    wire signed [47:0] base_gi_out;
    wire signed [47:0] last_k_v_out;
    wire signed [47:0] last_k_ge_out;
    wire signed [47:0] last_k_gi_out;
    wire signed [47:0] weighted_v_out;
    wire signed [47:0] weighted_ge_out;
    wire signed [47:0] weighted_gi_out;

    sc_cobalifneuron uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .v_out(v_out),
        .g_e_out(g_e_out),
        .g_i_out(g_i_out),
        .refractory_time_out(refractory_time_out),
        .spike_flag_out(spike_flag_out),
        .phase_out(phase_out),
        .base_v_out(base_v_out),
        .base_ge_out(base_ge_out),
        .base_gi_out(base_gi_out),
        .last_k_v_out(last_k_v_out),
        .last_k_ge_out(last_k_ge_out),
        .last_k_gi_out(last_k_gi_out),
        .weighted_v_out(weighted_v_out),
        .weighted_ge_out(weighted_ge_out),
        .weighted_gi_out(weighted_gi_out)
    );

`ifdef FORMAL
    // Minimal safety: async reset clears the spike flag.
    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end
`endif

endmodule
