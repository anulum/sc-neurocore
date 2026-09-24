// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue formal harness for sc_cobalifneuron

`default_nettype none

// Formal wrapper for the curated equation-compiler COBA LIF RTL.
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

    // Reachable edge protocol: the source reset voltage is also the configured
    // threshold, so the first complete RK4 macro step must emit an event. The
    // maintained conductance increments remain represented in the datapath.
    sc_cobalifneuron #(
        .P_V_THRESHOLD(-48'sd1006632960),
        .P_DELTA_GE(48'sd2516582),
        .P_DELTA_GI(48'sd1174405)
    ) uut (
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
    reg protocol_started = 1'b0;
    reg [3:0] protocol_cycle = 4'd0;
    reg seen_spike = 1'b0;

    always @(posedge clk) begin
        if (!protocol_started)
            assume (!rst_n);
        else
            assume (rst_n);
        assume ($signed(I_t) == 48'sd10905190400);
        protocol_started <= 1'b1;
        if (protocol_started)
            protocol_cycle <= protocol_cycle + 4'd1;
        seen_spike <= seen_spike || spike_out;

        if (protocol_cycle == 4'd5)
            assert (seen_spike);

        if (protocol_started && rst_n && spike_out) begin
            assert ($signed(v_out) == -48'sd1006632960);
            assert ($signed(refractory_time_out) == 48'sd83886080);
            assert ($signed(spike_flag_out) >= 48'sd8388608);
        end

        if (protocol_started && rst_n && $past(spike_out)) begin
            assert (!spike_out);
            assert ($signed(v_out) == -48'sd1006632960);
            assert ($signed(refractory_time_out) == 48'sd83886080);
        end
    end

    // Async reset clears every public biological state and event output.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
            assert ($signed(v_out) == -48'sd1006632960);
            assert ($signed(g_e_out) == 48'sd0);
            assert ($signed(g_i_out) == 48'sd0);
            assert ($signed(refractory_time_out) == 48'sd0);
            assert ($signed(spike_flag_out) == 48'sd0);
        end
    end
`endif

endmodule
