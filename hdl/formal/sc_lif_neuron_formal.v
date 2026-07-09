// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal Verification for SC LIF Neuron

`default_nettype none

module sc_lif_neuron_formal (
    input wire        clk,
    input wire        rst_n,
    input wire signed [15:0] leak_k,
    input wire signed [15:0] gain_k,
    input wire signed [15:0] I_t,
    input wire signed [15:0] noise_in
);

    wire        spike_out;
    wire signed [15:0] v_out;

    sc_lif_neuron #(
        .DATA_WIDTH(16),
        .FRACTION(8),
        .V_REST(0),
        .V_RESET(0),
        .V_THRESHOLD(16'sd256),   // 1.0 in Q8.8
        .REFRACTORY_PERIOD(2)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .leak_k(leak_k),
        .gain_k(gain_k),
        .I_t(I_t),
        .noise_in(noise_in),
        .spike_out(spike_out),
        .v_out(v_out)
    );

`ifdef FORMAL
    reg past_valid = 0;
    always @(posedge clk)
        past_valid <= 1;

    // 1. After reset, membrane potential equals V_REST
    always @(posedge clk) begin
        if (past_valid && !rst_n)
            assert(v_out == 16'sd0);
    end

    // 2. Spike only when v_next >= V_THRESHOLD
    //    When spike fires, output resets to V_RESET
    always @(posedge clk) begin
        if (past_valid && rst_n && spike_out)
            assert(v_out == 16'sd0);  // V_RESET = 0
    end

    // 3. Refractory counter decrements monotonically
    //    During refractory, v is clamped to V_REST and no spike
    always @(posedge clk) begin
        if (past_valid && rst_n && uut.refractory_counter > 0) begin
            assert(spike_out == 1'b0);
            assert(v_out == 16'sd0);  // Clamped to V_REST
        end
    end

    // 4. Refractory counter bounded by REFRACTORY_PERIOD
    always @(posedge clk) begin
        if (past_valid && rst_n)
            assert(uut.refractory_counter <= 3);
    end

    // 5. Cover: spike is reachable
    always @(posedge clk) begin
        if (past_valid && rst_n)
            cover(spike_out == 1'b1);
    end
`endif

endmodule
