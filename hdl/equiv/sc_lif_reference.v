// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Reference LIF model for formal equivalence proof
//
// Minimal LIF matching Python FixedPointLIFNeuron exactly:
//   v_next = v + (V_REST - v)*leak_k/2^FRAC + I_t*gain_k/2^FRAC + noise
//   if v_next >= V_THRESHOLD: spike, v = V_RESET
//
// No refractory period (REFRACTORY_PERIOD=0) for equivalence scope.

`timescale 1ns / 1ps

module sc_lif_reference #(
    parameter integer DATA_WIDTH = 16,
    parameter integer FRACTION = 8,
    parameter signed [DATA_WIDTH-1:0] V_REST      = 0,
    parameter signed [DATA_WIDTH-1:0] V_RESET     = 0,
    parameter signed [DATA_WIDTH-1:0] V_THRESHOLD = (1 << FRACTION)
)(
    input wire                            clk,
    input wire                            rst_n,
    input wire signed [DATA_WIDTH-1:0]    leak_k,
    input wire signed [DATA_WIDTH-1:0]    gain_k,
    input wire signed [DATA_WIDTH-1:0]    I_t,
    input wire signed [DATA_WIDTH-1:0]    noise_in,
    output reg                            spike_out,
    output reg signed [DATA_WIDTH-1:0]    v_out
);

    reg signed [DATA_WIDTH-1:0] v;

    wire signed [2*DATA_WIDTH-1:0] leak_product;
    wire signed [2*DATA_WIDTH-1:0] input_product;
    wire signed [DATA_WIDTH-1:0] dv_leak;
    wire signed [DATA_WIDTH-1:0] dv_input;
    wire signed [DATA_WIDTH-1:0] v_next;

    assign leak_product = (V_REST - v) * leak_k;
    assign dv_leak = leak_product >>> FRACTION;

    assign input_product = I_t * gain_k;
    assign dv_input = input_product >>> FRACTION;

    assign v_next = v + dv_leak + dv_input + noise_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v         <= V_REST;
            v_out     <= V_REST;
            spike_out <= 1'b0;
        end else begin
            if (v_next >= V_THRESHOLD) begin
                spike_out <= 1'b1;
                v         <= V_RESET;
                v_out     <= V_RESET;
            end else begin
                spike_out <= 1'b0;
                v         <= v_next;
                v_out     <= v_next;
            end
        end
    end

endmodule
