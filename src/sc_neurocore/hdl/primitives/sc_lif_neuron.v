// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fixed-point Leaky Integrate-and-Fire neuron for SC pipelines

//hdl/sc_lif_neuron.v
//
// Fixed-point Leaky Integrate-and-Fire neuron for SC pipelines.
//
// - I_t: fixed-point input current (e.g. from sc_dotproduct_to_current)
// - Internal state v: fixed-point membrane potential
// - Parameters define leak rate, input gain, and thresholds
//
// Update per clock:
//   dv_leak = (V_REST - v) * ALPHA_LEAK >> FRACTION
//   dv_in   = I_t * GAIN_IN      >> FRACTION
//   v_next  = v + dv_leak + dv_in
//
// If v_next >= V_THRESHOLD -> spike, reset to V_RESET
//
// All values are signed two's complement with DATA_WIDTH bits.
// Scale: Q(FRACTION) fixed-point.

`timescale 1ns / 1ps

module sc_lif_neuron #(
    parameter integer DATA_WIDTH = 16,
    parameter integer FRACTION = 8,

    // Membrane parameters in Q(FRACTION) fixed-point
    parameter signed [DATA_WIDTH-1:0] V_REST      = 0,                   // 0.0
    parameter signed [DATA_WIDTH-1:0] V_RESET     = 0,                   // 0.0
    parameter signed [DATA_WIDTH-1:0] V_THRESHOLD = (1 << FRACTION),     // 1.0 in Q(FRACTION)
    parameter integer REFRACTORY_PERIOD = 0                              // Cycles
)(
    input wire                            clk,
    input wire                            rst_n,

    // Runtime Configuration (Q(FRACTION))
    input wire signed [DATA_WIDTH-1:0]    leak_k,  // Was ALPHA_LEAK
    input wire signed [DATA_WIDTH-1:0]    gain_k,  // Was GAIN_IN

    // Fixed-point input current I_t (Q(FRACTION))
    input wire signed [DATA_WIDTH-1:0]    I_t,

    // External Noise Input (Q(FRACTION))
    input wire signed [DATA_WIDTH-1:0]    noise_in,

    // Output spike: 1 when v crosses threshold, else 0
    output reg                            spike_out,

    // Optional: expose membrane potential for debug
    output reg signed [DATA_WIDTH-1:0]    v_out
);

// Internal membrane potential
reg signed [DATA_WIDTH-1:0] v_reg;

// Refractory counter
reg [31:0] refractory_counter;

// Intermediate wide products for fixed-point math
wire signed [2*DATA_WIDTH-1:0] leak_mul;
wire signed [2*DATA_WIDTH-1:0] in_mul;

// Shifted (scaled back) increments
wire signed [DATA_WIDTH-1:0] dv_leak;
wire signed [DATA_WIDTH-1:0] dv_in;

// Next membrane potential
wire signed [DATA_WIDTH-1:0] v_next;


// Compute (V_REST - v) * leak_k
assign leak_mul = (V_REST - v_reg) * leak_k;
assign dv_leak  = leak_mul >>> FRACTION;

// Compute I_t * gain_k
assign in_mul = I_t * gain_k;
assign dv_in  = in_mul >>> FRACTION;

// v_next includes noise
assign v_next = v_reg + dv_leak + dv_in + noise_in;


// Sequential update
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        v_reg     <= V_REST;
        v_out     <= V_REST;
        spike_out <= 1'b0;
        refractory_counter <= 0;
    end else begin
        if (refractory_counter > 0) begin
            // In refractory period
            refractory_counter <= refractory_counter - 1;
            spike_out <= 1'b0;
            v_reg <= V_REST; // Clamp to rest during refractory (matches Python model)
            v_out <= V_REST;
        end else begin
            // Threshold check on next potential
            if (v_next >= V_THRESHOLD) begin
                spike_out <= 1'b1;
                v_reg     <= V_RESET;
                v_out     <= V_RESET;
                refractory_counter <= REFRACTORY_PERIOD;
            end else begin
                spike_out <= 1'b0;
                v_reg     <= v_next;
                v_out     <= v_next;
            end
        end
    end
end

endmodule
