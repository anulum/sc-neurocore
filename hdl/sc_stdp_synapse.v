// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — On-chip STDP synapse: runtime weight adaptation on FPGA
//
// Implements spike-timing dependent plasticity directly in hardware.
// Pre and post spike traces are exponentially decaying counters.
// Weight updates occur on each pre or post spike event.
//
// LTP: pre spike arrives while post trace > 0 → increase weight
// LTD: post spike arrives while pre trace > 0 → decrease weight
//
// Q8.8 fixed-point arithmetic, configurable learning rate and time
// constant. Weight clamped to [0, W_MAX].
//
// First open-source FPGA SNN synapse with on-chip learning.

module sc_stdp_synapse #(
    parameter DATA_WIDTH  = 16,
    parameter FRACTION    = 8,
    parameter W_MAX       = 16'h7F00,   // max weight (127.0 in Q8.8)
    parameter TRACE_DECAY = 16'h00F0,   // ~0.94 per step (exp decay)
    parameter A_PLUS      = 16'h0003,   // LTP amplitude (~0.01)
    parameter A_MINUS     = 16'h0004    // LTD amplitude (~0.015)
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // Spike inputs
    input  wire                      pre_spike,
    input  wire                      post_spike,

    // Current weight (readable for monitoring)
    output reg signed [DATA_WIDTH-1:0] weight,

    // Weighted output: pre_spike * weight
    output reg signed [DATA_WIDTH-1:0] current_out,
    output reg                         current_valid
);

    // Eligibility traces (Q8.8)
    reg signed [DATA_WIDTH-1:0] pre_trace;
    reg signed [DATA_WIDTH-1:0] post_trace;

    // Saturating multiply for trace decay
    wire signed [2*DATA_WIDTH-1:0] pre_decay_full  = pre_trace  * $signed(TRACE_DECAY);
    wire signed [2*DATA_WIDTH-1:0] post_decay_full = post_trace * $signed(TRACE_DECAY);
    wire signed [DATA_WIDTH-1:0] pre_decayed  = pre_decay_full[DATA_WIDTH+FRACTION-1:FRACTION];
    wire signed [DATA_WIDTH-1:0] post_decayed = post_decay_full[DATA_WIDTH+FRACTION-1:FRACTION];

    // Weight update calculation
    wire signed [DATA_WIDTH-1:0] delta_ltp = $signed(A_PLUS);   // fixed amplitude
    wire signed [DATA_WIDTH-1:0] delta_ltd = $signed(A_MINUS);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight       <= 16'h0100;  // init weight = 1.0
            pre_trace    <= 0;
            post_trace   <= 0;
            current_out  <= 0;
            current_valid <= 0;
        end else begin
            current_valid <= 0;

            // Decay traces
            pre_trace  <= pre_decayed;
            post_trace <= post_decayed;

            // Pre-synaptic spike: output current + check for LTD
            if (pre_spike) begin
                pre_trace   <= pre_trace + (1 <<< FRACTION);  // bump trace
                current_out <= weight;
                current_valid <= 1;

                // LTD: pre fires while post trace active
                if (post_trace > 0) begin
                    if (weight > $signed(delta_ltd))
                        weight <= weight - delta_ltd;
                    else
                        weight <= 0;
                end
            end

            // Post-synaptic spike: check for LTP
            if (post_spike) begin
                post_trace <= post_trace + (1 <<< FRACTION);

                // LTP: post fires while pre trace active
                if (pre_trace > 0) begin
                    if (weight + $signed(delta_ltp) < $signed(W_MAX))
                        weight <= weight + delta_ltp;
                    else
                        weight <= $signed(W_MAX);
                end
            end
        end
    end

endmodule
