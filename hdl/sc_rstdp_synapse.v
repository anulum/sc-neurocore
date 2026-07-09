// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Reward-modulated STDP synapse for on-chip learning
//
// Extends sc_stdp_synapse with a reward signal that gates weight updates.
// Implements the three-factor learning rule:
//
//   Δw = reward × eligibility_trace × learning_rate
//
// where the eligibility trace is computed from pre/post spike timing
// correlations (standard STDP), and the reward signal comes from a
// global reward/punishment channel.
//
// This enables reinforcement-style on-chip learning with an on-chip
// reward signal and no mandatory host CPU communication in the update path.
//
// Architecture:
//   - Pre/post traces with exponential decay (Q8.8 fixed-point)
//   - Eligibility trace accumulates STDP-style correlations
//   - Eligibility itself decays with configurable time constant
//   - Weight update occurs only when reward is non-zero
//   - Weight clamped to [0, W_MAX]
//
// Compared to sc_stdp_synapse:
//   + reward input port (signed Q8.8)
//   + eligibility trace with independent decay
//   + weight update gated by reward magnitude

module sc_rstdp_synapse #(
    parameter DATA_WIDTH     = 16,
    parameter FRACTION       = 8,
    parameter W_MAX          = 16'h7F00,   // max weight (127.0 in Q8.8)
    parameter W_INIT         = 16'h0100,   // initial weight (1.0 in Q8.8)
    parameter TRACE_DECAY    = 16'h00F0,   // ~0.94 per step (pre/post trace)
    parameter ELIG_DECAY     = 16'h00F8,   // ~0.97 per step (eligibility)
    parameter A_PLUS         = 16'h0003,   // LTP amplitude (~0.01)
    parameter A_MINUS        = 16'h0004    // LTD amplitude (~0.015)
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Spike inputs
    input  wire                          pre_spike,
    input  wire                          post_spike,

    // Reward signal (signed Q8.8: positive = reward, negative = punishment)
    input  wire signed [DATA_WIDTH-1:0]  reward,

    // Outputs
    output reg  signed [DATA_WIDTH-1:0]  weight,
    output reg  signed [DATA_WIDTH-1:0]  current_out,
    output reg                           current_valid,
    output reg  signed [DATA_WIDTH-1:0]  eligibility  // exposed for monitoring
);

    // Pre- and post-synaptic traces
    reg signed [DATA_WIDTH-1:0] pre_trace;
    reg signed [DATA_WIDTH-1:0] post_trace;

    // Trace decay via fixed-point multiply
    wire signed [2*DATA_WIDTH-1:0] pre_decay_full  = pre_trace  * $signed(TRACE_DECAY);
    wire signed [2*DATA_WIDTH-1:0] post_decay_full = post_trace * $signed(TRACE_DECAY);
    wire signed [DATA_WIDTH-1:0]   pre_decayed     = pre_decay_full[DATA_WIDTH+FRACTION-1:FRACTION];
    wire signed [DATA_WIDTH-1:0]   post_decayed    = post_decay_full[DATA_WIDTH+FRACTION-1:FRACTION];

    // Eligibility trace decay
    wire signed [2*DATA_WIDTH-1:0] elig_decay_full = eligibility * $signed(ELIG_DECAY);
    wire signed [DATA_WIDTH-1:0]   elig_decayed    = elig_decay_full[DATA_WIDTH+FRACTION-1:FRACTION];

    // Weight update: reward * eligibility (fixed-point)
    wire signed [2*DATA_WIDTH-1:0] delta_full = reward * eligibility;
    wire signed [DATA_WIDTH-1:0]   delta_w    = delta_full[DATA_WIDTH+FRACTION-1:FRACTION];

    // Saturating weight update
    wire signed [DATA_WIDTH:0] weight_candidate = {weight[DATA_WIDTH-1], weight} + {delta_w[DATA_WIDTH-1], delta_w};
    wire signed [DATA_WIDTH-1:0] weight_clamped =
        (weight_candidate > $signed({1'b0, W_MAX})) ? $signed(W_MAX) :
        (weight_candidate < 0)                       ? {DATA_WIDTH{1'b0}} :
        weight_candidate[DATA_WIDTH-1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight        <= $signed(W_INIT);
            pre_trace     <= 0;
            post_trace    <= 0;
            eligibility   <= 0;
            current_out   <= 0;
            current_valid <= 0;
        end else begin
            current_valid <= 0;

            // Decay all traces
            pre_trace   <= pre_decayed;
            post_trace  <= post_decayed;
            eligibility <= elig_decayed;

            // Pre-synaptic spike: output current + accumulate LTD eligibility
            if (pre_spike) begin
                pre_trace   <= pre_trace + (1 <<< FRACTION);
                current_out <= weight;
                current_valid <= 1;

                // LTD contribution: pre fires while post trace active
                if (post_trace > 0)
                    eligibility <= elig_decayed - $signed(A_MINUS);
            end

            // Post-synaptic spike: accumulate LTP eligibility
            if (post_spike) begin
                post_trace <= post_trace + (1 <<< FRACTION);

                // LTP contribution: post fires while pre trace active
                if (pre_trace > 0)
                    eligibility <= elig_decayed + $signed(A_PLUS);
            end

            // Weight update: apply reward × eligibility
            if (reward != 0)
                weight <= weight_clamped;
        end
    end

endmodule
