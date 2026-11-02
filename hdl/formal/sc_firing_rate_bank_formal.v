// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal Verification for SC Firing Rate Bank

`default_nettype none

module sc_firing_rate_bank_formal (
    input wire        clk,
    input wire        rst_n,
    input wire [6:0]  spikes,
    input wire        step_valid,
    input wire        run_active,
    input wire        run_done,
    input wire [31:0] SCALE_Q16
);

    wire [31:0] rate_q16 [0:6];
    wire [111:0] formal_accumulators;
    wire [15:0] formal_accumulator_0 = formal_accumulators[0 +: 16];
    wire [15:0] formal_accumulator_1 = formal_accumulators[16 +: 16];
    wire [15:0] formal_accumulator_2 = formal_accumulators[32 +: 16];
    wire [15:0] formal_accumulator_3 = formal_accumulators[48 +: 16];
    wire [15:0] formal_accumulator_4 = formal_accumulators[64 +: 16];
    wire [15:0] formal_accumulator_5 = formal_accumulators[80 +: 16];
    wire [15:0] formal_accumulator_6 = formal_accumulators[96 +: 16];


    sc_firing_rate_bank #(
        .N_NEURONS(7),
        .CNT_WIDTH(16),
        .SCALE_WIDTH(32)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .spikes(spikes),
        .step_valid(step_valid),
        .run_active(run_active),
        .run_done(run_done),
        .SCALE_Q16(SCALE_Q16),
        .rate_q16(rate_q16),
        .formal_accumulators(formal_accumulators)
    );

`ifdef FORMAL
    reg past_valid = 0;
    // Establish only the first reset; later inputs stay unconstrained.
    always @(posedge clk) if (!past_valid) assume(!rst_n);
    always @(posedge clk)
        past_valid <= 1;

    // Protocol: run_active and run_done are mutually exclusive
    always @* assume(!(run_active && run_done));

    // 1. After reset, all accumulators are zero
    always @(posedge clk) begin
        if (past_valid && !rst_n) begin
            assert(formal_accumulator_0 == 0);
            assert(formal_accumulator_1 == 0);
            assert(formal_accumulator_2 == 0);
            assert(formal_accumulator_3 == 0);
            assert(formal_accumulator_4 == 0);
            assert(formal_accumulator_5 == 0);
            assert(formal_accumulator_6 == 0);
        end
    end

    // 2. Accumulators never overflow: bounded by CNT_WIDTH max (2^16 - 1)
    //    Since each tick adds at most 1 per neuron, the accumulator can
    //    reach at most stream_len. Prove the weaker invariant: acc < 2^16.
    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            assert(formal_accumulator_0 < 16'hFFFF);
            assert(formal_accumulator_1 < 16'hFFFF);
            assert(formal_accumulator_2 < 16'hFFFF);
            assert(formal_accumulator_3 < 16'hFFFF);
            assert(formal_accumulator_4 < 16'hFFFF);
            assert(formal_accumulator_5 < 16'hFFFF);
            assert(formal_accumulator_6 < 16'hFFFF);
        end
    end

    // 3. rate_q16 only updates on run_done pulse
    //    When run_done is low and run_active is low, rate_q16 holds its value
    always @(posedge clk) begin
        if (past_valid && rst_n && !$past(run_done) && !$past(!rst_n)) begin
            assert(rate_q16[0] == $past(rate_q16[0]));
            assert(rate_q16[1] == $past(rate_q16[1]));
            assert(rate_q16[2] == $past(rate_q16[2]));
            assert(rate_q16[3] == $past(rate_q16[3]));
            assert(rate_q16[4] == $past(rate_q16[4]));
            assert(rate_q16[5] == $past(rate_q16[5]));
            assert(rate_q16[6] == $past(rate_q16[6]));
        end
    end

    // 4. Cover: accumulator counts spikes
    always @(posedge clk) begin
        if (past_valid && rst_n)
            cover(formal_accumulator_0 > 0);
    end
`endif

endmodule
