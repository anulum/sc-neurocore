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
        .rate_q16(rate_q16)
    );

`ifdef FORMAL
    reg past_valid = 0;
    always @(posedge clk)
        past_valid <= 1;

    // Protocol: run_active and run_done are mutually exclusive
    always @* assume(!(run_active && run_done));

    // 1. After reset, all accumulators are zero
    always @(posedge clk) begin
        if (past_valid && !rst_n) begin
            assert(uut.accumulators[0] == 0);
            assert(uut.accumulators[1] == 0);
            assert(uut.accumulators[2] == 0);
            assert(uut.accumulators[3] == 0);
            assert(uut.accumulators[4] == 0);
            assert(uut.accumulators[5] == 0);
            assert(uut.accumulators[6] == 0);
        end
    end

    // 2. Accumulators never overflow: bounded by CNT_WIDTH max (2^16 - 1)
    //    Since each tick adds at most 1 per neuron, the accumulator can
    //    reach at most stream_len. Prove the weaker invariant: acc < 2^16.
    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            assert(uut.accumulators[0] < 16'hFFFF);
            assert(uut.accumulators[1] < 16'hFFFF);
            assert(uut.accumulators[2] < 16'hFFFF);
            assert(uut.accumulators[3] < 16'hFFFF);
            assert(uut.accumulators[4] < 16'hFFFF);
            assert(uut.accumulators[5] < 16'hFFFF);
            assert(uut.accumulators[6] < 16'hFFFF);
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
            cover(uut.accumulators[0] > 0);
    end
`endif

endmodule
