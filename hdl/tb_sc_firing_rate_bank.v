// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for SC Firing Rate Bank

// Testbench for sc_firing_rate_bank

`timescale 1ns / 1ps

module tb_sc_firing_rate_bank;

    localparam N_NEURONS    = 4;
    localparam CNT_WIDTH    = 16;
    localparam SCALE_WIDTH  = 32;
    localparam CLK_PERIOD   = 10;

    reg                       clk;
    reg                       rst_n;
    reg  [N_NEURONS-1:0]      spikes;
    reg                       step_valid;
    reg                       run_active;
    reg                       run_done;
    reg  [SCALE_WIDTH-1:0]    SCALE_Q16;
    wire [31:0]               rate_q16 [0:N_NEURONS-1];

    integer pass_count;
    integer fail_count;
    integer i;

    sc_firing_rate_bank #(
        .N_NEURONS  (N_NEURONS),
        .CNT_WIDTH  (CNT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH)
    ) uut (
        .clk        (clk),
        .rst_n      (rst_n),
        .spikes     (spikes),
        .step_valid (step_valid),
        .run_active (run_active),
        .run_done   (run_done),
        .SCALE_Q16  (SCALE_Q16),
        .rate_q16   (rate_q16)
    );

    always #(CLK_PERIOD/2) clk = ~clk;

    // Drive spikes for N cycles while run_active, then pulse run_done
    task run_spike_phase;
        input integer num_steps;
        input [N_NEURONS-1:0] spike_pattern;
        integer s;
        begin
            run_active = 1'b1;
            run_done   = 1'b0;
            for (s = 0; s < num_steps; s = s + 1) begin
                spikes     = spike_pattern;
                step_valid = 1'b1;
                @(posedge clk); #1;
            end
            spikes     = {N_NEURONS{1'b0}};
            step_valid = 1'b0;
            run_active = 1'b0;
            @(posedge clk); #1;
            // Pulse run_done
            run_done = 1'b1;
            @(posedge clk); #1;
            run_done = 1'b0;
            @(posedge clk); #1;
        end
    endtask

    initial begin
        clk        = 0;
        rst_n      = 0;
        spikes     = 0;
        step_valid = 0;
        run_active = 0;
        run_done   = 0;
        SCALE_Q16  = 32'd0;
        pass_count = 0;
        fail_count = 0;

        // Reset
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #1;

        // --- Test A: 10 spikes on neuron 0 only, SCALE=1 (trivial) ---
        SCALE_Q16 = 32'd1;
        run_spike_phase(10, 4'b0001);

        if (rate_q16[0] == 32'd10) begin
            $display("[PASS] A: neuron0 rate=%0d (expected 10)", rate_q16[0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] A: neuron0 rate=%0d (expected 10)", rate_q16[0]);
            fail_count = fail_count + 1;
        end
        if (rate_q16[1] == 32'd0) begin
            $display("[PASS] A: neuron1 rate=%0d (expected 0)", rate_q16[1]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] A: neuron1 rate=%0d (expected 0)", rate_q16[1]);
            fail_count = fail_count + 1;
        end

        // --- Test B: 8 spikes on all neurons, SCALE=65536 (Q16.16 = 1.0) ---
        SCALE_Q16 = 32'd65536;
        run_spike_phase(8, 4'b1111);

        // Expected: count=8, rate = 8 * 65536 = 524288
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            if (rate_q16[i] == 32'd524288) begin
                $display("[PASS] B: neuron%0d rate=%0d (expected 524288)", i, rate_q16[i]);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] B: neuron%0d rate=%0d (expected 524288)", i, rate_q16[i]);
                fail_count = fail_count + 1;
            end
        end

        // --- Test C: after run_done, accumulators cleared -> next run starts fresh ---
        SCALE_Q16 = 32'd1;
        run_spike_phase(5, 4'b0010); // 5 spikes on neuron 1

        if (rate_q16[1] == 32'd5) begin
            $display("[PASS] C: fresh-run neuron1 rate=%0d (expected 5)", rate_q16[1]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] C: fresh-run neuron1 rate=%0d (expected 5)", rate_q16[1]);
            fail_count = fail_count + 1;
        end
        if (rate_q16[0] == 32'd0) begin
            $display("[PASS] C: fresh-run neuron0 rate=%0d (expected 0)", rate_q16[0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] C: fresh-run neuron0 rate=%0d (expected 0)", rate_q16[0]);
            fail_count = fail_count + 1;
        end

        $display("---------------------------------------");
        $display("tb_sc_firing_rate_bank: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("[PASS] All tests passed");
        else
            $display("[FAIL] %0d test(s) failed", fail_count);
        $finish;
    end

endmodule
