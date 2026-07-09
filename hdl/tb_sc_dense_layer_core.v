// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for SC Dense Layer Core

// Testbench for sc_dense_layer_core

`timescale 1ns / 1ps

module tb_sc_dense_layer_core;

    localparam N_INPUTS   = 3;
    localparam N_NEURONS  = 5;
    localparam DATA_WIDTH = 16;
    localparam CLK_PERIOD = 10;
    localparam STREAM_LEN = 64;

    reg                                clk;
    reg                                rst_n;
    reg                                start_pulse;
    reg  [31:0]                        stream_len;
    reg  [N_INPUTS*DATA_WIDTH-1:0]     x_input_fp;
    reg  [N_INPUTS*DATA_WIDTH-1:0]     weight_fp;
    reg  [DATA_WIDTH-1:0]              y_min_fp;
    reg  [DATA_WIDTH-1:0]              y_max_fp;
    reg  [DATA_WIDTH-1:0]              cfg_leak;
    reg  [DATA_WIDTH-1:0]              cfg_gain;
    wire [DATA_WIDTH-1:0]              I_t;
    wire [N_NEURONS-1:0]               spikes;
    wire                               step_valid;
    wire                               run_done;
    wire                               running;

    integer pass_count;
    integer fail_count;
    integer timeout_cnt;

    sc_dense_layer_core #(
        .N_INPUTS  (N_INPUTS),
        .N_NEURONS (N_NEURONS),
        .DATA_WIDTH(DATA_WIDTH)
    ) uut (
        .clk         (clk),
        .rst_n       (rst_n),
        .start_pulse (start_pulse),
        .stream_len  (stream_len),
        .x_input_fp  (x_input_fp),
        .weight_fp   (weight_fp),
        .y_min_fp    (y_min_fp),
        .y_max_fp    (y_max_fp),
        .cfg_leak    (cfg_leak),
        .cfg_gain    (cfg_gain),
        .I_t         (I_t),
        .spikes      (spikes),
        .step_valid  (step_valid),
        .run_done    (run_done),
        .running     (running)
    );

    always #(CLK_PERIOD/2) clk = ~clk;

    initial begin
        clk         = 0;
        rst_n       = 0;
        start_pulse = 0;
        stream_len  = STREAM_LEN;
        x_input_fp  = 0;
        weight_fp   = 0;
        y_min_fp    = 0;
        y_max_fp    = 0;
        cfg_leak    = 0;
        cfg_gain    = 0;
        pass_count  = 0;
        fail_count  = 0;

        // Reset
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #1;

        // --- Test A: FSM idle after reset ---
        if (running === 1'b0 && run_done === 1'b0) begin
            $display("[PASS] A: idle after reset (running=0, run_done=0)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] A: not idle after reset (running=%b, run_done=%b)", running, run_done);
            fail_count = fail_count + 1;
        end

        // Configure Q8.8 values
        // x = {0.5, 0.5, 0.5} = 0x0080 each
        x_input_fp = {16'h0080, 16'h0080, 16'h0080};
        // w = {1.0, 1.0, 1.0} = 0x0100 each
        weight_fp  = {16'h0100, 16'h0100, 16'h0100};
        // y_min = -1.0 (0xFF00), y_max = +1.0 (0x0100)
        y_min_fp   = 16'hFF00;
        y_max_fp   = 16'h0100;
        // leak = 0.9 ~ 0x00E6, gain = 1.0 = 0x0100
        cfg_leak   = 16'h00E6;
        cfg_gain   = 16'h0100;
        stream_len = STREAM_LEN;

        // --- Test B: start_pulse triggers running ---
        @(posedge clk); #1;
        start_pulse = 1'b1;
        @(posedge clk); #1;
        start_pulse = 1'b0;

        // Wait one cycle for FSM to register
        @(posedge clk); #1;
        if (running === 1'b1) begin
            $display("[PASS] B: running asserted after start_pulse");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] B: running not asserted after start_pulse");
            fail_count = fail_count + 1;
        end

        // --- Test C: wait for run_done ---
        timeout_cnt = 0;
        while (!run_done && timeout_cnt < STREAM_LEN + 20) begin
            @(posedge clk); #1;
            timeout_cnt = timeout_cnt + 1;
        end

        if (run_done === 1'b1) begin
            $display("[PASS] C: run_done asserted after %0d cycles", timeout_cnt);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] C: run_done not asserted (timeout at %0d)", timeout_cnt);
            fail_count = fail_count + 1;
        end

        // --- Test D: running deasserted after completion ---
        if (running === 1'b0) begin
            $display("[PASS] D: running deasserted after run_done");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] D: running still high after run_done");
            fail_count = fail_count + 1;
        end

        // --- Test E: run_done clears on next idle cycle ---
        @(posedge clk); #1;
        if (run_done === 1'b0) begin
            $display("[PASS] E: run_done cleared after one idle cycle");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] E: run_done still high after idle cycle");
            fail_count = fail_count + 1;
        end

        $display("---------------------------------------");
        $display("tb_sc_dense_layer_core: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("[PASS] All tests passed");
        else
            $display("[FAIL] %0d test(s) failed", fail_count);
        $finish;
    end

endmodule
