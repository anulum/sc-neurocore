// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for SC Dense Matrix Layer

// Testbench for sc_dense_matrix_layer

`timescale 1ns / 1ps

module tb_sc_dense_matrix_layer;

    localparam N_INPUTS   = 4;
    localparam N_NEURONS  = 2;
    localparam DATA_WIDTH = 16;
    localparam FRACTION   = 8;
    localparam CLK_PERIOD = 10;
    localparam STREAM_LEN = 64;

    reg                                              clk;
    reg                                              rst_n;
    reg                                              start_pulse;
    reg  [31:0]                                      stream_len;
    reg  [N_INPUTS*DATA_WIDTH-1:0]                   x_input_fp;
    reg  [N_NEURONS*N_INPUTS*DATA_WIDTH-1:0]         weight_fp;
    reg  [DATA_WIDTH-1:0]                            y_min_fp;
    reg  [DATA_WIDTH-1:0]                            y_max_fp;
    reg  [DATA_WIDTH-1:0]                            cfg_leak;
    reg  [DATA_WIDTH-1:0]                            cfg_gain;
    wire [N_NEURONS-1:0]                             spikes;
    wire                                             step_valid;
    wire                                             run_done;
    wire                                             running;

    integer pass_count;
    integer fail_count;
    integer timeout_cnt;

    sc_dense_matrix_layer #(
        .N_INPUTS  (N_INPUTS),
        .N_NEURONS (N_NEURONS),
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTION  (FRACTION)
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

        // --- Test A: idle after reset ---
        if (running === 1'b0 && run_done === 1'b0) begin
            $display("[PASS] A: idle after reset");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] A: not idle after reset (running=%b, run_done=%b)", running, run_done);
            fail_count = fail_count + 1;
        end

        // Configure Q8.8 values
        // x = {0.5, 0.5, 0.5, 0.5} = 0x0080 each
        x_input_fp = {16'h0080, 16'h0080, 16'h0080, 16'h0080};
        // W[1][0..3] = 1.0, W[0][0..3] = 0.5
        // Pack: [(j*N_INPUTS+i)*DW +: DW]
        // Neuron 0 weights (0.5 each): 0x0080
        // Neuron 1 weights (1.0 each): 0x0100
        weight_fp = {16'h0100, 16'h0100, 16'h0100, 16'h0100,
                     16'h0080, 16'h0080, 16'h0080, 16'h0080};
        y_min_fp  = 16'hFF00; // -1.0 Q8.8
        y_max_fp  = 16'h0100; // +1.0 Q8.8
        cfg_leak  = 16'h00E6; // ~0.9
        cfg_gain  = 16'h0100; // 1.0
        stream_len = STREAM_LEN;

        // --- Test B: start triggers running ---
        @(posedge clk); #1;
        start_pulse = 1'b1;
        @(posedge clk); #1;
        start_pulse = 1'b0;
        @(posedge clk); #1;

        if (running === 1'b1) begin
            $display("[PASS] B: running asserted after start");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] B: running not asserted");
            fail_count = fail_count + 1;
        end

        // --- Test C: step_valid tracks running ---
        if (step_valid === running) begin
            $display("[PASS] C: step_valid matches running");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] C: step_valid=%b != running=%b", step_valid, running);
            fail_count = fail_count + 1;
        end

        // --- Test D: wait for run_done ---
        timeout_cnt = 0;
        while (!run_done && timeout_cnt < STREAM_LEN + 20) begin
            @(posedge clk); #1;
            timeout_cnt = timeout_cnt + 1;
        end

        if (run_done === 1'b1) begin
            $display("[PASS] D: run_done asserted after %0d cycles", timeout_cnt);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] D: run_done timeout at %0d cycles", timeout_cnt);
            fail_count = fail_count + 1;
        end

        // --- Test E: running deasserted ---
        if (running === 1'b0) begin
            $display("[PASS] E: running deasserted after completion");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] E: running still high");
            fail_count = fail_count + 1;
        end

        // --- Test F: run second time to verify re-startability ---
        @(posedge clk); #1;
        @(posedge clk); #1;
        start_pulse = 1'b1;
        @(posedge clk); #1;
        start_pulse = 1'b0;
        @(posedge clk); #1;

        if (running === 1'b1) begin
            $display("[PASS] F: second run started successfully");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] F: second run did not start");
            fail_count = fail_count + 1;
        end

        // Wait for second run to finish
        timeout_cnt = 0;
        while (!run_done && timeout_cnt < STREAM_LEN + 20) begin
            @(posedge clk); #1;
            timeout_cnt = timeout_cnt + 1;
        end

        if (run_done === 1'b1) begin
            $display("[PASS] F2: second run completed");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] F2: second run timed out");
            fail_count = fail_count + 1;
        end

        $display("---------------------------------------");
        $display("tb_sc_dense_matrix_layer: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("[PASS] All tests passed");
        else
            $display("[FAIL] %0d test(s) failed", fail_count);
        $finish;
    end

endmodule
