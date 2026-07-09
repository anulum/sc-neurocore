// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — Testbench for sc_edt_controller

`timescale 1ns / 1ps

module tb_sc_edt_controller;

    parameter DATA_WIDTH    = 16;
    parameter MARGIN        = 16'h0040;  // 0.25 in Q8.8
    parameter STABLE_CYCLES = 4;         // shorter for test

    reg                          clk;
    reg                          rst_n;
    reg                          enable;
    reg  signed [DATA_WIDTH-1:0] accumulator;
    reg  signed [DATA_WIDTH-1:0] threshold;
    wire                         decision_ready;
    wire                         decision_value;
    wire                         freeze;

    sc_edt_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .MARGIN(MARGIN),
        .STABLE_CYCLES(STABLE_CYCLES)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .accumulator(accumulator),
        .threshold(threshold),
        .decision_ready(decision_ready),
        .decision_value(decision_value),
        .freeze(freeze)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle_count;

    initial begin
        $dumpfile("tb_sc_edt_controller.vcd");
        $dumpvars(0, tb_sc_edt_controller);

        // Reset
        rst_n       = 0;
        enable      = 0;
        accumulator = 0;
        threshold   = 16'sd128;  // 0.5 in Q8.8
        #20;
        rst_n = 1;
        #10;

        // Test 1: EDT disabled — should never assert decision_ready
        enable = 0;
        accumulator = 16'sd256;  // clearly above threshold
        repeat (20) @(posedge clk);
        if (decision_ready !== 1'b0)
            $display("FAIL: decision_ready asserted while EDT disabled");
        else
            $display("PASS: EDT disabled, no premature decision");

        // Test 2: EDT enabled, accumulator well above threshold
        enable = 1;
        accumulator = 16'sd256;  // 1.0 in Q8.8, threshold 0.5
        cycle_count = 0;
        while (!decision_ready && cycle_count < 100) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end
        if (decision_ready && decision_value === 1'b1)
            $display("PASS: EDT decided positive after %0d cycles", cycle_count);
        else
            $display("FAIL: EDT did not decide positive (ready=%b, value=%b)",
                     decision_ready, decision_value);

        // Test 3: Reset and try accumulator below threshold
        rst_n = 0; #20; rst_n = 1; #10;
        enable = 1;
        accumulator = -16'sd256;  // clearly below threshold (0.5)
        cycle_count = 0;
        while (!decision_ready && cycle_count < 100) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end
        if (decision_ready && decision_value === 1'b0)
            $display("PASS: EDT decided negative after %0d cycles", cycle_count);
        else
            $display("FAIL: EDT did not decide negative (ready=%b, value=%b)",
                     decision_ready, decision_value);

        // Test 4: Accumulator near threshold — should NOT decide quickly
        rst_n = 0; #20; rst_n = 1; #10;
        enable = 1;
        accumulator = 16'sd130;  // 0.508 in Q8.8, threshold 0.5 → |diff| < MARGIN
        repeat (20) @(posedge clk);
        if (decision_ready === 1'b0)
            $display("PASS: EDT correctly deferred near-threshold decision");
        else
            $display("FAIL: EDT decided prematurely near threshold");

        // Test 5: Freeze holds
        rst_n = 0; #20; rst_n = 1; #10;
        enable = 1;
        accumulator = 16'sd512;
        repeat (20) @(posedge clk);
        if (freeze) begin
            accumulator = -16'sd512;  // flip input after freeze
            repeat (10) @(posedge clk);
            if (freeze && decision_value === 1'b1)
                $display("PASS: Freeze holds decision despite input change");
            else
                $display("FAIL: Freeze did not hold");
        end

        $display("All EDT tests complete.");
        $finish;
    end

endmodule
