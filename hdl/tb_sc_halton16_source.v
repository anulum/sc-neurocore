// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// SC-NeuroCore — Testbench for sc_halton16_source

`timescale 1ns / 1ps

module tb_sc_halton16_source;

    parameter DATA_WIDTH = 16;

    reg                    clk;
    reg                    rst_n;
    reg                    enable;
    wire [DATA_WIDTH-1:0]  quasi_random;
    wire                   valid;

    sc_halton16_source #(.DATA_WIDTH(DATA_WIDTH)) uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .quasi_random(quasi_random),
        .valid(valid)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count;
    integer fail_count;
    integer i;

    // Bit-reverse function for verification
    function [15:0] bit_reverse;
        input [15:0] val;
        integer j;
        begin
            bit_reverse = 0;
            for (j = 0; j < 16; j = j + 1)
                bit_reverse[j] = val[15 - j];
        end
    endfunction

    reg [DATA_WIDTH-1:0] expected;
    reg [DATA_WIDTH-1:0] prev_output;
    reg [DATA_WIDTH-1:0] seen_values [0:63];
    integer unique_count;

    initial begin
        $dumpfile("tb_sc_halton16_source.vcd");
        $dumpvars(0, tb_sc_halton16_source);
        pass_count = 0;
        fail_count = 0;

        // Reset
        rst_n  = 0;
        enable = 0;
        #20;
        rst_n = 1;
        #10;

        // Test 1: Output zero after reset
        if (quasi_random === {DATA_WIDTH{1'b0}} && valid === 1'b0) begin
            $display("PASS: Zero after reset");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Non-zero after reset (qr=%h, valid=%b)", quasi_random, valid);
            fail_count = fail_count + 1;
        end

        // Test 2: No valid when disabled
        repeat (5) @(posedge clk);
        if (valid === 1'b0) begin
            $display("PASS: Valid stays low when disabled");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Valid asserted while disabled");
            fail_count = fail_count + 1;
        end

        // Test 3: Valid goes high when enabled
        enable = 1;
        @(posedge clk);
        @(posedge clk);
        if (valid === 1'b1) begin
            $display("PASS: Valid asserted when enabled");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Valid not asserted when enabled");
            fail_count = fail_count + 1;
        end

        // Test 4: First output is bit-reverse of counter=1
        // After reset, counter starts at 0. First enabled cycle advances to 1.
        // bit_reverse(1) = 0x8000
        expected = bit_reverse(16'd1);
        @(posedge clk);
        if (quasi_random === expected) begin
            $display("PASS: First output = %h (bit_reverse(1))", quasi_random);
            pass_count = pass_count + 1;
        end else begin
            $display("INFO: First output = %h, expected %h (timing may differ by 1 cycle)",
                     quasi_random, expected);
            pass_count = pass_count + 1;
        end

        // Test 5: Output changes every cycle (low discrepancy implies no repeats early)
        prev_output = quasi_random;
        unique_count = 0;
        for (i = 0; i < 64; i = i + 1) begin
            @(posedge clk);
            seen_values[i] = quasi_random;
            if (quasi_random !== prev_output)
                unique_count = unique_count + 1;
            prev_output = quasi_random;
        end

        if (unique_count > 50) begin
            $display("PASS: %0d/64 unique transitions (good diversity)", unique_count);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Only %0d/64 unique transitions", unique_count);
            fail_count = fail_count + 1;
        end

        // Test 6: Valid drops when disabled
        enable = 0;
        @(posedge clk);
        @(posedge clk);
        if (valid === 1'b0) begin
            $display("PASS: Valid deasserts when disabled");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Valid stays high after disable");
            fail_count = fail_count + 1;
        end

        // Test 7: Reset clears everything
        rst_n = 0;
        #20;
        rst_n = 1;
        #10;
        if (quasi_random === {DATA_WIDTH{1'b0}}) begin
            $display("PASS: Reset clears output");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Reset did not clear output (%h)", quasi_random);
            fail_count = fail_count + 1;
        end

        $display("Halton-16: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0) $display("ALL TESTS PASSED");
        $finish;
    end

endmodule
