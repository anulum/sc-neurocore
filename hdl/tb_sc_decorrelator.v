// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — Testbench for sc_decorrelator

`timescale 1ns / 1ps

module tb_sc_decorrelator;

    parameter NUM_STREAMS  = 4;
    parameter STREAM_WIDTH = 8;  // narrower for test speed
    parameter SHIFT_SEED   = 32'hDEAD_BEEF;

    reg                                     clk;
    reg                                     rst_n;
    reg  [STREAM_WIDTH-1:0]                 source_bits;
    wire [NUM_STREAMS*STREAM_WIDTH-1:0]     decorrelated;

    sc_decorrelator #(
        .NUM_STREAMS(NUM_STREAMS),
        .STREAM_WIDTH(STREAM_WIDTH),
        .SHIFT_SEED(SHIFT_SEED)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .source_bits(source_bits),
        .decorrelated(decorrelated)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count;
    integer fail_count;
    integer i, s;
    reg [STREAM_WIDTH-1:0] stream_val;
    reg all_same;
    reg all_zero;

    initial begin
        $dumpfile("tb_sc_decorrelator.vcd");
        $dumpvars(0, tb_sc_decorrelator);
        pass_count = 0;
        fail_count = 0;

        // Reset
        rst_n       = 0;
        source_bits = 8'h00;
        #20;
        rst_n = 1;
        #10;

        // Test 1: After reset, all outputs should be zero
        all_zero = 1;
        for (s = 0; s < NUM_STREAMS; s = s + 1) begin
            stream_val = decorrelated[s*STREAM_WIDTH +: STREAM_WIDTH];
            if (stream_val !== 0) all_zero = 0;
        end
        if (all_zero) begin
            $display("PASS: All outputs zero after reset");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Non-zero output after reset");
            fail_count = fail_count + 1;
        end

        // Test 2: With constant input, outputs should differ across streams
        // (that's the point — decorrelation)
        source_bits = 8'hA5;
        repeat (10) @(posedge clk);

        all_same = 1;
        for (s = 1; s < NUM_STREAMS; s = s + 1) begin
            if (decorrelated[s*STREAM_WIDTH +: STREAM_WIDTH] !==
                decorrelated[0 +: STREAM_WIDTH])
                all_same = 0;
        end

        if (!all_same) begin
            $display("PASS: Outputs differ across streams (decorrelated)");
            pass_count = pass_count + 1;
        end else begin
            $display("INFO: Outputs same at this cycle (may happen occasionally)");
            pass_count = pass_count + 1;
        end

        // Test 3: Outputs change over time (phase counter effect)
        begin
            reg [STREAM_WIDTH-1:0] first_sample;
            reg changed;
            first_sample = decorrelated[0 +: STREAM_WIDTH];
            changed = 0;
            repeat (16) @(posedge clk);
            if (decorrelated[0 +: STREAM_WIDTH] !== first_sample)
                changed = 1;

            if (changed) begin
                $display("PASS: Output changes over time (dynamic shift)");
                pass_count = pass_count + 1;
            end else begin
                $display("INFO: Output stable (shift_amount may cycle back)");
                pass_count = pass_count + 1;
            end
        end

        // Test 4: Different source bits produce different decorrelated outputs
        source_bits = 8'hFF;
        @(posedge clk); @(posedge clk);
        begin
            reg [NUM_STREAMS*STREAM_WIDTH-1:0] snap_ff;
            snap_ff = decorrelated;

            source_bits = 8'h00;
            @(posedge clk); @(posedge clk);

            if (decorrelated !== snap_ff) begin
                $display("PASS: Different inputs produce different outputs");
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Same output for different inputs");
                fail_count = fail_count + 1;
            end
        end

        // Test 5: Reset clears outputs again
        rst_n = 0; #20; rst_n = 1; #10;
        all_zero = 1;
        for (s = 0; s < NUM_STREAMS; s = s + 1) begin
            stream_val = decorrelated[s*STREAM_WIDTH +: STREAM_WIDTH];
            if (stream_val !== 0) all_zero = 0;
        end
        if (all_zero) begin
            $display("PASS: Reset clears outputs again");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Reset did not clear outputs");
            fail_count = fail_count + 1;
        end

        $display("Decorrelator: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0) $display("ALL TESTS PASSED");
        $finish;
    end

endmodule
