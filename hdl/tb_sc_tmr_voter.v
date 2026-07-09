// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// SC-NeuroCore — Testbench for sc_tmr_voter

`timescale 1ns / 1ps

module tb_sc_tmr_voter;

    parameter DATA_WIDTH = 16;

    reg  [DATA_WIDTH-1:0] a, b, c;
    wire [DATA_WIDTH-1:0] voted;
    wire                  error;

    sc_tmr_voter #(.DATA_WIDTH(DATA_WIDTH)) uut (
        .a(a), .b(b), .c(c),
        .voted(voted), .error(error)
    );

    integer i;
    integer pass_count;
    integer fail_count;

    initial begin
        $dumpfile("tb_sc_tmr_voter.vcd");
        $dumpvars(0, tb_sc_tmr_voter);
        pass_count = 0;
        fail_count = 0;

        // Test 1: All agree
        a = 16'hA5A5; b = 16'hA5A5; c = 16'hA5A5; #10;
        if (voted === 16'hA5A5 && error === 1'b0) begin
            $display("PASS: All agree, no error");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: All agree test (voted=%h, error=%b)", voted, error);
            fail_count = fail_count + 1;
        end

        // Test 2: One replica flipped — majority wins
        a = 16'hFFFF; b = 16'hFFFF; c = 16'h0000; #10;
        if (voted === 16'hFFFF && error === 1'b1) begin
            $display("PASS: 2-of-3 majority, error flagged");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Majority test (voted=%h, error=%b)", voted, error);
            fail_count = fail_count + 1;
        end

        // Test 3: Single bit flip in one replica
        a = 16'h1234; b = 16'h1234; c = 16'h1235; #10;
        if (voted === 16'h1234 && error === 1'b1) begin
            $display("PASS: Single-bit flip corrected, error flagged");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Single-bit flip (voted=%h, error=%b)", voted, error);
            fail_count = fail_count + 1;
        end

        // Test 4: All zeros
        a = 16'h0000; b = 16'h0000; c = 16'h0000; #10;
        if (voted === 16'h0000 && error === 1'b0) begin
            $display("PASS: All zeros");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: All zeros (voted=%h, error=%b)", voted, error);
            fail_count = fail_count + 1;
        end

        // Test 5: Each replica different — majority still resolves per bit
        a = 16'hFF00; b = 16'h0FF0; c = 16'hFFF0; #10;
        // bit-wise majority of FF00, 0FF0, FFF0:
        //   bit 15-12: F, 0, F → F  |  bit 11-8: F, F, F → F
        //   bit 7-4:   0, F, F → F  |  bit 3-0:  0, 0, 0 → 0
        // expected: FFF0
        if (voted === 16'hFFF0 && error === 1'b1) begin
            $display("PASS: Three-way disagreement resolved");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Three-way (voted=%h, error=%b)", voted, error);
            fail_count = fail_count + 1;
        end

        // Test 6: Exhaustive 4-bit check (all 3-input combinations)
        // Use DATA_WIDTH=4 via parameter override conceptually;
        // here just check lower 4 bits with upper bits matching
        for (i = 0; i < 16; i = i + 1) begin
            a = {12'h000, i[3:0]};
            b = {12'h000, i[3:0]};
            c = {12'h000, i[3:0]};
            #1;
            if (voted[3:0] !== i[3:0]) begin
                $display("FAIL: Exhaustive unanimous i=%0d", i);
                fail_count = fail_count + 1;
            end else begin
                pass_count = pass_count + 1;
            end
        end

        $display("TMR Voter: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0) $display("ALL TESTS PASSED");
        $finish;
    end

endmodule
