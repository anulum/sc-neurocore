// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — Testbench for sc_masking_shield

`timescale 1ns / 1ps

module tb_sc_masking_shield;

    parameter DATA_WIDTH = 16;

    reg                    clk;
    reg                    rst_n;
    reg                    enable;
    reg  [DATA_WIDTH-1:0]  data_in;
    reg  [DATA_WIDTH-1:0]  mask_rng;
    wire [DATA_WIDTH-1:0]  share_a;
    wire [DATA_WIDTH-1:0]  share_b;
    wire [DATA_WIDTH-1:0]  recombined;
    wire                   shares_valid;

    sc_masking_shield #(.DATA_WIDTH(DATA_WIDTH)) uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .data_in(data_in),
        .mask_rng(mask_rng),
        .share_a(share_a),
        .share_b(share_b),
        .recombined(recombined),
        .shares_valid(shares_valid)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count;
    integer fail_count;
    integer i;
    reg [DATA_WIDTH-1:0] test_data;
    reg [DATA_WIDTH-1:0] test_mask;

    initial begin
        $dumpfile("tb_sc_masking_shield.vcd");
        $dumpvars(0, tb_sc_masking_shield);
        pass_count = 0;
        fail_count = 0;

        // Reset
        rst_n    = 0;
        enable   = 0;
        data_in  = 0;
        mask_rng = 0;
        #20;
        rst_n = 1;
        #10;

        // Test 1: Shares zero after reset
        if (share_a === 0 && share_b === 0 && shares_valid === 0) begin
            $display("PASS: Shares zero after reset");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Non-zero after reset");
            fail_count = fail_count + 1;
        end

        // Test 2: Recombination correctness with known values
        enable   = 1;
        data_in  = 16'hCAFE;
        mask_rng = 16'hBEEF;
        @(posedge clk);
        @(posedge clk);

        if (recombined === 16'hCAFE) begin
            $display("PASS: Recombined = 0x%h (matches data_in)", recombined);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Recombined = 0x%h (expected 0xCAFE)", recombined);
            fail_count = fail_count + 1;
        end

        // Test 3: Shares differ from data_in (masking actually works)
        if (share_a !== data_in && share_b !== data_in) begin
            $display("PASS: Neither share equals data_in (masked)");
            pass_count = pass_count + 1;
        end else if (mask_rng === 0) begin
            $display("INFO: Trivial mask (all zeros) — shares may equal data");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: A share equals data_in (no masking)");
            fail_count = fail_count + 1;
        end

        // Test 4: shares_valid is asserted
        if (shares_valid === 1'b1) begin
            $display("PASS: shares_valid asserted when enabled");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: shares_valid not asserted");
            fail_count = fail_count + 1;
        end

        // Test 5: Sweep 100 random values — recombination always correct
        begin
            reg all_correct;
            all_correct = 1;

            for (i = 0; i < 100; i = i + 1) begin
                // Pseudo-random test vectors
                data_in  = $random;
                mask_rng = $random;
                @(posedge clk);
                @(posedge clk);
                if (recombined !== data_in) begin
                    $display("  Recombination mismatch at i=%0d: got %h, expected %h",
                             i, recombined, data_in);
                    all_correct = 0;
                end
            end

            if (all_correct) begin
                $display("PASS: 100/100 random recombinations correct");
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Some recombinations incorrect");
                fail_count = fail_count + 1;
            end
        end

        // Test 6: Disable stops valid
        enable = 0;
        @(posedge clk);
        @(posedge clk);
        if (shares_valid === 1'b0) begin
            $display("PASS: shares_valid deasserts when disabled");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: shares_valid stays high after disable");
            fail_count = fail_count + 1;
        end

        // Test 7: share_b always equals mask_rng (by construction)
        enable   = 1;
        data_in  = 16'h1234;
        mask_rng = 16'h5678;
        @(posedge clk);
        @(posedge clk);
        if (share_b === 16'h5678) begin
            $display("PASS: share_b = mask_rng");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: share_b != mask_rng (%h)", share_b);
            fail_count = fail_count + 1;
        end

        $display("Masking Shield: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0) $display("ALL TESTS PASSED");
        $finish;
    end

endmodule
