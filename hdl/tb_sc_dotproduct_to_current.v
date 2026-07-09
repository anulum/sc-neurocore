// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for SC Dotproduct To Current

// Testbench for sc_dotproduct_to_current

`timescale 1ns / 1ps

module tb_sc_dotproduct_to_current;

    localparam N_INPUTS   = 3;
    localparam DATA_WIDTH = 16;

    reg  [N_INPUTS-1:0]          post_bits;
    reg  signed [DATA_WIDTH-1:0] y_min;
    reg  signed [DATA_WIDTH-1:0] y_max;
    wire signed [DATA_WIDTH-1:0] I_t;

    integer pass_count;
    integer fail_count;

    sc_dotproduct_to_current #(
        .N_INPUTS  (N_INPUTS),
        .DATA_WIDTH(DATA_WIDTH)
    ) uut (
        .post_bits (post_bits),
        .y_min     (y_min),
        .y_max     (y_max),
        .I_t       (I_t)
    );

    initial begin
        pass_count = 0;
        fail_count = 0;

        // Q8.8: y_min = -1.0 = 0xFF00, y_max = +1.0 = 0x0100
        y_min = 16'shFF00;
        y_max = 16'sh0100;

        // --- Test A: all bits 0 -> I_t == y_min ---
        post_bits = 3'b000;
        #10;
        if (I_t === y_min) begin
            $display("[PASS] A: all-zero -> I_t == y_min (%0d)", I_t);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] A: all-zero -> I_t=%0d, expected %0d", I_t, y_min);
            fail_count = fail_count + 1;
        end

        // --- Test B: all bits 1 -> I_t == y_max ---
        post_bits = 3'b111;
        #10;
        if (I_t === y_max) begin
            $display("[PASS] B: all-ones -> I_t == y_max (%0d)", I_t);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] B: all-ones -> I_t=%0d, expected %0d", I_t, y_max);
            fail_count = fail_count + 1;
        end

        // --- Test C: one bit set -> I_t strictly between y_min and y_max ---
        post_bits = 3'b010;
        #10;
        if (I_t > y_min && I_t < y_max) begin
            $display("[PASS] C: one-bit -> I_t=%0d in (%0d, %0d)", I_t, y_min, y_max);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] C: one-bit -> I_t=%0d not in (%0d, %0d)", I_t, y_min, y_max);
            fail_count = fail_count + 1;
        end

        // --- Test D: two bits set -> between one-bit and all-ones ---
        post_bits = 3'b110;
        #10;
        // Expected: y_min + 2*(y_max-y_min)/3 = -256 + 2*512/3 = -256+341 = 85
        if (I_t > y_min && I_t < y_max) begin
            $display("[PASS] D: two-bits -> I_t=%0d in (%0d, %0d)", I_t, y_min, y_max);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] D: two-bits -> I_t=%0d not in (%0d, %0d)", I_t, y_min, y_max);
            fail_count = fail_count + 1;
        end

        $display("---------------------------------------");
        $display("tb_sc_dotproduct_to_current: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("[PASS] All tests passed");
        else
            $display("[FAIL] %0d test(s) failed", fail_count);
        $finish;
    end

endmodule
