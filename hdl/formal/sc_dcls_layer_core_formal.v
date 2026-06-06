// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - DCLS layer core formal harness

`timescale 1ns / 1ps
`default_nettype none

module sc_dcls_layer_core_formal (
    input wire clk
);
    reg rst_n = 1'b0;
    reg in_valid = 1'b0;
    reg spike_in = 1'b0;
    reg past_valid = 1'b0;

    wire out_valid;
    wire signed [15:0] weighted_sum_q88;
    wire signed [31:0] accumulator_q16_16;
    wire overflow;
    wire invalid_sigma;

    always @(posedge clk) begin
        past_valid <= 1'b1;
        rst_n <= 1'b1;
        if (rst_n) begin
            in_valid <= 1'b1;
            spike_in <= 1'b1;
        end
    end

    sc_dcls_layer_core #(
        .N_TAPS(3),
        .DATA_WIDTH(16),
        .FRACTION(8),
        .ACC_WIDTH(32),
        .DELAY_DEPTH(4),
        .PTR_WIDTH(2)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .spike_in(spike_in),
        .tap_offsets({2'd2, 2'd1, 2'd0}),
        .tap_weights_q88({16'sd64, 16'sd128, 16'sd256}),
        .centre_q88(16'sd256),
        .sigma_q88(16'sd512),
        .out_valid(out_valid),
        .weighted_sum_q88(weighted_sum_q88),
        .accumulator_q16_16(accumulator_q16_16),
        .overflow(overflow),
        .invalid_sigma(invalid_sigma)
    );

    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            assert(!invalid_sigma);
            assert(!overflow);
            assert(accumulator_q16_16 >= 0);
            assert(weighted_sum_q88 >= 0);
            if ($past(in_valid)) begin
                assert(out_valid);
            end
            cover(out_valid && weighted_sum_q88 > 0);
        end
    end
endmodule

`default_nettype wire
