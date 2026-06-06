// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - DCLS layer core

`timescale 1ns / 1ps

module sc_dcls_layer_core #(
    parameter integer N_TAPS = 3,
    parameter integer DATA_WIDTH = 16,
    parameter integer FRACTION = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer DELAY_DEPTH = 31,
    parameter integer PTR_WIDTH = 5
)(
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    input wire spike_in,
    input wire [N_TAPS*PTR_WIDTH-1:0] tap_offsets,
    input wire signed [N_TAPS*DATA_WIDTH-1:0] tap_weights_q88,
    input wire signed [DATA_WIDTH-1:0] centre_q88,
    input wire signed [DATA_WIDTH-1:0] sigma_q88,
    output reg out_valid,
    output wire signed [DATA_WIDTH-1:0] weighted_sum_q88,
    output wire signed [ACC_WIDTH-1:0] accumulator_q16_16,
    output wire overflow,
    output wire invalid_sigma
);
    wire [N_TAPS-1:0] delayed_spikes;

    genvar tap_idx;
    generate
        for (tap_idx = 0; tap_idx < N_TAPS; tap_idx = tap_idx + 1) begin : gen_dcls_delay
            sc_dcls_axonal_delay #(
                .DEPTH(DELAY_DEPTH),
                .PTR_WIDTH(PTR_WIDTH)
            ) delay_line (
                .clk(clk),
                .rst_n(rst_n),
                .spike_in(spike_in),
                .read_offset(tap_offsets[tap_idx*PTR_WIDTH +: PTR_WIDTH]),
                .spike_out(delayed_spikes[tap_idx])
            );
        end
    endgenerate

    sc_dcls_tent_kernel #(
        .N_TAPS(N_TAPS),
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTION(FRACTION),
        .ACC_WIDTH(ACC_WIDTH)
    ) tent_kernel (
        .tap_spikes(delayed_spikes),
        .tap_weights_q88(tap_weights_q88),
        .centre_q88(centre_q88),
        .sigma_q88(sigma_q88),
        .weighted_sum_q88(weighted_sum_q88),
        .accumulator_q16_16(accumulator_q16_16),
        .overflow(overflow),
        .invalid_sigma(invalid_sigma)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
        end else begin
            out_valid <= in_valid;
        end
    end
endmodule
