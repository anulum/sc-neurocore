// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - DCLS Q8.8 tent kernel

`timescale 1ns / 1ps

module sc_dcls_tent_kernel #(
    parameter integer N_TAPS = 3,
    parameter integer DATA_WIDTH = 16,
    parameter integer FRACTION = 8,
    parameter integer ACC_WIDTH = 32
)(
    input wire [N_TAPS-1:0] tap_spikes,
    input wire signed [N_TAPS*DATA_WIDTH-1:0] tap_weights_q88,
    input wire signed [DATA_WIDTH-1:0] centre_q88,
    input wire signed [DATA_WIDTH-1:0] sigma_q88,
    output reg signed [DATA_WIDTH-1:0] weighted_sum_q88,
    output reg signed [ACC_WIDTH-1:0] accumulator_q16_16,
    output reg overflow,
    output reg invalid_sigma
);
    localparam signed [63:0] Q88_ONE = 64'sd1 <<< FRACTION;
    localparam signed [63:0] I32_MAX_VALUE = 64'sd2147483647;
    localparam signed [63:0] I32_MIN_VALUE = -64'sd2147483648;
    localparam signed [63:0] I16_MAX_Q16_16 = 64'sd32767 <<< FRACTION;
    localparam signed [63:0] I16_MIN_Q16_16 = -64'sd32768 <<< FRACTION;

    integer tap_idx;
    reg signed [DATA_WIDTH-1:0] weight_q88;
    reg signed [63:0] centre_ext;
    reg signed [63:0] sigma_ext;
    reg signed [63:0] delay_q88;
    reg signed [63:0] distance_q88;
    reg signed [63:0] numerator_q88;
    reg signed [63:0] gate_q88;
    reg signed [63:0] contribution_q16_16;
    reg signed [63:0] accumulator_wide;

    always @* begin
        weighted_sum_q88 = {DATA_WIDTH{1'b0}};
        accumulator_q16_16 = {ACC_WIDTH{1'b0}};
        overflow = 1'b0;
        invalid_sigma = sigma_q88 <= 0;
        accumulator_wide = 64'sd0;
        centre_ext = {{(64-DATA_WIDTH){centre_q88[DATA_WIDTH-1]}}, centre_q88};
        sigma_ext = {{(64-DATA_WIDTH){sigma_q88[DATA_WIDTH-1]}}, sigma_q88};
        // Default the per-tap scratch registers so every control path assigns them: on the
        // invalid-sigma path and the non-spiking tap path they are otherwise unwritten, which
        // infers a combinational latch. Each is always reassigned before it is read on the
        // paths that use it, so these defaults never reach an output — they only make the
        // block fully combinational (no latch) and behaviourally identical.
        weight_q88 = {DATA_WIDTH{1'b0}};
        delay_q88 = 64'sd0;
        distance_q88 = 64'sd0;
        numerator_q88 = 64'sd0;
        gate_q88 = 64'sd0;
        contribution_q16_16 = 64'sd0;

        if (!invalid_sigma) begin
            for (tap_idx = 0; tap_idx < N_TAPS; tap_idx = tap_idx + 1) begin
                weight_q88 = tap_weights_q88[tap_idx*DATA_WIDTH +: DATA_WIDTH];
                // Widen the 32-bit loop index to the 64-bit accumulator width before shifting
                // so the shift is evaluated at full width (tap_idx is always non-negative).
                delay_q88 = $signed({{32{1'b0}}, tap_idx}) <<< FRACTION;
                if (delay_q88 >= centre_ext) begin
                    distance_q88 = delay_q88 - centre_ext;
                end else begin
                    distance_q88 = centre_ext - delay_q88;
                end

                if (tap_spikes[tap_idx] && distance_q88 < sigma_ext) begin
                    numerator_q88 = sigma_ext - distance_q88;
                    gate_q88 = (numerator_q88 <<< FRACTION) / sigma_ext;
                    if (gate_q88 > Q88_ONE) begin
                        gate_q88 = Q88_ONE;
                    end
                    contribution_q16_16 = weight_q88 * gate_q88;
                    accumulator_wide = accumulator_wide + contribution_q16_16;
                end
            end

            if (accumulator_wide > I32_MAX_VALUE) begin
                accumulator_q16_16 = {1'b0, {(ACC_WIDTH-1){1'b1}}};
                overflow = 1'b1;
            end else if (accumulator_wide < I32_MIN_VALUE) begin
                accumulator_q16_16 = {1'b1, {(ACC_WIDTH-1){1'b0}}};
                overflow = 1'b1;
            end else begin
                accumulator_q16_16 = accumulator_wide[ACC_WIDTH-1:0];
            end

            if (accumulator_wide > I16_MAX_Q16_16) begin
                weighted_sum_q88 = {1'b0, {(DATA_WIDTH-1){1'b1}}};
                overflow = 1'b1;
            end else if (accumulator_wide < I16_MIN_Q16_16) begin
                weighted_sum_q88 = {1'b1, {(DATA_WIDTH-1){1'b0}}};
                overflow = 1'b1;
            end else begin
                // This branch is only reached when accumulator_wide is within the Q16.16 range
                // [I16_MIN_Q16_16, I16_MAX_Q16_16], so the arithmetic right shift fits the Q8.8
                // output exactly; the narrowing assignment is therefore intentional and lossless.
                /* verilator lint_off WIDTHTRUNC */
                weighted_sum_q88 = accumulator_wide >>> FRACTION;
                /* verilator lint_on WIDTHTRUNC */
            end
        end
    end
endmodule
