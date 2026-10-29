// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parameterized signed fixed-point DCLS tent kernel

// Port names retain their Q8.8/Q16.16 compatibility spelling. FRACTION defines
// the actual weight/output binary point; accumulator words have 2*FRACTION bits.
// DATA_WIDTH is 2..64 and ACC_WIDTH is 1..64; FRACTION is smaller than DATA_WIDTH.
// N_TAPS is positive. Widened arithmetic retains the complete tent numerator,
// weight products and tap sum before saturating either exposed result.

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
    localparam integer REQUIRED_WIDTH = DATA_WIDTH + FRACTION + $clog2(N_TAPS) + 4;
    localparam integer WIDE_WIDTH = REQUIRED_WIDTH > 64 ? REQUIRED_WIDTH : 64;
    localparam signed [WIDE_WIDTH-1:0] WIDE_ONE = {{(WIDE_WIDTH-1){1'b0}}, 1'b1};
    localparam signed [WIDE_WIDTH-1:0] Q_ONE = WIDE_ONE <<< FRACTION;
    localparam signed [WIDE_WIDTH-1:0] ACC_MAX_VALUE =
        (WIDE_ONE <<< (ACC_WIDTH-1)) - WIDE_ONE;
    localparam signed [WIDE_WIDTH-1:0] ACC_MIN_VALUE =
        -(WIDE_ONE <<< (ACC_WIDTH-1));
    localparam signed [WIDE_WIDTH-1:0] WORD_MAX_ACCUMULATOR =
        ((WIDE_ONE <<< (DATA_WIDTH-1)) - WIDE_ONE) <<< FRACTION;
    localparam signed [WIDE_WIDTH-1:0] WORD_MIN_ACCUMULATOR =
        (-(WIDE_ONE <<< (DATA_WIDTH-1))) <<< FRACTION;

    initial begin
        if (N_TAPS < 1 || DATA_WIDTH < 2 || DATA_WIDTH > 64
            || ACC_WIDTH < 1 || ACC_WIDTH > 64 || FRACTION < 0 || FRACTION >= DATA_WIDTH)
            $fatal(1, "DCLS requires positive taps, 2..64-bit data, 1..64-bit accumulator and FRACTION < DATA_WIDTH");
    end

    integer tap_idx;
    reg signed [DATA_WIDTH-1:0] weight_q88;
    reg signed [WIDE_WIDTH-1:0] centre_ext;
    reg signed [WIDE_WIDTH-1:0] sigma_ext;
    reg signed [WIDE_WIDTH-1:0] delay_q88;
    reg signed [WIDE_WIDTH-1:0] distance_q88;
    reg signed [WIDE_WIDTH-1:0] numerator_q88;
    reg signed [WIDE_WIDTH-1:0] gate_q88;
    reg signed [WIDE_WIDTH-1:0] contribution_q16_16;
    reg signed [WIDE_WIDTH-1:0] accumulator_wide;

    always @* begin
        weighted_sum_q88 = {DATA_WIDTH{1'b0}};
        accumulator_q16_16 = {ACC_WIDTH{1'b0}};
        overflow = 1'b0;
        invalid_sigma = sigma_q88 <= 0;
        accumulator_wide = {WIDE_WIDTH{1'b0}};
        centre_ext = {{(WIDE_WIDTH-DATA_WIDTH){centre_q88[DATA_WIDTH-1]}}, centre_q88};
        sigma_ext = {{(WIDE_WIDTH-DATA_WIDTH){sigma_q88[DATA_WIDTH-1]}}, sigma_q88};
        // Default the per-tap scratch registers so every control path assigns them: on the
        // invalid-sigma path and the non-spiking tap path they are otherwise unwritten, which
        // infers a combinational latch. Each is always reassigned before it is read on the
        // paths that use it, so these defaults never reach an output — they only make the
        // block fully combinational (no latch) and behaviourally identical.
        weight_q88 = {DATA_WIDTH{1'b0}};
        delay_q88 = {WIDE_WIDTH{1'b0}};
        distance_q88 = {WIDE_WIDTH{1'b0}};
        numerator_q88 = {WIDE_WIDTH{1'b0}};
        gate_q88 = {WIDE_WIDTH{1'b0}};
        contribution_q16_16 = {WIDE_WIDTH{1'b0}};

        if (!invalid_sigma) begin
            for (tap_idx = 0; tap_idx < N_TAPS; tap_idx = tap_idx + 1) begin
                weight_q88 = tap_weights_q88[tap_idx*DATA_WIDTH +: DATA_WIDTH];
                // Widen the 32-bit loop index to the full accumulator width before shifting
                // so the shift is evaluated at full width (tap_idx is always non-negative).
                delay_q88 = $signed({{(WIDE_WIDTH-32){1'b0}}, tap_idx}) <<< FRACTION;
                if (delay_q88 >= centre_ext) begin
                    distance_q88 = delay_q88 - centre_ext;
                end else begin
                    distance_q88 = centre_ext - delay_q88;
                end

                if (tap_spikes[tap_idx] && distance_q88 < sigma_ext) begin
                    numerator_q88 = sigma_ext - distance_q88;
                    gate_q88 = (numerator_q88 <<< FRACTION) / sigma_ext;
                    if (gate_q88 > Q_ONE) begin
                        gate_q88 = Q_ONE;
                    end
                    contribution_q16_16 = weight_q88 * gate_q88;
                    accumulator_wide = accumulator_wide + contribution_q16_16;
                end
            end

            if (accumulator_wide > ACC_MAX_VALUE) begin
                accumulator_q16_16 = {1'b0, {(ACC_WIDTH-1){1'b1}}};
                overflow = 1'b1;
            end else if (accumulator_wide < ACC_MIN_VALUE) begin
                accumulator_q16_16 = {1'b1, {(ACC_WIDTH-1){1'b0}}};
                overflow = 1'b1;
            end else begin
                accumulator_q16_16 = accumulator_wide[ACC_WIDTH-1:0];
            end

            if (accumulator_wide > WORD_MAX_ACCUMULATOR) begin
                weighted_sum_q88 = {1'b0, {(DATA_WIDTH-1){1'b1}}};
                overflow = 1'b1;
            end else if (accumulator_wide < WORD_MIN_ACCUMULATOR) begin
                weighted_sum_q88 = {1'b1, {(DATA_WIDTH-1){1'b0}}};
                overflow = 1'b1;
            end else begin
                // Saturation has proved the shifted word fits the declared signed output.
                weighted_sum_q88 = accumulator_wide[FRACTION +: DATA_WIDTH];
            end
        end
    end
endmodule
