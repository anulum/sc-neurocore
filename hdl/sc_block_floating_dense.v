// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Block-floating dense layer reference RTL

`timescale 1ns / 1ps

module sc_block_floating_dense #(
    parameter integer N_INPUTS = 64,
    parameter integer N_OUTPUTS = 32,
    parameter integer MANTISSA_WIDTH = 16,
    parameter integer EXPONENT_WIDTH = 3,
    parameter integer BLOCK_SIZE = 32,
    parameter integer INPUT_WIDTH = 32,
    parameter integer ACCUM_WIDTH = 32,
    parameter integer EXPONENT_BIAS = ((1 << (EXPONENT_WIDTH - 1)) - 1)
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire signed [N_OUTPUTS*N_INPUTS*MANTISSA_WIDTH-1:0] mantissas_bfp,
    input wire [((N_OUTPUTS*N_INPUTS + BLOCK_SIZE - 1)/BLOCK_SIZE)*EXPONENT_WIDTH-1:0] exponents_bfp,
    input wire signed [N_INPUTS*INPUT_WIDTH-1:0] inputs_q1616,
    output reg valid_out,
    output reg signed [N_OUTPUTS*ACCUM_WIDTH-1:0] outputs_q1616,
    output reg overflow
);

localparam integer PRODUCT_WIDTH = MANTISSA_WIDTH + INPUT_WIDTH;
localparam integer NUM_WEIGHTS = N_INPUTS * N_OUTPUTS;
localparam integer NUM_BLOCKS = (NUM_WEIGHTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
localparam integer MAX_SHIFT = (1 << EXPONENT_WIDTH) - 1;
localparam integer GUARD_WIDTH = ((N_INPUTS < 2) ? 1 : $clog2(N_INPUTS)) + MAX_SHIFT + 1;
localparam integer SUM_WIDTH = PRODUCT_WIDTH + GUARD_WIDTH + 1;

localparam signed [ACCUM_WIDTH-1:0] ACCUM_MAX = {1'b0, {ACCUM_WIDTH-1{1'b1}}};
localparam signed [ACCUM_WIDTH-1:0] ACCUM_MIN = {1'b1, {ACCUM_WIDTH-1{1'b0}}};
localparam signed [SUM_WIDTH-1:0] ACCUM_MAX_EXT =
    {{(SUM_WIDTH-ACCUM_WIDTH){1'b0}}, ACCUM_MAX};
localparam signed [SUM_WIDTH-1:0] ACCUM_MIN_EXT =
    {{(SUM_WIDTH-ACCUM_WIDTH){1'b1}}, ACCUM_MIN};

integer output_idx;
integer input_idx;
integer linear_idx;
integer block_idx;
integer mantissa_offset;
integer exponent_offset;
integer input_offset;
integer unbiased_shift;

reg signed [MANTISSA_WIDTH-1:0] mantissa_lane;
reg [EXPONENT_WIDTH-1:0] exponent_lane;
reg signed [INPUT_WIDTH-1:0] input_lane;
reg signed [PRODUCT_WIDTH-1:0] product;
reg signed [SUM_WIDTH-1:0] shifted_product;
reg signed [SUM_WIDTH-1:0] sum;
reg signed [N_OUTPUTS*ACCUM_WIDTH-1:0] outputs_next;
reg overflow_next;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out <= 1'b0;
        outputs_q1616 <= {N_OUTPUTS*ACCUM_WIDTH{1'b0}};
        overflow <= 1'b0;
    end else begin
        outputs_next = {N_OUTPUTS*ACCUM_WIDTH{1'b0}};
        overflow_next = 1'b0;

        if (valid_in) begin
            for (output_idx = 0; output_idx < N_OUTPUTS; output_idx = output_idx + 1) begin
                sum = {SUM_WIDTH{1'b0}};
                for (input_idx = 0; input_idx < N_INPUTS; input_idx = input_idx + 1) begin
                    linear_idx = output_idx*N_INPUTS + input_idx;
                    block_idx = linear_idx / BLOCK_SIZE;
                    mantissa_offset = linear_idx * MANTISSA_WIDTH;
                    exponent_offset = block_idx * EXPONENT_WIDTH;
                    input_offset = input_idx * INPUT_WIDTH;

                    mantissa_lane = mantissas_bfp[mantissa_offset +: MANTISSA_WIDTH];
                    exponent_lane = exponents_bfp[exponent_offset +: EXPONENT_WIDTH];
                    input_lane = inputs_q1616[input_offset +: INPUT_WIDTH];
                    product = mantissa_lane * input_lane;

                    unbiased_shift = exponent_lane - EXPONENT_BIAS;
                    if (unbiased_shift >= 0) begin
                        shifted_product = product <<< unbiased_shift;
                    end else begin
                        shifted_product = product >>> (-unbiased_shift);
                    end
                    sum = sum + shifted_product;
                end

                if (sum > ACCUM_MAX_EXT) begin
                    outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = ACCUM_MAX;
                    overflow_next = 1'b1;
                end else if (sum < ACCUM_MIN_EXT) begin
                    outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = ACCUM_MIN;
                    overflow_next = 1'b1;
                end else begin
                    outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = sum[ACCUM_WIDTH-1:0];
                end
            end
        end

        valid_out <= valid_in;
        outputs_q1616 <= outputs_next;
        overflow <= overflow_next;
    end
end

endmodule
