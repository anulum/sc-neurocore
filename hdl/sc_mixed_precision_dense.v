// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Mixed-precision dense layer reference RTL

`timescale 1ns / 1ps

module sc_mixed_precision_dense #(
    parameter integer N_INPUTS = 64,
    parameter integer N_OUTPUTS = 32,
    parameter integer WEIGHT_WIDTH = 16,
    parameter integer INPUT_WIDTH = 32,
    parameter integer ACCUM_WIDTH = 32,
    parameter integer WEIGHT_FRAC = 8
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire signed [N_OUTPUTS*N_INPUTS*WEIGHT_WIDTH-1:0] weights_q88,
    input wire signed [N_INPUTS*INPUT_WIDTH-1:0] inputs_q1616,
    output reg valid_out,
    output reg signed [N_OUTPUTS*ACCUM_WIDTH-1:0] outputs_q1616,
    output reg [N_OUTPUTS-1:0] overflow_vector,
    output reg overflow
);

localparam integer PRODUCT_WIDTH = WEIGHT_WIDTH + INPUT_WIDTH;
localparam integer GUARD_WIDTH = (N_INPUTS < 2) ? 1 : $clog2(N_INPUTS);
localparam integer SUM_WIDTH = PRODUCT_WIDTH + GUARD_WIDTH + 1;

localparam signed [ACCUM_WIDTH-1:0] ACCUM_MAX = {1'b0, {ACCUM_WIDTH-1{1'b1}}};
localparam signed [ACCUM_WIDTH-1:0] ACCUM_MIN = {1'b1, {ACCUM_WIDTH-1{1'b0}}};
localparam signed [SUM_WIDTH-1:0] ACCUM_MAX_EXT =
    {{(SUM_WIDTH-ACCUM_WIDTH){1'b0}}, ACCUM_MAX};
localparam signed [SUM_WIDTH-1:0] ACCUM_MIN_EXT =
    {{(SUM_WIDTH-ACCUM_WIDTH){1'b1}}, ACCUM_MIN};

integer output_idx;
integer input_idx;
integer weight_offset;
integer input_offset;

reg signed [WEIGHT_WIDTH-1:0] weight_lane;
reg signed [INPUT_WIDTH-1:0] input_lane;
reg signed [PRODUCT_WIDTH-1:0] product;
reg signed [SUM_WIDTH-1:0] sum;
reg signed [SUM_WIDTH-1:0] scaled_sum;
reg signed [N_OUTPUTS*ACCUM_WIDTH-1:0] outputs_next;
reg [N_OUTPUTS-1:0] overflow_vector_next;
reg overflow_next;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out <= 1'b0;
        outputs_q1616 <= {N_OUTPUTS*ACCUM_WIDTH{1'b0}};
        overflow_vector <= {N_OUTPUTS{1'b0}};
        overflow <= 1'b0;
    end else begin
        outputs_next = {N_OUTPUTS*ACCUM_WIDTH{1'b0}};
        overflow_vector_next = {N_OUTPUTS{1'b0}};
        overflow_next = 1'b0;

        if (valid_in) begin
            for (output_idx = 0; output_idx < N_OUTPUTS; output_idx = output_idx + 1) begin
                sum = {SUM_WIDTH{1'b0}};
                for (input_idx = 0; input_idx < N_INPUTS; input_idx = input_idx + 1) begin
                    weight_offset = (output_idx*N_INPUTS + input_idx) * WEIGHT_WIDTH;
                    input_offset = input_idx * INPUT_WIDTH;
                    weight_lane = weights_q88[weight_offset +: WEIGHT_WIDTH];
                    input_lane = inputs_q1616[input_offset +: INPUT_WIDTH];
                    product = weight_lane * input_lane;
                    sum = sum + product;
                end

                scaled_sum = sum >>> WEIGHT_FRAC;
                if (scaled_sum > ACCUM_MAX_EXT) begin
                    outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = ACCUM_MAX;
                    overflow_vector_next[output_idx] = 1'b1;
                    overflow_next = 1'b1;
                end else if (scaled_sum < ACCUM_MIN_EXT) begin
                    outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = ACCUM_MIN;
                    overflow_vector_next[output_idx] = 1'b1;
                    overflow_next = 1'b1;
                end else begin
                    outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = scaled_sum[ACCUM_WIDTH-1:0];
                end
            end
        end

        valid_out <= valid_in;
        outputs_q1616 <= outputs_next;
        overflow_vector <= overflow_vector_next;
        overflow <= overflow_next;
    end
end

endmodule
