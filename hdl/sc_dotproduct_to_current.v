// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dot-product-like reduction of SC post-synaptic bits into

// hdl/sc_dotproduct_to_current.v
//
// Dot-product-like reduction of SC post-synaptic bits into a fixed-point current.
//
// Concept:
//  - Input: N_INPUTS post_bits (0/1)
//  - Count ones: count = sum(post_bits)
//  - Compute prob ~= count / N_INPUTS (unipolar probability)
//  - Map to [y_min, y_max]:
//      I_t = y_min + prob * (y_max - y_min)
//
// Fixed-point notes:
//  - y_min, y_max, I_t are DATA_WIDTH-wide signed fixed-point values (Q8.8).
//  - We compute:
//      range = y_max - y_min
//      product = range * count
//      scaled = product / N_INPUTS
//      I_t = y_min + scaled
//
// All arithmetic is combinational here.

`timescale 1ns / 1ps

module sc_dotproduct_to_current #(
    parameter integer N_INPUTS = 3,
    parameter integer DATA_WIDTH = 16
)(
    input wire [N_INPUTS-1:0]          post_bits,
    input wire signed [DATA_WIDTH-1:0] y_min,
    input wire signed [DATA_WIDTH-1:0] y_max,
    output reg signed [DATA_WIDTH-1:0] I_t
);

// Width needed to count up to N_INPUTS
localparam integer CNT_WIDTH = $clog2(N_INPUTS + 1);

integer i;

reg [CNT_WIDTH-1:0]                     count_ones;
reg signed [DATA_WIDTH-1:0]             range;
reg signed [DATA_WIDTH + CNT_WIDTH-1:0] product;
reg signed [DATA_WIDTH-1:0]             scaled;

always @* begin
    // 1) Count ones in post_bits (population count)
    count_ones = {CNT_WIDTH{1'b0}};
    for (i = 0; i < N_INPUTS; i = i + 1) begin
        if (post_bits[i]) begin
            count_ones = count_ones + 1'b1;
        end
    end

    // 2) Compute range = y_max - y_min
    range = y_max - y_min;

    // 3) Multiply by count_ones
    // product width: DATA_WIDTH + CNT_WIDTH bits to avoid overflow
    product = range * count_ones;

    // 4) Divide product by N_INPUTS (constant divisor)
    // This is synthesizable as a constant divider; tools will implement it
    // using shifts/adders or a small multiplier by reciprocal.
    if (N_INPUTS > 0) begin
        scaled = product / N_INPUTS;
    end else begin
        scaled = {DATA_WIDTH{1'b0}};
    end

    // 5) Map back into [y_min, y_max]
    I_t = y_min + scaled;
end

endmodule
