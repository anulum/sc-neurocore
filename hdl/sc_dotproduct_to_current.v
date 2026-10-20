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
//  - y_min, y_max, I_t are DATA_WIDTH-wide signed fixed-point words.
//  - N_INPUTS is positive. The binary point is unchanged by this module.
//  - Both endpoint orders and the complete signed word range are supported.
//  - We compute:
//      range = y_max - y_min
//      product = range * count
//      scaled = product / N_INPUTS (signed division, truncated towards zero)
//      I_t = y_min + scaled
//
// All arithmetic is combinational here.

`timescale 1ns / 1ps

module sc_dotproduct_to_current #(
    parameter integer N_INPUTS = 3,
    parameter integer DATA_WIDTH = 16
)(
    input wire [(N_INPUTS > 0 ? N_INPUTS : 1)-1:0] post_bits,
    input wire signed [DATA_WIDTH-1:0] y_min,
    input wire signed [DATA_WIDTH-1:0] y_max,
    output reg signed [DATA_WIDTH-1:0] I_t
);

generate
if (N_INPUTS > 0) begin : VALID_INPUT_COUNT

// Width needed to count up to N_INPUTS
localparam integer CNT_WIDTH = $clog2(N_INPUTS + 1);

integer i;

reg [CNT_WIDTH-1:0]                     count_ones;
// A difference of two signed words needs an additional bit. Keep the count
// explicitly positive and retain the complete product through the division.
localparam integer PRODUCT_WIDTH = DATA_WIDTH + CNT_WIDTH + 2;
localparam signed [PRODUCT_WIDTH-1:0] INPUT_COUNT =
    {{(DATA_WIDTH+1){1'b0}}, N_INPUTS[CNT_WIDTH:0]};
reg signed [DATA_WIDTH:0]              range;
reg signed [PRODUCT_WIDTH-1:0]         product;
reg signed [PRODUCT_WIDTH-1:0]         scaled_with_unused_carry;

always @* begin
    // 1) Count ones in post_bits (population count)
    count_ones = {CNT_WIDTH{1'b0}};
    for (i = 0; i < N_INPUTS; i = i + 1) begin
        if (post_bits[i]) begin
            count_ones = count_ones + 1'b1;
        end
    end

    // 2) Compute range = y_max - y_min
    range = $signed({y_max[DATA_WIDTH-1], y_max})
          - $signed({y_min[DATA_WIDTH-1], y_min});

    // 3) Multiply by count_ones
    product = range * $signed({1'b0, count_ones});

    // 4) Divide product by N_INPUTS (constant divisor)
    // This is synthesizable as a constant divider; tools will implement it
    // using shifts/adders or a small multiplier by reciprocal.
    scaled_with_unused_carry = product / INPUT_COUNT;

    // 5) Map back into [y_min, y_max]
    // Only the final word is returned. Dropping the scaled carry before this
    // modular addition is exact because the interpolated result fits the word.
    I_t = y_min + scaled_with_unused_carry[DATA_WIDTH-1:0];
end

end else begin : INVALID_INPUT_COUNT
    initial $fatal(1, "sc_dotproduct_to_current: N_INPUTS must be positive");
    always @* I_t = y_min;
end
endgenerate

endmodule
