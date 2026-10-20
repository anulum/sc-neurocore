// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal Verification for SC Dotproduct To Current

`default_nettype none

module sc_dotproduct_to_current_formal (
    input wire [2:0]              post_bits,
    input wire signed [15:0]      y_min,
    input wire signed [15:0]      y_max
);

    wire signed [15:0] I_t;

    sc_dotproduct_to_current #(
        .N_INPUTS(3),
        .DATA_WIDTH(16)
    ) uut (
        .post_bits(post_bits),
        .y_min(y_min),
        .y_max(y_max),
        .I_t(I_t)
    );

`ifdef FORMAL
    // Independent widened integer specification: no endpoint-order assumption.
    wire signed [18:0] low = {{3{y_min[15]}}, y_min};
    wire signed [18:0] high = {{3{y_max[15]}}, y_max};
    wire signed [18:0] population = {18'd0, post_bits[0]}
                                   + {18'd0, post_bits[1]}
                                   + {18'd0, post_bits[2]};
    wire signed [18:0] numerator = (high - low) * population;
    wire signed [18:0] expected = low + numerator / 19'sd3;

    always @* assert($signed(I_t) == expected);

    // 1. All post_bits zero => I_t == y_min
    always @* begin
        if (post_bits == 3'b000)
            assert(I_t == y_min);
    end

    // 2. All post_bits one => I_t == y_max
    always @* begin
        if (post_bits == 3'b111)
            assert(I_t == y_max);
    end

    // 3. Interpolation stays between either ordering of the signed endpoints.
    always @* begin
        if (y_min <= y_max) begin
            assert(I_t >= y_min);
            assert(I_t <= y_max);
        end else begin
            assert(I_t <= y_min);
            assert(I_t >= y_max);
        end
    end

    // 4. Cover: mid-range output reachable
    always @* begin
        cover(I_t > y_min && I_t < y_max);
    end
`endif

endmodule
