// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — Formal properties for sc_tmr_voter
//
// Prove:
// P1. Majority correctness: output is bitwise majority of inputs
// P2. Error detection: error flag asserted iff any input disagrees with majority
// P3. All-agree: when a==b==c, voted==a and error==0

module tmr_voter_formal #(
    parameter DATA_WIDTH = 16
)(
    input wire [DATA_WIDTH-1:0] a,
    input wire [DATA_WIDTH-1:0] b,
    input wire [DATA_WIDTH-1:0] c
);

    wire [DATA_WIDTH-1:0] voted;
    wire error;

    sc_tmr_voter #(.DATA_WIDTH(DATA_WIDTH)) uut (
        .a(a), .b(b), .c(c),
        .voted(voted), .error(error)
    );

    // P1: Majority correctness (bit-wise)
    // For each bit i: voted[i] == majority(a[i], b[i], c[i])
    genvar i;
    generate
        for (i = 0; i < DATA_WIDTH; i = i + 1) begin : gen_p1
            wire maj = (a[i] & b[i]) | (b[i] & c[i]) | (a[i] & c[i]);

            always @(*) begin
                assert(voted[i] == maj);
            end
        end
    endgenerate

    // P2: Error flag is raised iff any replica disagrees with majority
    wire any_disagree = |(a ^ voted) | |(b ^ voted) | |(c ^ voted);
    always @(*) begin
        assert(error == any_disagree);
    end

    // P3: Unanimous agreement → no error
    always @(*) begin
        if (a == b && b == c) begin
            assert(voted == a);
            assert(error == 1'b0);
        end
    end

endmodule
