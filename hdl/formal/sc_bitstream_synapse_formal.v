// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal Verification for SC Bitstream Synapse

`default_nettype none

module sc_bitstream_synapse_formal (
    input wire pre_bit,
    input wire w_bit
);

    wire post_bit;

    sc_bitstream_synapse uut (
        .pre_bit(pre_bit),
        .w_bit(w_bit),
        .post_bit(post_bit)
    );

`ifdef FORMAL
    // 1. Combinational correctness: post = pre AND weight
    always @(*) begin
        assert(post_bit == (pre_bit & w_bit));
    end

    // 2. If either input is 0, output is 0
    always @(*) begin
        if (!pre_bit)
            assert(post_bit == 1'b0);
        if (!w_bit)
            assert(post_bit == 1'b0);
    end

    // 3. Both inputs high => output high
    always @(*) begin
        if (pre_bit && w_bit)
            assert(post_bit == 1'b1);
    end

    // 4. Cover all input combinations
    always @(*) begin
        cover(pre_bit == 1'b0 && w_bit == 1'b0);
        cover(pre_bit == 1'b0 && w_bit == 1'b1);
        cover(pre_bit == 1'b1 && w_bit == 1'b0);
        cover(pre_bit == 1'b1 && w_bit == 1'b1);
    end
`endif

endmodule
