// SPDX-License-Identifier: AGPL-3.0-or-later
// hdl/sc_bitstream_synapse.v
//
// Stochastic synapse for unipolar SC:
//  - pre_bit: input bitstream (prob p)
//  - w_bit: weight bitstream (prob w)
//  - post_bit ≈ Bernoulli(p * w) via AND.

`timescale 1ns / 1ps

module sc_bitstream_synapse (
    input wire pre_bit,
    input wire w_bit,
    output wire post_bit
);

assign post_bit = pre_bit & w_bit;

endmodule
