// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic bitstream encoder:

// hdl/sc_bitstream_encoder.v
//
// Stochastic bitstream encoder:
//  - Input: fixed-point x_value in [0, 1] represented on DATA_WIDTH bits
//  - Output: bit_out (0/1) such that P(bit_out = 1) ~ x_value / (2^DATA_WIDTH - 1)
//
// Internally uses a simple LFSR-based PRNG and comparator.
//
// This is a practical hardware analogue of the Python BitstreamEncoder
// (probability -> Bernoulli bitstream).

`timescale 1ns / 1ps

module sc_bitstream_encoder #(
    parameter integer DATA_WIDTH = 16,
    // Width of internal LFSR for randomness. Must be <= DATA_WIDTH
    parameter integer LFSR_WIDTH = 16,
    // Per-instance seed to decorrelate parallel encoders.
    // Each encoder instance MUST receive a unique non-zero value;
    // otherwise all encoders sharing the same seed will produce
    // identical bitstreams for equal x_value inputs.
    parameter [LFSR_WIDTH-1:0] SEED_INIT = 16'hACE1
)(
    input wire                      clk,
    input wire                      rst_n,

    // Fixed-point input value representing probability.
    // Range interpretation:
    //  x_value = 0                   -> P(bit_out=1) ~ 0.0
    //  x_value = 2^DATA_WIDTH-1      -> P(bit_out=1) ~ 1.0
    input wire [DATA_WIDTH-1:0]   x_value,

    // Optional time index (for consistency with Python API).
    // Not required for core operation; can be used for reseeding or
    // decorrelation if desired.
    input wire [31:0]               t_index,

    output reg                      bit_out
);

// ----------------------------------------------------------------
// LFSR for pseudo-random numbers
// ----------------------------------------------------------------
// Simple maximal-length LFSR of width LFSR_WIDTH.
// For LFSR_WIDTH = 16, use taps corresponding to polynomial x^16 + x^14 + x^13 + x^11 + 1
// (common maximal-length polynomial).
// Tools will infer a small register + XOR network.
reg [LFSR_WIDTH-1:0] lfsr_reg;
wire                 feedback;

assign feedback = lfsr_reg[LFSR_WIDTH-1] ^
                  lfsr_reg[LFSR_WIDTH-3] ^
                  lfsr_reg[LFSR_WIDTH-4] ^
                  lfsr_reg[LFSR_WIDTH-6];

// Extend LFSR output to DATA_WIDTH by zero-extension or truncation.
// If DATA_WIDTH > LFSR_WIDTH, upper bits will be zero.
wire [DATA_WIDTH-1:0] rnd_value;
assign rnd_value = {{(DATA_WIDTH-LFSR_WIDTH){1'b0}}, lfsr_reg};


// ----------------------------------------------------------------
// Sequential logic
// ----------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Use per-instance SEED_INIT to ensure each encoder starts from
        // a distinct LFSR state.  XOR with t_index allows additional
        // run-to-run variation when t_index is driven to a non-zero value
        // before reset de-assertion.
        lfsr_reg <= (SEED_INIT ^ t_index[LFSR_WIDTH-1:0]) != {LFSR_WIDTH{1'b0}}
                  ? (SEED_INIT ^ t_index[LFSR_WIDTH-1:0])
                  : SEED_INIT;
        bit_out <= 1'b0;
    end else begin
        // Advance LFSR
        lfsr_reg <= {lfsr_reg[LFSR_WIDTH-2:0], feedback};

        // Compare random value with x_value
        // If rnd < x_value -> bit_out = 1, else 0
        if (rnd_value < x_value)
            bit_out <= 1'b1;
        else
            bit_out <= 1'b0;
    end
end

endmodule
