// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Barrel-shifter signal decorrelator for stochastic bitstreams
//
// Addresses SSC (Stochastic Sequence Correlation) by circular-shifting
// N parallel stochastic bitstreams by configurable offsets.
// Shares one source across neuron groups with deterministic phase offsets.
// Rotations preserve each word's population count; they do not guarantee
// statistical independence or zero correlation between output streams.
//
// Architecture:
//   - N copies of the source word fed through N barrel shifters
//   - Rotation of stream i: (i XOR phase_counter[3:0] XOR seed nibble i)
//     modulo STREAM_WIDTH, a left rotation. Streams get distinct rotations
//     only when the values i XOR (seed nibble i) differ modulo STREAM_WIDTH;
//     the default seed gives eight distinct ones for STREAM_WIDTH = 16.
//   - One register stage: each output word is the rotation of the source
//     word of the previous clock.
// hdl/formal/decorrelator.sby proves the rotation, the population count and,
// for the default parameters, distinct streams for a one-hot source word.
//
// Usage:
//   Instantiate between the stochastic source and the neuron array.
//   Each neuron receives a decorrelated view of the shared bitstream.

module sc_decorrelator #(
    parameter NUM_STREAMS  = 8,     // number of parallel bitstreams
    parameter STREAM_WIDTH = 16,    // bits per stream (Q8.8 default)
    parameter SHIFT_SEED   = 32'hA5A5_5A5A  // diversification seed
)(
    input  wire                                  clk,
    input  wire                                  rst_n,
    input  wire [STREAM_WIDTH-1:0]               source_bits,  // shared source
    output reg  [NUM_STREAMS*STREAM_WIDTH-1:0]   decorrelated  // N decorrelated outputs
`ifdef FORMAL
    // Exact phase counter for the formal harness; absent from production ports.
    , output wire [15:0] formal_phase_counter
`endif
);

    // Width of the right-shift amount STREAM_WIDTH - rotation. For a power-of-two
    // width it wraps to 0 when rotation is 0 (x >> 0 | x << 0 = x), as the
    // 16-bit design always did; other widths keep the full range 1 .. STREAM_WIDTH
    // (x >> STREAM_WIDTH = 0).
    localparam integer POW2 = ((STREAM_WIDTH & (STREAM_WIDTH - 1)) == 0) ? 1 : 0;
    localparam integer BACK_BITS = (POW2 == 0) ? $clog2(STREAM_WIDTH + 1)
                                 : (STREAM_WIDTH > 1 ? $clog2(STREAM_WIDTH) : 1);

    // Running counter for dynamic shift variation
    reg [15:0] phase_counter;
`ifdef FORMAL
    assign formal_phase_counter = phase_counter;
`endif

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            phase_counter <= 16'd0;
        else
            phase_counter <= phase_counter + 16'd1;
    end

    // Generate per-stream decorrelated outputs via barrel shift
    genvar i;
    generate
        for (i = 0; i < NUM_STREAMS; i = i + 1) begin : gen_shift
            // Per-stream shift amount: XOR of stream index, counter, and seed nibble
            // Uses constant expression for seed extraction (elaboration-time)
            localparam [3:0] SEED_NIBBLE = SHIFT_SEED[(4*((i % 8))+3) -: 4];
            wire [3:0] shift_amount = (i[3:0] ^ phase_counter[3:0] ^ SEED_NIBBLE);

            // Rotate left by shift_amount modulo the word width. A shift of
            // STREAM_WIDTH or more would push every bit out of a narrower
            // word, so the amount is reduced first; a rotation by 0 keeps
            // the word (the right shift by STREAM_WIDTH gives 0).
            wire [3:0] rotation;
            if (STREAM_WIDTH >= 16) begin : g_rot_full
                assign rotation = shift_amount;
            end else if (POW2 == 1) begin : g_rot_mask
                localparam integer MASK = STREAM_WIDTH - 1;
                assign rotation = shift_amount & MASK[3:0];
            end else begin : g_rot_mod
                wire [31:0] reduced = {28'd0, shift_amount} % STREAM_WIDTH;
                assign rotation = reduced[3:0];
            end
            // The right shift is narrowed to the bits that STREAM_WIDTH needs,
            // so the shifter is no wider than the word.
            wire [31:0] back_full = STREAM_WIDTH - {28'd0, rotation};
            wire [BACK_BITS-1:0] back = back_full[BACK_BITS-1:0];
            wire [STREAM_WIDTH-1:0] shifted =
                (source_bits << rotation) |
                (source_bits >> back);

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    decorrelated[i*STREAM_WIDTH +: STREAM_WIDTH] <= {STREAM_WIDTH{1'b0}};
                else
                    decorrelated[i*STREAM_WIDTH +: STREAM_WIDTH] <= shifted;
            end
        end
    endgenerate

endmodule
