// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Barrel-shifter signal decorrelator for stochastic bitstreams
//
// Addresses SSC (Stochastic Sequence Correlation) by circular-shifting
// N parallel stochastic bitstreams by distinct, configurable offsets.
// Allows sharing a single high-quality RNG (LFSR / Sobol) across
// neuron groups whilst maintaining bitstream independence.
//
// Architecture:
//   - N input streams fed through N barrel shifters with unique rotation
//   - Shift amounts derived from a counter XOR'd with a per-stream
//     constant, producing pseudo-independent phase offsets
//   - Zero latency: combinational shuffling, registered output
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
);

    // Running counter for dynamic shift variation
    reg [15:0] phase_counter;

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

            // Barrel-shift the shared source by shift_amount bits (circular rotate)
            wire [3:0] right_shift = STREAM_WIDTH[3:0] - shift_amount;
            wire [STREAM_WIDTH-1:0] shifted =
                (source_bits << shift_amount) |
                (source_bits >> right_shift);

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    decorrelated[i*STREAM_WIDTH +: STREAM_WIDTH] <= {STREAM_WIDTH{1'b0}};
                else
                    decorrelated[i*STREAM_WIDTH +: STREAM_WIDTH] <= shifted;
            end
        end
    endgenerate

endmodule
