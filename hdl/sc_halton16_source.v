// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Halton base-2 quasi-random source (Van der Corput)
//
// Hardware implementation of the Halton radical-inverse sequence
// for stochastic number generation.  Base-2 Halton is simply
// bit-reversal of the counter, making it trivially synthesisable
// with zero multipliers and zero LUTs for the core logic.
//
// Compared to LFSR: better uniformity, no correlation, no seed dependency.
// Compared to Sobol: no direction number storage, simpler hardware.
//
// Output: 16-bit quasi-random value per clock cycle, monotonically
// exploring [0, 2^16) with low discrepancy.

module sc_halton16_source #(
    parameter DATA_WIDTH = 16
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    enable,
    output reg  [DATA_WIDTH-1:0]   quasi_random,
    output reg                     valid
);

    // Running counter
    reg [DATA_WIDTH-1:0] counter;

    // Bit-reversal of counter = Van der Corput base-2 radical inverse
    // Scaled to [0, 2^DATA_WIDTH) integer representation
    wire [DATA_WIDTH-1:0] reversed;

    // Generate bit-reversal wiring (zero logic — pure routing)
    genvar i;
    generate
        for (i = 0; i < DATA_WIDTH; i = i + 1) begin : gen_rev
            assign reversed[i] = counter[DATA_WIDTH - 1 - i];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter      <= {DATA_WIDTH{1'b0}};
            quasi_random <= {DATA_WIDTH{1'b0}};
            valid        <= 1'b0;
        end else if (enable) begin
            quasi_random <= reversed;
            valid        <= 1'b1;
            counter      <= counter + 1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule
