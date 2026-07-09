// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Early Decision Termination controller for SC inference
//
// Monitors popcount accumulator convergence during stochastic computing
// inference. When the running tally exceeds a configurable confidence
// margin for STABLE_CYCLES consecutive clock cycles, the controller
// asserts `decision_ready` and freezes computation.
//
// Behaviour:
//   - Latency can be proportional to input difficulty, not worst-case
//   - Compatible with accumulate-and-compare downstream topologies
//
// Architecture:
//   - Continuously compares |accumulator − threshold| against MARGIN
//   - Stability counter tracks consecutive confident cycles
//   - Once stable for STABLE_CYCLES, asserts decision_ready + freeze
//   - freeze output can gate upstream clock enables or mux-select

module sc_edt_controller #(
    parameter DATA_WIDTH     = 16,
    parameter MARGIN         = 16'h0040,  // confidence margin (0.25 in Q8.8)
    parameter STABLE_CYCLES  = 8          // consecutive cycles before decision
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          enable,         // active-high: EDT monitoring on
    input  wire signed [DATA_WIDTH-1:0]  accumulator,    // running SC popcount value
    input  wire signed [DATA_WIDTH-1:0]  threshold,      // decision threshold
    output reg                           decision_ready, // asserted when confident
    output reg                           decision_value, // 1 if accum > threshold, 0 otherwise
    output reg                           freeze          // gate upstream computation
);

    // Absolute difference |accumulator - threshold|
    wire signed [DATA_WIDTH-1:0] diff = accumulator - threshold;
    wire signed [DATA_WIDTH-1:0] abs_diff = (diff < 0) ? (-diff) : diff;
    wire confident = (abs_diff >= $signed(MARGIN));

    // Stability counter
    reg [$clog2(STABLE_CYCLES+1)-1:0] stable_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stable_count   <= 0;
            decision_ready <= 1'b0;
            decision_value <= 1'b0;
            freeze         <= 1'b0;
        end else if (!enable) begin
            // EDT disabled — pass through, no freezing
            stable_count   <= 0;
            decision_ready <= 1'b0;
            freeze         <= 1'b0;
        end else if (freeze) begin
            // Already decided — hold state until reset or disable
            decision_ready <= 1'b1;
        end else begin
            if (confident) begin
                if (stable_count >= STABLE_CYCLES - 1) begin
                    decision_ready <= 1'b1;
                    decision_value <= (accumulator > threshold);
                    freeze         <= 1'b1;
                end else begin
                    stable_count <= stable_count + 1;
                end
            end else begin
                // Not confident — reset counter
                stable_count   <= 0;
                decision_ready <= 1'b0;
            end
        end
    end

endmodule
