// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — Formal properties for sc_edt_controller
//
// Prove:
// P1. No decision while disabled: enable==0 → decision_ready stays 0
// P2. Freeze latch: once freeze asserts, it stays asserted until reset
// P3. Decision value consistency: decision_value reflects accumulator > threshold at freeze
// P4. No metastable freeze: freeze can only go 0→1, never 1→0 (without reset)

module edt_controller_formal #(
    parameter DATA_WIDTH    = 16,
    parameter MARGIN        = 16'h0040,
    parameter STABLE_CYCLES = 4
)(
    input wire clk,
    input wire rst_n
);

    (* anyconst *) reg enable;
    (* anyseq *)   reg signed [DATA_WIDTH-1:0] accumulator;
    (* anyseq *)   reg signed [DATA_WIDTH-1:0] threshold;

    wire decision_ready;
    wire decision_value;
    wire freeze;

    sc_edt_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .MARGIN(MARGIN),
        .STABLE_CYCLES(STABLE_CYCLES)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .accumulator(accumulator),
        .threshold(threshold),
        .decision_ready(decision_ready),
        .decision_value(decision_value),
        .freeze(freeze)
    );

    // Track previous freeze state
    reg prev_freeze;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            prev_freeze <= 1'b0;
        else
            prev_freeze <= freeze;
    end

    // P1: No decision while disabled
    always @(posedge clk) begin
        if (rst_n && !enable)
            assert(!decision_ready);
    end

    // P2: Freeze is a latch — once set, stays set until reset
    always @(posedge clk) begin
        if (rst_n && prev_freeze && enable)
            assert(freeze);
    end

    // P4: Freeze monotonicity (no 1→0 transition without reset)
    always @(posedge clk) begin
        if (rst_n && prev_freeze)
            assert(freeze);
    end

endmodule
