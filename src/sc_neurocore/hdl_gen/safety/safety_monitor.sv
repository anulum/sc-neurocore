// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neuro-Safe Hardware Monitor
//
// Enforces all 6 formally proven safety boundaries at the nanosecond scale.
// Each property maps 1:1 to a theorem in safety_bounds.lean.
//
// Property Map:
//   [P1] monitor_soundness   → halt ↔ coherence < limit
//   [P2] safe_transition      → monotone coherence preserved
//   [P3] sc_precision_bound   → variance proxy: 4*k*(N-k) ≤ N²
//   [P4] sc_add_preserves_range → SC addition result ≤ denominator
//   [P5] lif_membrane_bounded  → membrane ≤ v_max
//   [P6] correlation_range     → |SCC numerator| ≤ denominator

module neuro_safe_monitor #(
    parameter [15:0] MAX_CURRENT     = 16'h7FFF,
    parameter [15:0] MAX_VOLTAGE     = 16'hC000,
    parameter [15:0] COHERENCE_LIMIT = 16'h0100,
    parameter [15:0] SC_DENOM        = 16'h0100,  // bitstream length N
    parameter [15:0] LIF_V_MAX       = 16'hC000
)(
    input wire        clk,
    input wire        rst_n,

    // [P1/P2] Controller probes
    input wire [15:0] probe_current,
    input wire [15:0] probe_voltage,
    input wire [15:0] probe_coherence,

    // [P3] SC precision probes
    input wire [15:0] probe_popcount_k,  // number of 1-bits in bitstream

    // [P4] SC addition probes
    input wire [15:0] probe_sc_add_result,

    // [P5] LIF membrane probes
    input wire [15:0] probe_membrane,

    // [P6] SCC correlation probes
    input wire signed [15:0] probe_scc_numer,
    input wire        [15:0] probe_scc_denom,

    // Halt and violation flags
    output reg        hardware_halt,
    output reg [5:0]  violation_flags  // one bit per property
);

    // Internal state for [P2] monotone coherence tracking
    reg [15:0] prev_coherence;

    // Combinational violation detection
    wire v_current_voltage;   // [P1]
    wire v_coherence;         // [P1]
    wire v_monotone;          // [P2]
    wire v_precision;         // [P3]
    wire v_sc_range;          // [P4]
    wire v_membrane;          // [P5]
    wire v_scc_range;         // [P6]

    // [P1] monitor_soundness: halt when current or voltage exceeds limit,
    //       or coherence drops below threshold
    assign v_current_voltage = (probe_current > MAX_CURRENT) ||
                               (probe_voltage > MAX_VOLTAGE);
    assign v_coherence = (probe_coherence < COHERENCE_LIMIT);

    // [P2] safe_transition: coherence must not decrease between cycles
    //       s2.coherence >= s1.coherence (monotone)
    assign v_monotone = (probe_coherence < prev_coherence);

    // [P3] sc_precision_bound: 4*k*(N-k) must be ≤ N²
    //       Simplified check: popcount must be in [0, N]
    assign v_precision = (probe_popcount_k > SC_DENOM);

    // [P4] sc_add_preserves_range: result of SC addition ≤ denominator
    assign v_sc_range = (probe_sc_add_result > SC_DENOM);

    // [P5] lif_membrane_bounded: membrane ≤ v_max
    assign v_membrane = (probe_membrane > LIF_V_MAX);

    // [P6] correlation_range: |SCC numerator| ≤ denominator
    //       For signed comparison: -denom ≤ numer ≤ denom
    wire [15:0] abs_scc_numer;
    assign abs_scc_numer = (probe_scc_numer < 0) ?
                           (~probe_scc_numer[15:0] + 1'b1) :
                           probe_scc_numer[15:0];
    assign v_scc_range = (abs_scc_numer > probe_scc_denom);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hardware_halt   <= 0;
            violation_flags <= 6'b000000;
            prev_coherence  <= 16'h0000;
        end else begin
            // Track coherence for monotone check
            prev_coherence <= probe_coherence;

            // Latch violation flags (sticky — once set, require reset to clear)
            violation_flags[0] <= violation_flags[0] | v_current_voltage | v_coherence;
            violation_flags[1] <= violation_flags[1] | v_monotone;
            violation_flags[2] <= violation_flags[2] | v_precision;
            violation_flags[3] <= violation_flags[3] | v_sc_range;
            violation_flags[4] <= violation_flags[4] | v_membrane;
            violation_flags[5] <= violation_flags[5] | v_scc_range;

            // Assert halt on ANY property violation
            if (v_current_voltage || v_coherence || v_monotone ||
                v_precision || v_sc_range || v_membrane || v_scc_range) begin
                hardware_halt <= 1;
            end
        end
    end

endmodule
