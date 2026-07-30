// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Kobayashi 2009 MAT(1) enrolled in signed Q32.32

// One rising edge advances one 0.001 ms source-model step. Voltage is never
// reset; a sampled event increments the one-timescale threshold history and
// starts the 2 ms absolute refractory interval. This bounded fixed-point
// representative makes no binary64-equivalence, timing, PPA, or device claim.
`timescale 1ns / 1ps

module sc_non_resetting_lif (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] current_t,
    output reg signed [63:0] v_out,
    output reg signed [63:0] theta_out,
    output reg signed [63:0] refractory_out,
    output reg event_out
);
    localparam signed [63:0] DT = 64'sd4294967;
    localparam signed [63:0] DT_OVER_TAU_M = 64'sd858993;
    localparam signed [63:0] THETA_DECAY = 64'sd4294881398;
    localparam signed [63:0] OMEGA = 64'sd81604378624;
    localparam signed [63:0] ALPHA = 64'sd158913789952;
    localparam signed [63:0] REFRACTORY_PERIOD = 64'sd8589934592;
    localparam signed [63:0] V_MIN = -64'sd858993459200;
    localparam signed [63:0] V_MAX = 64'sd858993459200;
    localparam signed [63:0] THETA_MAX = 64'sd8796093022208;

    reg signed [63:0] v_reg;
    reg signed [63:0] theta_reg;
    reg signed [63:0] refractory_reg;

    function automatic signed [63:0] qmul(
        input signed [63:0] left,
        input signed [63:0] right
    );
        reg signed [127:0] product;
        begin
            product = $signed(left) * $signed(right);
            qmul = product >>> 32;
        end
    endfunction

    function automatic signed [63:0] bound_voltage(input signed [127:0] value);
        begin
            if (value < V_MIN) bound_voltage = V_MIN;
            else if (value > V_MAX) bound_voltage = V_MAX;
            else bound_voltage = value[63:0];
        end
    endfunction

    function automatic signed [63:0] bound_theta(input signed [127:0] value);
        begin
            if (value < 0) bound_theta = 0;
            else if (value > THETA_MAX) bound_theta = THETA_MAX;
            else bound_theta = value[63:0];
        end
    endfunction

    wire signed [127:0] resistance_current = $signed(current_t) * 128'sd50;
    wire signed [127:0] membrane_drive = -$signed(v_reg) + resistance_current;
    wire signed [63:0] membrane_drive_q = bound_voltage(membrane_drive);
    wire signed [63:0] v_candidate =
        bound_voltage($signed(v_reg) + qmul(membrane_drive_q, DT_OVER_TAU_M));
    wire signed [63:0] theta_decay = bound_theta(qmul(theta_reg, THETA_DECAY));
    wire signed [63:0] refractory_decay =
        refractory_reg <= DT ? 64'sd0 : refractory_reg - DT;
    wire candidate_event = refractory_decay == 0
        && $signed(v_candidate) >= $signed(OMEGA) + $signed(theta_decay);
    wire signed [63:0] theta_event = bound_theta($signed(theta_decay) + $signed(ALPHA));

    always @(posedge clk) begin
        if (!rst_n) begin
            v_reg <= 0;
            theta_reg <= 0;
            refractory_reg <= 0;
            v_out <= 0;
            theta_out <= 0;
            refractory_out <= 0;
            event_out <= 0;
        end else begin
            v_reg <= v_candidate;
            theta_reg <= candidate_event ? theta_event : theta_decay;
            refractory_reg <= candidate_event ? REFRACTORY_PERIOD : refractory_decay;
            v_out <= v_candidate;
            theta_out <= candidate_event ? theta_event : theta_decay;
            refractory_out <= candidate_event ? REFRACTORY_PERIOD : refractory_decay;
            event_out <= candidate_event;
        end
    end
endmodule
