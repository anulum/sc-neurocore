// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Kobayashi 2009 MAT* enrolled in signed Q32.32

// One rising edge advances one 0.001 ms source-model step. current_t is nA in
// signed Q32.32; voltage and adaptive-threshold outputs are mV in signed
// Q32.32; refractory_out is ms in signed Q32.32. The default enrolled profile
// is the paper's RS example. Voltage is never reset. Multiplications truncate
// arithmetically to Q32.32 and explicit saturation prevents wraparound. There
// is one-cycle sampled-event latency and no backpressure. This bounded fixed-
// point representative does not claim binary64 equivalence, device timing,
// PPA, or physical-silicon evidence.
`timescale 1ns / 1ps

module sc_mat (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] current_t,
    output reg signed [63:0] v_out,
    output reg signed [63:0] theta1_out,
    output reg signed [63:0] theta2_out,
    output reg signed [63:0] refractory_out,
    output reg event_out
);
    localparam signed [63:0] DT = 64'sd4294967;
    localparam signed [63:0] DT_OVER_TAU_M = 64'sd858993;
    localparam signed [63:0] DECAY_1 = 64'sd4294537821;
    localparam signed [63:0] DECAY_2 = 64'sd4294945821;
    localparam signed [63:0] OMEGA = 64'sd81604378624;
    localparam signed [63:0] ALPHA_1 = 64'sd158913789952;
    localparam signed [63:0] ALPHA_2 = 64'sd8589934592;
    localparam signed [63:0] REFRACTORY_PERIOD = 64'sd8589934592;
    localparam signed [63:0] V_MIN = -64'sd858993459200;
    localparam signed [63:0] V_MAX = 64'sd858993459200;
    localparam signed [63:0] THETA_MAX = 64'sd8796093022208;

    reg signed [63:0] v_reg;
    reg signed [63:0] theta1_reg;
    reg signed [63:0] theta2_reg;
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
    wire signed [127:0] v_candidate_raw = $signed(v_reg) + qmul(membrane_drive_q, DT_OVER_TAU_M);
    wire signed [63:0] v_candidate = bound_voltage(v_candidate_raw);
    wire signed [63:0] theta1_decay = bound_theta(qmul(theta1_reg, DECAY_1));
    wire signed [63:0] theta2_decay = bound_theta(qmul(theta2_reg, DECAY_2));
    wire signed [63:0] refractory_decay =
        refractory_reg <= DT ? 64'sd0 : refractory_reg - DT;
    wire signed [127:0] threshold = $signed(OMEGA) + $signed(theta1_decay) + $signed(theta2_decay);
    wire candidate_event = refractory_decay == 0 && $signed(v_candidate) >= $signed(threshold);
    wire signed [63:0] theta1_event = bound_theta($signed(theta1_decay) + $signed(ALPHA_1));
    wire signed [63:0] theta2_event = bound_theta($signed(theta2_decay) + $signed(ALPHA_2));

    always @(posedge clk) begin
        if (!rst_n) begin
            v_reg <= 0;
            theta1_reg <= 0;
            theta2_reg <= 0;
            refractory_reg <= 0;
            v_out <= 0;
            theta1_out <= 0;
            theta2_out <= 0;
            refractory_out <= 0;
            event_out <= 0;
        end else begin
            v_reg <= v_candidate;
            theta1_reg <= candidate_event ? theta1_event : theta1_decay;
            theta2_reg <= candidate_event ? theta2_event : theta2_decay;
            refractory_reg <= candidate_event ? REFRACTORY_PERIOD : refractory_decay;
            v_out <= v_candidate;
            theta1_out <= candidate_event ? theta1_event : theta1_decay;
            theta2_out <= candidate_event ? theta2_event : theta2_decay;
            refractory_out <= candidate_event ? REFRACTORY_PERIOD : refractory_decay;
            event_out <= candidate_event;
        end
    end
endmodule
