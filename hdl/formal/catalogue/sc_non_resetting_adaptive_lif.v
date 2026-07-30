// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained project non-resetting adaptive LIF in Q32.32

// One rising edge advances the exact affine-rest recurrence by 0.1 ms.
// Voltage is never reset and no refractory gate is applied. This is a bounded
// fixed-point project representative, not a literature or device claim.
`timescale 1ns / 1ps

module sc_non_resetting_adaptive_lif (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] current_t,
    output reg signed [63:0] v_out,
    output reg signed [63:0] theta_out,
    output reg event_out
);
    localparam signed [63:0] V_REST = -64'sd279172874240;
    localparam signed [63:0] THETA_REST = -64'sd214748364800;
    localparam signed [63:0] DELTA_THETA = 64'sd21474836480;
    localparam signed [63:0] V_DECAY = 64'sd4252231657;
    localparam signed [63:0] THETA_DECAY = 64'sd4286385946;
    localparam signed [63:0] V_MIN = -64'sd858993459200;
    localparam signed [63:0] V_MAX = 64'sd858993459200;
    localparam signed [63:0] THETA_MIN = -64'sd858993459200;
    localparam signed [63:0] THETA_MAX = 64'sd8796093022208;

    reg signed [63:0] v_reg;
    reg signed [63:0] theta_reg;

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
            if (value < THETA_MIN) bound_theta = THETA_MIN;
            else if (value > THETA_MAX) bound_theta = THETA_MAX;
            else bound_theta = value[63:0];
        end
    endfunction

    wire signed [63:0] equilibrium = bound_voltage($signed(V_REST) + $signed(current_t));
    wire signed [63:0] v_candidate = bound_voltage(
        $signed(equilibrium) + qmul($signed(v_reg) - $signed(equilibrium), V_DECAY)
    );
    wire signed [63:0] theta_decay = bound_theta(
        $signed(THETA_REST) + qmul($signed(theta_reg) - $signed(THETA_REST), THETA_DECAY)
    );
    wire candidate_event = $signed(v_candidate) >= $signed(theta_decay);
    wire signed [63:0] theta_event = bound_theta($signed(theta_decay) + $signed(DELTA_THETA));

    always @(posedge clk) begin
        if (!rst_n) begin
            v_reg <= V_REST;
            theta_reg <= THETA_REST;
            v_out <= V_REST;
            theta_out <= THETA_REST;
            event_out <= 0;
        end else begin
            v_reg <= v_candidate;
            theta_reg <= candidate_event ? theta_event : theta_decay;
            v_out <= v_candidate;
            theta_out <= candidate_event ? theta_event : theta_decay;
            event_out <= candidate_event;
        end
    end
endmodule
