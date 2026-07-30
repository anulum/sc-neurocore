// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — project resetting MAT enrolled in signed Q32.32

// One rising edge advances the historical 1 ms SC recurrence. The linear RK4
// stages are algebraically collapsed to their fourth-order stability
// polynomial: 0.9048375 for tau=10 ms and 0.9950124791927083 for tau=200 ms.
// current_t, voltage, and threshold states are signed Q32.32. A sampled event
// resets voltage and adds both threshold increments. Products truncate
// arithmetically; explicit saturation prevents wraparound. Latency is one
// rising edge. This representative claims neither binary64 identity nor timing,
// PPA, device, or physical-silicon evidence.
`timescale 1ns / 1ps

module sc_resetting_mat (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] current_t,
    output reg signed [63:0] v_out,
    output reg signed [63:0] theta1_out,
    output reg signed [63:0] theta2_out,
    output reg event_out
);
    localparam signed [63:0] V_REST = -64'sd300647710720;
    localparam signed [63:0] V_RESET = -64'sd300647710720;
    localparam signed [63:0] V_THRESHOLD_BASE = -64'sd214748364800;
    localparam signed [63:0] RK4_FAST = 64'sd3886247471;
    localparam signed [63:0] RK4_SLOW = 64'sd4273546057;
    localparam signed [63:0] H1 = 64'sd21474836480;
    localparam signed [63:0] H2 = 64'sd12884901888;
    localparam signed [63:0] V_MIN = -64'sd858993459200;
    localparam signed [63:0] V_MAX = 64'sd429496729600;
    localparam signed [63:0] THETA_MAX = 64'sd8796093022208;

    reg signed [63:0] v_reg;
    reg signed [63:0] theta1_reg;
    reg signed [63:0] theta2_reg;

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

    wire signed [127:0] equilibrium_raw = $signed(V_REST) + $signed(current_t);
    wire signed [63:0] equilibrium = bound_voltage(equilibrium_raw);
    wire signed [127:0] v_candidate_raw =
        $signed(equilibrium) + qmul($signed(v_reg) - $signed(equilibrium), RK4_FAST);
    wire signed [63:0] v_candidate = bound_voltage(v_candidate_raw);
    wire signed [63:0] theta1_candidate = bound_theta(qmul(theta1_reg, RK4_FAST));
    wire signed [63:0] theta2_candidate = bound_theta(qmul(theta2_reg, RK4_SLOW));
    wire signed [127:0] threshold =
        $signed(V_THRESHOLD_BASE) + $signed(theta1_candidate) + $signed(theta2_candidate);
    wire candidate_event = $signed(v_candidate) >= $signed(threshold);
    wire signed [63:0] theta1_event = bound_theta($signed(theta1_candidate) + $signed(H1));
    wire signed [63:0] theta2_event = bound_theta($signed(theta2_candidate) + $signed(H2));

    always @(posedge clk) begin
        if (!rst_n) begin
            v_reg <= V_REST;
            theta1_reg <= 0;
            theta2_reg <= 0;
            v_out <= V_REST;
            theta1_out <= 0;
            theta2_out <= 0;
            event_out <= 0;
        end else begin
            v_reg <= candidate_event ? V_RESET : v_candidate;
            theta1_reg <= candidate_event ? theta1_event : theta1_candidate;
            theta2_reg <= candidate_event ? theta2_event : theta2_candidate;
            v_out <= candidate_event ? V_RESET : v_candidate;
            theta1_out <= candidate_event ? theta1_event : theta1_candidate;
            theta2_out <= candidate_event ? theta2_event : theta2_candidate;
            event_out <= candidate_event;
        end
    end
endmodule
