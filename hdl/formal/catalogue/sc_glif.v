// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Teeter 2018 GLIF5 default-profile Q32.32 RTL

`timescale 1ns / 1ps
`default_nettype none

// Compiler-bound default operating profile. Constants are nearest Q32.32
// encodings of the exact one-millisecond flow coefficients.
module sc_glif (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] I_t,
    output reg spike_out,
    output reg signed [63:0] v_out,
    output reg signed [63:0] theta_spike_out,
    output reg signed [63:0] i_asc1_out,
    output reg signed [63:0] i_asc2_out,
    output reg signed [63:0] theta_voltage_out,
    output reg [1:0] refractory_out
);

localparam signed [63:0] E_L = -64'sd300647710720;
localparam signed [63:0] THETA_INF = -64'sd214748364800;
localparam signed [63:0] MEMBRANE_DECAY = 64'sd3886247119;
localparam signed [63:0] SPIKE_DECAY = 64'sd4252231657;
localparam signed [63:0] ASC1_DECAY = 64'sd3886247119;
localparam signed [63:0] ASC2_DECAY = 64'sd4273546057;
localparam signed [63:0] STEADY_FORCING = 64'sd4273563864;
localparam signed [63:0] VOLTAGE_CONVOLUTION = 64'sd4066494874;
localparam signed [63:0] A_VOLTAGE = 64'sd429497;
localparam signed [63:0] DELTA_THETA_SPIKE = 64'sd8589934592;
localparam signed [63:0] DELTA_I_ASC1 = 64'sd4294967296;
localparam signed [63:0] DELTA_I_ASC2 = 64'sd2147483648;

reg signed [63:0] v_reg;
reg signed [63:0] theta_spike_reg;
reg signed [63:0] i_asc1_reg;
reg signed [63:0] i_asc2_reg;
reg signed [63:0] theta_voltage_reg;
reg [1:0] refractory_reg;

function automatic signed [63:0] qmul;
    input signed [63:0] left;
    input signed [63:0] right;
    reg signed [127:0] product;
    begin
        product = left * right;
        qmul = product >>> 32;
    end
endfunction

wire signed [63:0] total_current = I_t + i_asc1_reg + i_asc2_reg;
wire signed [63:0] voltage_offset = v_reg - E_L;
wire signed [63:0] next_v = E_L + total_current
    + qmul(voltage_offset - total_current, MEMBRANE_DECAY);
wire signed [63:0] next_theta_spike = qmul(theta_spike_reg, SPIKE_DECAY);
wire signed [63:0] next_i_asc1 = qmul(i_asc1_reg, ASC1_DECAY);
wire signed [63:0] next_i_asc2 = qmul(i_asc2_reg, ASC2_DECAY);
wire signed [63:0] threshold_forcing = qmul(total_current, STEADY_FORCING)
    + qmul(voltage_offset - total_current, VOLTAGE_CONVOLUTION);
wire signed [63:0] next_theta_voltage = qmul(theta_voltage_reg, SPIKE_DECAY)
    + qmul(A_VOLTAGE, threshold_forcing);
wire signed [63:0] next_threshold = THETA_INF + next_theta_spike
    + next_theta_voltage;
wire candidate_event = next_v > next_threshold;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        v_reg <= E_L;
        theta_spike_reg <= 64'sd0;
        i_asc1_reg <= 64'sd0;
        i_asc2_reg <= 64'sd0;
        theta_voltage_reg <= 64'sd0;
        refractory_reg <= 2'd0;
        spike_out <= 1'b0;
        v_out <= E_L;
        theta_spike_out <= 64'sd0;
        i_asc1_out <= 64'sd0;
        i_asc2_out <= 64'sd0;
        theta_voltage_out <= 64'sd0;
        refractory_out <= 2'd0;
    end else if (refractory_reg != 0) begin
        refractory_reg <= refractory_reg - 1'b1;
        refractory_out <= refractory_reg - 1'b1;
        spike_out <= 1'b0;
    end else if (candidate_event) begin
        v_reg <= E_L;
        theta_spike_reg <= next_theta_spike + DELTA_THETA_SPIKE;
        i_asc1_reg <= next_i_asc1 + DELTA_I_ASC1;
        i_asc2_reg <= next_i_asc2 + DELTA_I_ASC2;
        theta_voltage_reg <= next_theta_voltage;
        refractory_reg <= 2'd2;
        spike_out <= 1'b1;
        v_out <= E_L;
        theta_spike_out <= next_theta_spike + DELTA_THETA_SPIKE;
        i_asc1_out <= next_i_asc1 + DELTA_I_ASC1;
        i_asc2_out <= next_i_asc2 + DELTA_I_ASC2;
        theta_voltage_out <= next_theta_voltage;
        refractory_out <= 2'd2;
    end else begin
        v_reg <= next_v;
        theta_spike_reg <= next_theta_spike;
        i_asc1_reg <= next_i_asc1;
        i_asc2_reg <= next_i_asc2;
        theta_voltage_reg <= next_theta_voltage;
        spike_out <= 1'b0;
        v_out <= next_v;
        theta_spike_out <= next_theta_spike;
        i_asc1_out <= next_i_asc1;
        i_asc2_out <= next_i_asc2;
        theta_voltage_out <= next_theta_voltage;
    end
end

endmodule

`default_nettype wire
