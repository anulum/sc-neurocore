// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Lapicque 1907 normalized polarization-threshold RTL

`timescale 1ns / 1ps
`default_nettype none

// Q32.32 specialization of Lapicque's constant-source-voltage exact flow:
//   v[n+1] = exp(-dt/beta) v[n] + (1-exp(-dt/beta)) V rho/(R+rho)
// for the maintained normalized receipt K=1.1, R=10, rho=1, dt=0.01 ms,
// beta=1 ms, and v_threshold=1. It latches the first threshold attainment;
// no automatic post-event reset is introduced.
module sc_lapicque_1907 #(
    parameter signed [63:0] P_V_THRESHOLD = 64'sd4294967296,
    parameter signed [63:0] P_DECAY = 64'sd4252231657,
    parameter signed [63:0] P_INPUT_GAIN = 64'sd3885058
) (
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] V_t,
    output reg spike_out,
    output reg excited_out,
    output reg signed [63:0] v_out
);

    localparam signed [127:0] MAX_Q3232 = 128'sd9223372036854775807;
    localparam signed [127:0] MIN_Q3232 = -128'sd9223372036854775808;

    reg signed [63:0] v_reg;
    wire signed [127:0] v_product = v_reg * P_DECAY;
    wire signed [127:0] input_product = V_t * P_INPUT_GAIN;
    wire signed [127:0] candidate_wide =
        (v_product >>> 32) + (input_product >>> 32);
    wire signed [63:0] v_candidate =
        (candidate_wide > MAX_Q3232) ? 64'sh7fff_ffff_ffff_ffff :
        (candidate_wide < MIN_Q3232) ? 64'sh8000_0000_0000_0000 :
        candidate_wide[63:0];
    wire first_attainment = !excited_out && (v_candidate >= P_V_THRESHOLD);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_reg <= 64'sd0;
            v_out <= 64'sd0;
            excited_out <= 1'b0;
            spike_out <= 1'b0;
        end else begin
            v_reg <= v_candidate;
            v_out <= v_candidate;
            spike_out <= first_attainment;
            if (first_attainment)
                excited_out <= 1'b1;
        end
    end

endmodule

`default_nettype wire
