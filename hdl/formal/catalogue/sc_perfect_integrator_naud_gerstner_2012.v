// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Naud-Gerstner 2012 perfect integrator, Q8.8
`timescale 1ns / 1ps

module sc_perfect_integrator_naud_gerstner_2012 #(
    parameter signed [15:0] P_C_M = 16'sd256,
    parameter signed [15:0] P_V_THRESHOLD = 16'sd256,
    parameter signed [15:0] P_V_RESET = 16'sd0
)(
    input wire clk,
    input wire rst_n,
    input wire signed [15:0] I_t,
    output reg spike_out,
    output reg signed [15:0] v_out
);

reg signed [15:0] v_reg;
// The enrolled dt is the exact rational 1/10. Keeping that ratio explicit
// preserves the source equality boundary instead of rounding dt to 26/256.
wire signed [31:0] current_q16 = $signed({{16{I_t[15]}}, I_t}) <<< 8;
wire signed [31:0] denominator = $signed(P_C_M) * 32'sd10;
wire signed [31:0] quotient_q8 = current_q16 / denominator;
wire signed [15:0] dv = quotient_q8[15:0];
wire signed [16:0] v_raw = v_reg + dv;
wire signed [15:0] v_next =
    (v_raw > 17'sd32767) ? 16'sd32767 :
    (v_raw < (-17'sd32768)) ? (-16'sd32768) : v_raw[15:0];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        v_reg <= 16'sd0;
        v_out <= 16'sd0;
        spike_out <= 1'b0;
    end else if (v_next > P_V_THRESHOLD) begin
        spike_out <= 1'b1;
        v_reg <= P_V_RESET;
        v_out <= P_V_RESET;
    end else begin
        spike_out <= 1'b0;
        v_reg <= v_next;
        v_out <= v_next;
    end
end

endmodule
