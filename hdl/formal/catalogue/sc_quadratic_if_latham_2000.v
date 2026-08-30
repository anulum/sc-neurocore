// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Latham 2000 normalized QIF, explicit-Euler Q16.16
`timescale 1ns / 1ps

module sc_quadratic_if_latham_2000 #(
    parameter signed [31:0] P_V_RESET = -32'sd196608,
    parameter signed [31:0] P_V_PEAK = 32'sd677205
)(
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t,
    output reg spike_out,
    output reg signed [31:0] v_out
);

reg signed [31:0] v_reg;

// x' = x^2 + eta. The source's normalized 1 ms step is 0.05;
// Q16.16 encodes it as 3277/65536. This is the declared schema/RTL Euler
// representative, separate from the production runtimes' exact Riccati map.
wire signed [63:0] v_square = $signed(v_reg) * $signed(v_reg);
wire signed [63:0] derivative_q16 = (v_square >>> 16) + $signed(I_t);
wire signed [63:0] dt_product = derivative_q16 * 64'sd3277;
wire signed [63:0] dv_q16 = dt_product >>> 16;
wire signed [63:0] candidate =
    $signed({{32{v_reg[31]}}, v_reg}) + dv_q16;
wire signed [31:0] v_next =
    (candidate > 64'sd2147483647) ? 32'sd2147483647 :
    (candidate < -64'sd2147483648) ? -32'sd2147483648 :
    candidate[31:0];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        v_reg <= -32'sd65536;
        v_out <= -32'sd65536;
        spike_out <= 1'b0;
    end else if (v_next >= P_V_PEAK) begin
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
