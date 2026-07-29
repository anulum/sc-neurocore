// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-faithful Nagumo–Sato Q16.16 RTL

// One-cycle map contract:
// y' = k*y - alpha*H(y) + bias + I, H(0)=1; spike_out = H(y').
// Every public numeric port and parameter uses signed Q16.16.
`timescale 1ns / 1ps

module sc_nagumo_sato_map #(
    parameter signed [31:0] P_K = 32'sd39322,
    parameter signed [31:0] P_ALPHA = 32'sd65536,
    parameter signed [31:0] P_BIAS = 32'sd13107,
    parameter signed [31:0] P_INITIAL_Y = 32'sd6554
)(
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t,
    output reg spike_out,
    output reg signed [31:0] y_out
);
    reg signed [31:0] y_reg;
    wire signed [63:0] y_product = P_K * y_reg;
    wire signed [63:0] y_decay = y_product >>> 16;
    wire signed [63:0] refractory = y_reg >= 0 ?
        $signed({{32{P_ALPHA[31]}}, P_ALPHA}) : 64'sd0;
    wire signed [63:0] candidate = y_decay - refractory +
        $signed({{32{P_BIAS[31]}}, P_BIAS}) +
        $signed({{32{I_t[31]}}, I_t});
    wire signed [31:0] y_next = candidate > 64'sd2147483647 ? 32'sd2147483647 :
        candidate < -64'sd2147483648 ? -32'sd2147483648 : candidate[31:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_reg <= P_INITIAL_Y;
            y_out <= P_INITIAL_Y;
            spike_out <= 1'b0;
        end else begin
            y_reg <= y_next;
            y_out <= y_next;
            spike_out <= ~y_next[31];
        end
    end
endmodule
