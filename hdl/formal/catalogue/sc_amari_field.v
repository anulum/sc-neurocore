// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — four-site Amari 1977 neural field in signed Q16.16

// One rising edge advances all four periodic sites simultaneously. Inputs and
// state outputs are signed Q16.16. `mean_rate_out` is unsigned-by-contract
// Q16.16 in {0, .25, .5, .75, 1}; it is a population rate, not a spike.
// Latency is one cycle, active-high behavior is gated by active-low reset, and
// state saturates to [-8, 8] solely as the enrolled hardware safety envelope.
`timescale 1ns / 1ps

module sc_amari_field (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I0_t,
    input wire signed [31:0] I1_t,
    input wire signed [31:0] I2_t,
    input wire signed [31:0] I3_t,
    output reg signed [31:0] u0_out,
    output reg signed [31:0] u1_out,
    output reg signed [31:0] u2_out,
    output reg signed [31:0] u3_out,
    output reg [31:0] mean_rate_out
);
    localparam signed [31:0] W0 = 32'sd49152;
    localparam signed [31:0] W1 = 32'sd6352;
    localparam signed [31:0] W2 = -32'sd4778;
    localparam signed [31:0] DX = 32'sd32768;
    localparam signed [31:0] DT_OVER_TAU = 32'sd3277;
    localparam signed [31:0] Q_LIMIT = 32'sd524288;

    reg signed [31:0] u0_reg, u1_reg, u2_reg, u3_reg;
    wire a0 = u0_reg > 0;
    wire a1 = u1_reg > 0;
    wire a2 = u2_reg > 0;
    wire a3 = u3_reg > 0;

    wire signed [34:0] interaction0 = (a0 ? W0 : 0) + (a1 ? W1 : 0) + (a2 ? W2 : 0) + (a3 ? W1 : 0);
    wire signed [34:0] interaction1 = (a0 ? W1 : 0) + (a1 ? W0 : 0) + (a2 ? W1 : 0) + (a3 ? W2 : 0);
    wire signed [34:0] interaction2 = (a0 ? W2 : 0) + (a1 ? W1 : 0) + (a2 ? W0 : 0) + (a3 ? W1 : 0);
    wire signed [34:0] interaction3 = (a0 ? W1 : 0) + (a1 ? W2 : 0) + (a2 ? W1 : 0) + (a3 ? W0 : 0);

    wire signed [63:0] conv0 = ($signed(interaction0) * $signed(DX)) >>> 16;
    wire signed [63:0] conv1 = ($signed(interaction1) * $signed(DX)) >>> 16;
    wire signed [63:0] conv2 = ($signed(interaction2) * $signed(DX)) >>> 16;
    wire signed [63:0] conv3 = ($signed(interaction3) * $signed(DX)) >>> 16;
    wire signed [63:0] rhs0 = -$signed(u0_reg) + conv0 + $signed(I0_t);
    wire signed [63:0] rhs1 = -$signed(u1_reg) + conv1 + $signed(I1_t);
    wire signed [63:0] rhs2 = -$signed(u2_reg) + conv2 + $signed(I2_t);
    wire signed [63:0] rhs3 = -$signed(u3_reg) + conv3 + $signed(I3_t);
    wire signed [63:0] raw0 = $signed(u0_reg) + (($signed(rhs0) * $signed(DT_OVER_TAU)) >>> 16);
    wire signed [63:0] raw1 = $signed(u1_reg) + (($signed(rhs1) * $signed(DT_OVER_TAU)) >>> 16);
    wire signed [63:0] raw2 = $signed(u2_reg) + (($signed(rhs2) * $signed(DT_OVER_TAU)) >>> 16);
    wire signed [63:0] raw3 = $signed(u3_reg) + (($signed(rhs3) * $signed(DT_OVER_TAU)) >>> 16);

    function automatic signed [31:0] saturate_state(input signed [63:0] value);
        begin
            if (value > Q_LIMIT) saturate_state = Q_LIMIT;
            else if (value < -Q_LIMIT) saturate_state = -Q_LIMIT;
            else saturate_state = value[31:0];
        end
    endfunction

    wire signed [31:0] next0 = saturate_state(raw0);
    wire signed [31:0] next1 = saturate_state(raw1);
    wire signed [31:0] next2 = saturate_state(raw2);
    wire signed [31:0] next3 = saturate_state(raw3);
    wire [2:0] active_count = {2'b0, (next0 > 0)} + {2'b0, (next1 > 0)} +
                              {2'b0, (next2 > 0)} + {2'b0, (next3 > 0)};

    always @(posedge clk) begin
        if (!rst_n) begin
            u0_reg <= 0; u1_reg <= 0; u2_reg <= 0; u3_reg <= 0;
            u0_out <= 0; u1_out <= 0; u2_out <= 0; u3_out <= 0;
            mean_rate_out <= 0;
        end else begin
            u0_reg <= next0; u1_reg <= next1; u2_reg <= next2; u3_reg <= next3;
            u0_out <= next0; u1_out <= next1; u2_out <= next2; u3_out <= next3;
            mean_rate_out <= {15'd0, active_count, 14'd0};
        end
    end
endmodule
