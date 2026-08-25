// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — bounded safety harness for SC resetting MAT Q32.32

`default_nettype none
module sc_resetting_mat_formal(
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] current_t
);
    localparam signed [63:0] CURRENT_MIN = -64'sd1099511627776;
    localparam signed [63:0] CURRENT_MAX = 64'sd1099511627776;
    localparam signed [63:0] V_RESET = -64'sd300647710720;
    localparam signed [63:0] V_MIN = -64'sd858993459200;
    localparam signed [63:0] V_MAX = 64'sd429496729600;
    localparam signed [63:0] THETA_MAX = 64'sd8796093022208;
    wire signed [63:0] v;
    wire signed [63:0] theta1;
    wire signed [63:0] theta2;
    wire event_out;
    sc_resetting_mat uut(
        .clk(clk), .rst_n(rst_n), .current_t(current_t), .v_out(v),
        .theta1_out(theta1), .theta2_out(theta2), .event_out(event_out)
    );
`ifdef FORMAL
    reg past_valid = 1'b0;
    initial assume(!rst_n);
    always @(posedge clk) begin
        assume(current_t >= CURRENT_MIN && current_t <= CURRENT_MAX);
        if (!past_valid) assume(!rst_n); else assume(rst_n);
        past_valid <= 1'b1;
        if (past_valid && $past(!rst_n)) begin
            assert(v == V_RESET && theta1 == 0 && theta2 == 0 && !event_out);
        end
        if (past_valid && rst_n) begin
            assert(v >= V_MIN && v <= V_MAX);
            assert(theta1 >= 0 && theta1 <= THETA_MAX);
            assert(theta2 >= 0 && theta2 <= THETA_MAX);
            if (event_out) assert(v == V_RESET);
        end
    end
`endif
endmodule
`default_nettype wire
