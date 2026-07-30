// SPDX-License-Identifier: AGPL-3.0-or-later
// SC-NeuroCore — bounded safety harness for source MAT* Q32.32

`default_nettype none
module sc_mat_formal(
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] current_t
);
    localparam signed [63:0] CURRENT_MIN = -64'sd17179869184;
    localparam signed [63:0] CURRENT_MAX = 64'sd17179869184;
    localparam signed [63:0] V_MIN = -64'sd858993459200;
    localparam signed [63:0] V_MAX = 64'sd858993459200;
    localparam signed [63:0] THETA_MAX = 64'sd8796093022208;
    localparam signed [63:0] REFRACTORY_PERIOD = 64'sd8589934592;
    wire signed [63:0] v;
    wire signed [63:0] theta1;
    wire signed [63:0] theta2;
    wire signed [63:0] refractory;
    wire event_out;
    sc_mat uut(
        .clk(clk), .rst_n(rst_n), .current_t(current_t), .v_out(v),
        .theta1_out(theta1), .theta2_out(theta2),
        .refractory_out(refractory), .event_out(event_out)
    );
`ifdef FORMAL
    reg past_valid = 1'b0;
    initial assume(!rst_n);
    always @(posedge clk) begin
        assume(current_t >= CURRENT_MIN && current_t <= CURRENT_MAX);
        if (!past_valid) assume(!rst_n); else assume(rst_n);
        past_valid <= 1'b1;
        if (past_valid && $past(!rst_n)) begin
            assert(v == 0 && theta1 == 0 && theta2 == 0);
            assert(refractory == 0 && !event_out);
        end
        if (past_valid && rst_n) begin
            assert(v >= V_MIN && v <= V_MAX);
            assert(theta1 >= 0 && theta1 <= THETA_MAX);
            assert(theta2 >= 0 && theta2 <= THETA_MAX);
            assert(refractory >= 0 && refractory <= REFRACTORY_PERIOD);
            if (event_out) assert(refractory == REFRACTORY_PERIOD);
        end
    end
`endif
endmodule
`default_nettype wire
