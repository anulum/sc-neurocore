// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — bounded safety harness for project non-resetting adaptive LIF

`default_nettype none
module sc_non_resetting_adaptive_lif_formal(
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] current_t
);
    localparam signed [63:0] CURRENT_MIN = -64'sd171798691840;
    localparam signed [63:0] CURRENT_MAX = 64'sd171798691840;
    localparam signed [63:0] V_MIN = -64'sd858993459200;
    localparam signed [63:0] V_MAX = 64'sd858993459200;
    localparam signed [63:0] THETA_MIN = -64'sd858993459200;
    localparam signed [63:0] THETA_MAX = 64'sd8796093022208;
    wire signed [63:0] v;
    wire signed [63:0] theta;
    wire event_out;
    sc_non_resetting_adaptive_lif uut(
        .clk(clk), .rst_n(rst_n), .current_t(current_t),
        .v_out(v), .theta_out(theta), .event_out(event_out)
    );
`ifdef FORMAL
    reg past_valid = 1'b0;
    initial assume(!rst_n);
    always @(posedge clk) begin
        assume(current_t >= CURRENT_MIN && current_t <= CURRENT_MAX);
        if (!past_valid) assume(!rst_n); else assume(rst_n);
        past_valid <= 1'b1;
        if (past_valid && $past(!rst_n)) begin
            assert(v == -64'sd279172874240);
            assert(theta == -64'sd214748364800 && !event_out);
        end
        if (past_valid && rst_n) begin
            assert(v >= V_MIN && v <= V_MAX);
            assert(theta >= THETA_MIN && theta <= THETA_MAX);
        end
    end
`endif
endmodule
`default_nettype wire
