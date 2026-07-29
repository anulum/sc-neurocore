// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — formal harness for the SC adaptive-threshold Q8.24 map

`default_nettype none

module sc_adaptive_threshold_map_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t
);
    localparam signed [31:0] Q_FIVE = 32'sd83886080;
    localparam signed [31:0] X_THRESHOLD = 32'sd13421773;
    wire spike_out;
    wire signed [31:0] x_out;
    wire signed [31:0] theta_out;

    sc_adaptive_threshold_map uut (
        .clk(clk), .rst_n(rst_n), .I_t(I_t), .spike_out(spike_out),
        .x_out(x_out), .theta_out(theta_out)
    );

`ifdef FORMAL
    reg past_valid = 1'b0;
    initial assume (!rst_n);
    always @(posedge clk) begin
        if (!past_valid)
            assume (!rst_n);
        else
            assume (rst_n);
        past_valid <= 1'b1;
        if (!rst_n)
            assert (spike_out == 1'b0);
        if (past_valid && rst_n && $past(rst_n)) begin
            assert (x_out >= -Q_FIVE && x_out <= Q_FIVE);
            assert (theta_out >= -Q_FIVE && theta_out <= Q_FIVE);
            assert (spike_out == ($past(x_out) < X_THRESHOLD && x_out >= X_THRESHOLD));
        end
        cover (past_valid && rst_n && spike_out);
    end
`endif
endmodule

`default_nettype wire
