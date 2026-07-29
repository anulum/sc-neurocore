// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — formal harness for Nagumo–Sato Q16.16 RTL

`default_nettype none

module sc_nagumo_sato_map_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t
);
    wire spike_out;
    wire signed [31:0] y_out;

    sc_nagumo_sato_map uut (
        .clk(clk), .rst_n(rst_n), .I_t(I_t), .spike_out(spike_out), .y_out(y_out)
    );

`ifdef FORMAL
    reg past_valid = 1'b0;
    reg active_since_reset = 1'b0;
    initial assume (!rst_n);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            active_since_reset <= 1'b0;
        else
            active_since_reset <= 1'b1;
    end
    always @(posedge clk) begin
        if (!past_valid)
            assume (!rst_n);
        else
            assume (rst_n);
        past_valid <= 1'b1;
        if (!rst_n)
            assert (spike_out == 1'b0);
        if (past_valid && active_since_reset && $past(active_since_reset)) begin
            assert (spike_out == !y_out[31]);
        end
        cover (past_valid && rst_n && spike_out);
        cover (past_valid && rst_n && !spike_out);
    end
`endif
endmodule

`default_nettype wire
