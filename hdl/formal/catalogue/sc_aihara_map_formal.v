// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — bounded formal harness for the Aihara map

`default_nettype none

module sc_aihara_map_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t
);
    wire spike_out;
    wire signed [31:0] y_out;

    sc_aihara_map uut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .y_out(y_out)
    );

`ifdef FORMAL
    reg past_valid = 1'b0;
    always @(posedge clk) begin
        past_valid <= 1'b1;
        if (!rst_n)
            assert (spike_out == 1'b0);
        if (past_valid && rst_n && $past(rst_n))
            assert (spike_out == !y_out[31]);
    end
`endif
endmodule

`default_nettype wire
