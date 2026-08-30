// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal safety for Latham QIF Q16.16 representative

`default_nettype none

module sc_quadratic_if_latham_2000_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t
);
    wire spike_out;
    wire signed [31:0] v_out;

    sc_quadratic_if_latham_2000 uut (
        .clk(clk), .rst_n(rst_n), .I_t(I_t),
        .spike_out(spike_out), .v_out(v_out)
    );

`ifdef FORMAL
    reg past_valid = 1'b0;
    always @(posedge clk)
        past_valid <= 1'b1;

    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end

    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            assert ($signed(v_out) >= -32'sd2147483648);
            assert ($signed(v_out) <= 32'sd2147483647);
            if (spike_out)
                assert (v_out == -32'sd196608);
        end
    end
`endif
endmodule
