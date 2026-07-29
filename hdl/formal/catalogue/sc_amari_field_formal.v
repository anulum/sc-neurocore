// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — bounded safety harness for the four-site Amari field

`default_nettype none
module sc_amari_field_formal(
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I0_t,
    input wire signed [31:0] I1_t,
    input wire signed [31:0] I2_t,
    input wire signed [31:0] I3_t
);
    localparam signed [31:0] Q_LIMIT = 32'sd524288;
    wire signed [31:0] u0, u1, u2, u3;
    wire [31:0] rate;
    sc_amari_field uut(
        .clk(clk), .rst_n(rst_n), .I0_t(I0_t), .I1_t(I1_t), .I2_t(I2_t), .I3_t(I3_t),
        .u0_out(u0), .u1_out(u1), .u2_out(u2), .u3_out(u3), .mean_rate_out(rate)
    );
`ifdef FORMAL
    reg past_valid = 1'b0;
    initial assume(!rst_n);
    always @(posedge clk) begin
        if (!past_valid) assume(!rst_n); else assume(rst_n);
        past_valid <= 1'b1;
        if (past_valid && $past(!rst_n)) begin
            assert(u0 == 0 && u1 == 0 && u2 == 0 && u3 == 0 && rate == 0);
        end
        if (past_valid && rst_n) begin
            assert(u0 >= -Q_LIMIT && u0 <= Q_LIMIT);
            assert(u1 >= -Q_LIMIT && u1 <= Q_LIMIT);
            assert(u2 >= -Q_LIMIT && u2 <= Q_LIMIT);
            assert(u3 >= -Q_LIMIT && u3 <= Q_LIMIT);
            assert(rate == 0 || rate == 32'd16384 || rate == 32'd32768 ||
                   rate == 32'd49152 || rate == 32'd65536);
        end
        cover(past_valid && rst_n && rate != 0 && rate != 32'd65536);
    end
`endif
endmodule
`default_nettype wire
