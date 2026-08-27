// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — bounded safety harness for Wang NMDA autapse Q16.16

module sc_nmda_autapse_formal(
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] current_t
);
    localparam signed [31:0] V_REST = -32'sd4587520;
    localparam signed [31:0] V_RESET = -32'sd3866624;
    localparam signed [31:0] V_MIN = -32'sd7864320;
    localparam signed [31:0] V_MAX = 32'sd5242880;
    localparam signed [31:0] ONE = 32'sd65536;
    localparam signed [31:0] REF_PERIOD = 32'sd131072;
    wire signed [31:0] v;
    wire signed [31:0] x_nmda;
    wire signed [31:0] s_nmda;
    wire signed [31:0] ca;
    wire signed [31:0] refractory;
    wire event_out;

    sc_nmda_autapse uut(
        .clk(clk), .rst_n(rst_n), .current_t(current_t), .v_out(v),
        .x_nmda_out(x_nmda), .s_nmda_out(s_nmda), .ca_out(ca),
        .refractory_out(refractory), .event_out(event_out)
    );

    reg past_valid = 1'b0;
    initial assume(!rst_n);
    always @(posedge clk) begin
        assume(current_t >= -32'sd65536 && current_t <= 32'sd196608);
        if (!past_valid) assume(!rst_n); else assume(rst_n);
        past_valid <= 1'b1;
        if (past_valid && $past(!rst_n)) begin
            assert(v == V_REST && x_nmda == 0 && s_nmda == 0 && ca == 0 &&
                   refractory == 0 && !event_out);
        end
        if (past_valid && rst_n) begin
            assert(v >= V_MIN && v <= V_MAX);
            assert(x_nmda >= 0 && s_nmda >= 0 && s_nmda <= ONE && ca >= 0);
            assert(refractory >= 0 && refractory <= REF_PERIOD);
            if (event_out) assert(v == V_RESET && refractory == REF_PERIOD);
            if (refractory > 0) assert(v == V_RESET);
        end
        cover(past_valid && rst_n && event_out);
    end
endmodule
