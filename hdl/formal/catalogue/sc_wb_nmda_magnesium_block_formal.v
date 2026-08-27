// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — bounded handshake safety for retained SC NMDA Q32.32

module sc_wb_nmda_magnesium_block_formal(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire signed [63:0] current_t
);
    localparam signed [63:0] ONE = 64'sd4294967296;
    localparam signed [63:0] V_REST = -64'sd279172874240;
    localparam signed [63:0] V_MIN = -64'sd429496729600;
    localparam signed [63:0] V_MAX = 64'sd257698037760;
    localparam signed [63:0] H_INIT = 64'sd2576980378;
    localparam signed [63:0] N_INIT = 64'sd1374389535;
    wire signed [63:0] v;
    wire signed [63:0] h;
    wire signed [63:0] n;
    wire signed [63:0] s_nmda;
    wire event_out;
    wire ready;
    wire busy;

    sc_wb_nmda_magnesium_block uut(
        .clk(clk), .rst_n(rst_n), .start(start), .current_t(current_t),
        .v_out(v), .h_out(h), .n_out(n), .s_nmda_out(s_nmda),
        .event_out(event_out), .ready(ready), .busy(busy)
    );

    reg past_valid = 1'b0;
    initial assume(!rst_n);
    always @(posedge clk) begin
        assume(current_t >= 0 && current_t <= 64'sd21474836480);
        if (!past_valid) assume(!rst_n); else assume(rst_n);
        past_valid <= 1'b1;
        if (past_valid && $past(!rst_n)) begin
            assert(v == V_REST && h == H_INIT && n == N_INIT && s_nmda == 0);
            assert(!event_out && !ready && !busy);
        end
        if (past_valid && rst_n) begin
            assert(v >= V_MIN && v <= V_MAX);
            assert(h >= 0 && h <= ONE && n >= 0 && n <= ONE);
            assert(s_nmda >= 0 && s_nmda <= ONE);
            if (ready) assert(!busy);
            if (event_out) assert(ready && !busy);
        end
    end
endmodule
