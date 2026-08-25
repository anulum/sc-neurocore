// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — bounded safety harness for Brunel-Wang Q16.16

`default_nettype none
module sc_brunel_wang_formal(
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] s_ampa_ext_t,
    input wire signed [31:0] s_ampa_rec_t,
    input wire signed [31:0] s_nmda_rec_t,
    input wire signed [31:0] s_gaba_t
);
    localparam signed [31:0] V_REST = -32'sd4587520;
    localparam signed [31:0] V_RESET = -32'sd3604480;
    localparam signed [31:0] V_MIN = -32'sd5242880;
    localparam signed [31:0] V_MAX = 32'sd0;
    localparam signed [31:0] GATE_MAX = 32'sd32768;
    localparam signed [31:0] REF_PERIOD = 32'sd131072;
    wire signed [31:0] v;
    wire signed [31:0] refractory;
    wire event_out;
    sc_brunel_wang uut(
        .clk(clk), .rst_n(rst_n), .s_ampa_ext_t(s_ampa_ext_t),
        .s_ampa_rec_t(s_ampa_rec_t), .s_nmda_rec_t(s_nmda_rec_t),
        .s_gaba_t(s_gaba_t), .v_out(v), .refractory_out(refractory),
        .event_out(event_out)
    );
`ifdef FORMAL
    reg past_valid = 1'b0;
    initial assume(!rst_n);
    always @(posedge clk) begin
        assume(s_ampa_ext_t >= 0 && s_ampa_ext_t <= GATE_MAX);
        assume(s_ampa_rec_t >= 0 && s_ampa_rec_t <= GATE_MAX);
        assume(s_nmda_rec_t >= 0 && s_nmda_rec_t <= GATE_MAX);
        assume(s_gaba_t >= 0 && s_gaba_t <= GATE_MAX);
        if (!past_valid) assume(!rst_n); else assume(rst_n);
        past_valid <= 1'b1;
        if (past_valid && $past(!rst_n)) begin
            assert(v == V_REST && refractory == 0 && !event_out);
        end
        if (past_valid && rst_n) begin
            assert(v >= V_MIN && v <= V_MAX);
            assert(refractory >= 0 && refractory <= REF_PERIOD);
            if (event_out) assert(v == V_RESET && refractory == REF_PERIOD);
            if (refractory > 0) assert(v == V_RESET);
        end
        cover(past_valid && rst_n && event_out);
    end
`endif
endmodule
`default_nettype wire
