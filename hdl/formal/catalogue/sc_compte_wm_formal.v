// SPDX-License-Identifier: AGPL-3.0-or-later
// SC-NeuroCore — bounded safety harness for Compte Q16.16

module sc_compte_wm_formal(
    input wire clk, input wire rst_n, input wire signed [31:0] current_t,
    input wire recurrent_event_t, input wire external_event_t,
    input wire inhibitory_event_t
);
    localparam signed [31:0] V_REST = -32'sd4587520;
    localparam signed [31:0] V_RESET = -32'sd3932160;
    localparam signed [31:0] V_MIN = -32'sd5242880;
    localparam signed [31:0] V_MAX = 32'sd0;
    localparam signed [31:0] REF_PERIOD = 32'sd131072;
    wire signed [31:0] v, s_ampa, s_nmda, x_nmda, s_gaba, refractory;
    wire event_out;
    sc_compte_wm uut(
        .clk(clk), .rst_n(rst_n), .current_t(current_t),
        .recurrent_event_t(recurrent_event_t), .external_event_t(external_event_t),
        .inhibitory_event_t(inhibitory_event_t), .v_out(v),
        .s_ampa_out(s_ampa), .s_nmda_out(s_nmda), .x_nmda_out(x_nmda),
        .s_gaba_out(s_gaba), .refractory_out(refractory), .event_out(event_out)
    );
    reg past_valid = 1'b0;
    initial assume(!rst_n);
    always @(posedge clk) begin
        assume(current_t >= -32'sd65536 && current_t <= 32'sd131072);
        if (!past_valid) assume(!rst_n); else assume(rst_n);
        past_valid <= 1'b1;
        if (past_valid && $past(!rst_n)) begin
            assert(v == V_REST && s_ampa == 0 && s_nmda == 0 &&
                   x_nmda == 0 && s_gaba == 0 && refractory == 0 && !event_out);
        end
        if (past_valid && rst_n) begin
            assert(v >= V_MIN && v <= V_MAX);
            assert(s_ampa >= 0 && s_nmda >= 0 && x_nmda >= 0 && s_gaba >= 0);
            assert(refractory >= 0 && refractory <= REF_PERIOD);
            if (event_out) assert(v == V_RESET && refractory == REF_PERIOD);
            if (refractory > 0) assert(v == V_RESET);
        end
        cover(past_valid && rst_n && event_out);
    end
endmodule
