// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compte pyramidal cell in signed Q16.16

// One rising edge applies the three presynaptic event jumps and advances one
// 0.02 ms coupled midpoint-RK2 step. current_t is nA in signed Q16.16.
// recurrent_event_t drives the NMDA precursor, external_event_t drives AMPA,
// and inhibitory_event_t drives GABAA. State outputs are signed Q16.16; NMDA
// uses a coarse voltage LUT. Arithmetic wraps only outside the formal/enrolled
// input envelope. This module does not implement the 2560-cell ring, Poisson
// generation, firing-time interpolation, timing, PPA, or binary64 equivalence.

module sc_compte_wm (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] current_t,
    input wire recurrent_event_t,
    input wire external_event_t,
    input wire inhibitory_event_t,
    output reg signed [31:0] v_out,
    output reg signed [31:0] s_ampa_out,
    output reg signed [31:0] s_nmda_out,
    output reg signed [31:0] x_nmda_out,
    output reg signed [31:0] s_gaba_out,
    output reg signed [31:0] refractory_out,
    output reg event_out
);
    localparam signed [31:0] ONE = 32'sd65536;
    localparam signed [31:0] V_REST = -32'sd4587520;
    localparam signed [31:0] V_RESET = -32'sd3932160;
    localparam signed [31:0] V_THRESHOLD = -32'sd3276800;
    localparam signed [31:0] V_MIN = -32'sd5242880;
    localparam signed [31:0] V_MAX = 32'sd0;
    localparam signed [31:0] DT = 32'sd1311;
    localparam signed [31:0] HALF_DT = 32'sd655;
    localparam signed [31:0] G_L = 32'sd1638;
    localparam signed [31:0] G_AMPA = 32'sd203;
    localparam signed [31:0] G_NMDA = 32'sd25;
    localparam signed [31:0] G_GABA = 32'sd88;
    localparam signed [31:0] INV_C_M = 32'sd131072;
    localparam signed [31:0] INV_TAU_AMPA = 32'sd32768;
    localparam signed [31:0] INV_TAU_NMDA = 32'sd655;
    localparam signed [31:0] INV_TAU_X = 32'sd32768;
    localparam signed [31:0] INV_TAU_GABA = 32'sd6554;
    localparam signed [31:0] ALPHA_NMDA = 32'sd32768;
    localparam signed [31:0] REF_PERIOD = 32'sd131072;

    reg signed [31:0] v_reg;
    reg signed [31:0] s_ampa_reg;
    reg signed [31:0] s_nmda_reg;
    reg signed [31:0] x_nmda_reg;
    reg signed [31:0] s_gaba_reg;
    reg signed [31:0] refractory_reg;

    function automatic signed [31:0] mg_block(input signed [31:0] voltage);
        integer mv;
        begin
            mv = voltage >>> 16;
            if (mv <= -80) mg_block = 32'sd1601;
            else if (mv <= -75) mg_block = 32'sd2163;
            else if (mv <= -70) mg_block = 32'sd2914;
            else if (mv <= -65) mg_block = 32'sd3910;
            else if (mv <= -60) mg_block = 32'sd5218;
            else if (mv <= -55) mg_block = 32'sd6915;
            else if (mv <= -50) mg_block = 32'sd9080;
            else if (mv <= -45) mg_block = 32'sd11786;
            else if (mv <= -40) mg_block = 32'sd15083;
            else if (mv <= -35) mg_block = 32'sd18978;
            else if (mv <= -30) mg_block = 32'sd23411;
            else if (mv <= -25) mg_block = 32'sd28251;
            else if (mv <= -20) mg_block = 32'sd33302;
            else if (mv <= -15) mg_block = 32'sd38326;
            else if (mv <= -10) mg_block = 32'sd43096;
            else if (mv <= -5) mg_block = 32'sd47424;
            else mg_block = 32'sd51196;
        end
    endfunction

    function automatic signed [63:0] dv(
        input signed [31:0] voltage,
        input signed [31:0] ampa,
        input signed [31:0] nmda,
        input signed [31:0] gaba,
        input signed [31:0] current
    );
        reg signed [63:0] leak;
        reg signed [63:0] i_ampa;
        reg signed [63:0] i_nmda;
        reg signed [63:0] i_gaba;
        reg signed [63:0] total;
        reg signed [31:0] block;
        begin
            leak = (($signed(G_L) * $signed(V_REST - voltage)) >>> 16);
            i_ampa = (($signed(G_AMPA) * $signed(ampa)) >>> 16);
            i_ampa = (($signed(i_ampa) * $signed(-voltage)) >>> 16);
            block = mg_block(voltage);
            i_nmda = (($signed(G_NMDA) * $signed(block)) >>> 16);
            i_nmda = (($signed(i_nmda) * $signed(nmda)) >>> 16);
            i_nmda = (($signed(i_nmda) * $signed(-voltage)) >>> 16);
            i_gaba = (($signed(G_GABA) * $signed(gaba)) >>> 16);
            i_gaba = (($signed(i_gaba) * $signed(V_REST - voltage)) >>> 16);
            total = leak + i_ampa + i_nmda + i_gaba + current;
            dv = (($signed(total) * $signed(INV_C_M)) >>> 16);
        end
    endfunction

    function automatic signed [63:0] d_ampa(input signed [31:0] value);
        begin d_ampa = -(($signed(value) * $signed(INV_TAU_AMPA)) >>> 16); end
    endfunction
    function automatic signed [63:0] d_x(input signed [31:0] value);
        begin d_x = -(($signed(value) * $signed(INV_TAU_X)) >>> 16); end
    endfunction
    function automatic signed [63:0] d_gaba(input signed [31:0] value);
        begin d_gaba = -(($signed(value) * $signed(INV_TAU_GABA)) >>> 16); end
    endfunction
    function automatic signed [63:0] d_nmda(
        input signed [31:0] value,
        input signed [31:0] precursor
    );
        reg signed [63:0] decay;
        reg signed [63:0] saturation;
        begin
            decay = -(($signed(value) * $signed(INV_TAU_NMDA)) >>> 16);
            saturation = (($signed(ALPHA_NMDA) * $signed(precursor)) >>> 16);
            saturation = (($signed(saturation) * $signed(ONE - value)) >>> 16);
            d_nmda = decay + saturation;
        end
    endfunction

    wire signed [31:0] pre_ampa = s_ampa_reg + (external_event_t ? ONE : 0);
    wire signed [31:0] pre_nmda = s_nmda_reg;
    wire signed [31:0] pre_x = x_nmda_reg + (recurrent_event_t ? ONE : 0);
    wire signed [31:0] pre_gaba = s_gaba_reg + (inhibitory_event_t ? ONE : 0);
    wire active = refractory_reg <= 0;
    wire signed [63:0] k1_v = active ? dv(v_reg, pre_ampa, pre_nmda, pre_gaba, current_t) : 0;
    wire signed [63:0] k1_ampa = d_ampa(pre_ampa);
    wire signed [63:0] k1_nmda = d_nmda(pre_nmda, pre_x);
    wire signed [63:0] k1_x = d_x(pre_x);
    wire signed [63:0] k1_gaba = d_gaba(pre_gaba);
    wire signed [63:0] mid_v_raw = $signed(v_reg) + (($signed(k1_v) * $signed(HALF_DT)) >>> 16);
    wire signed [63:0] mid_ampa_raw = $signed(pre_ampa) + (($signed(k1_ampa) * $signed(HALF_DT)) >>> 16);
    wire signed [63:0] mid_nmda_raw = $signed(pre_nmda) + (($signed(k1_nmda) * $signed(HALF_DT)) >>> 16);
    wire signed [63:0] mid_x_raw = $signed(pre_x) + (($signed(k1_x) * $signed(HALF_DT)) >>> 16);
    wire signed [63:0] mid_gaba_raw = $signed(pre_gaba) + (($signed(k1_gaba) * $signed(HALF_DT)) >>> 16);
    wire signed [31:0] mid_v = mid_v_raw[31:0];
    wire signed [31:0] mid_ampa = mid_ampa_raw[31:0];
    wire signed [31:0] mid_nmda = mid_nmda_raw[31:0];
    wire signed [31:0] mid_x = mid_x_raw[31:0];
    wire signed [31:0] mid_gaba = mid_gaba_raw[31:0];
    wire signed [63:0] k2_v = active ? dv(mid_v, mid_ampa, mid_nmda, mid_gaba, current_t) : 0;
    wire signed [63:0] k2_ampa = d_ampa(mid_ampa);
    wire signed [63:0] k2_nmda = d_nmda(mid_nmda, mid_x);
    wire signed [63:0] k2_x = d_x(mid_x);
    wire signed [63:0] k2_gaba = d_gaba(mid_gaba);
    wire signed [63:0] candidate_v = $signed(v_reg) + (($signed(k2_v) * $signed(DT)) >>> 16);
    wire signed [63:0] candidate_ampa = $signed(pre_ampa) + (($signed(k2_ampa) * $signed(DT)) >>> 16);
    wire signed [63:0] candidate_nmda = $signed(pre_nmda) + (($signed(k2_nmda) * $signed(DT)) >>> 16);
    wire signed [63:0] candidate_x = $signed(pre_x) + (($signed(k2_x) * $signed(DT)) >>> 16);
    wire signed [63:0] candidate_gaba = $signed(pre_gaba) + (($signed(k2_gaba) * $signed(DT)) >>> 16);
    wire candidate_event = active && candidate_v >= V_THRESHOLD;
    wire signed [31:0] refractory_next =
        refractory_reg <= DT ? 32'sd0 : refractory_reg - DT;

    function automatic signed [31:0] bound_voltage(input signed [63:0] value);
        begin
            if (value < V_MIN) bound_voltage = V_MIN;
            else if (value > V_MAX) bound_voltage = V_MAX;
            else bound_voltage = value[31:0];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            v_reg <= V_REST; s_ampa_reg <= 0; s_nmda_reg <= 0;
            x_nmda_reg <= 0; s_gaba_reg <= 0; refractory_reg <= 0;
            v_out <= V_REST; s_ampa_out <= 0; s_nmda_out <= 0;
            x_nmda_out <= 0; s_gaba_out <= 0; refractory_out <= 0; event_out <= 0;
        end else begin
            s_ampa_reg <= candidate_ampa[31:0];
            s_nmda_reg <= candidate_nmda[31:0];
            x_nmda_reg <= candidate_x[31:0];
            s_gaba_reg <= candidate_gaba[31:0];
            s_ampa_out <= candidate_ampa[31:0];
            s_nmda_out <= candidate_nmda[31:0];
            x_nmda_out <= candidate_x[31:0];
            s_gaba_out <= candidate_gaba[31:0];
            if (!active) begin
                v_reg <= V_RESET; refractory_reg <= refractory_next;
                v_out <= V_RESET; refractory_out <= refractory_next; event_out <= 0;
            end else if (candidate_event) begin
                v_reg <= V_RESET; refractory_reg <= REF_PERIOD;
                v_out <= V_RESET; refractory_out <= REF_PERIOD; event_out <= 1;
            end else begin
                v_reg <= bound_voltage(candidate_v); refractory_reg <= 0;
                v_out <= bound_voltage(candidate_v); refractory_out <= 0; event_out <= 0;
            end
        end
    end
endmodule
