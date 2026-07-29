// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Brunel-Wang 2001 pyramidal cell in signed Q16.16

// One rising edge advances one 0.1 ms midpoint-RK2 membrane step. The four
// non-negative Q16.16 inputs are already-aggregated channel gates. Voltage and
// refractory outputs are signed Q16.16; event_out is a sampled threshold event.
// The magnesium block uses a one-millivolt LUT over [-80, 0] mV. This finite
// hardware enrolment does not generate presynaptic spikes or integrate synapses.
`timescale 1ns / 1ps

module sc_brunel_wang (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] s_ampa_ext_t,
    input wire signed [31:0] s_ampa_rec_t,
    input wire signed [31:0] s_nmda_rec_t,
    input wire signed [31:0] s_gaba_t,
    output reg signed [31:0] v_out,
    output reg signed [31:0] refractory_out,
    output reg event_out
);
    localparam signed [31:0] V_REST = -32'sd4587520;
    localparam signed [31:0] V_RESET = -32'sd3604480;
    localparam signed [31:0] V_THRESHOLD = -32'sd3276800;
    localparam signed [31:0] V_MIN = -32'sd5242880;
    localparam signed [31:0] V_MAX = 32'sd0;
    localparam signed [31:0] DT = 32'sd6554;
    localparam signed [31:0] HALF_DT = 32'sd3277;
    localparam signed [31:0] INV_TAU_M = 32'sd3277;
    localparam signed [31:0] INV_C_M = 32'sd131072;
    localparam signed [31:0] G_AMPA_EXT = 32'sd136315;
    localparam signed [31:0] G_AMPA_REC = 32'sd6816;
    localparam signed [31:0] G_NMDA = 32'sd21430;
    localparam signed [31:0] G_GABA = 32'sd81920;
    localparam signed [31:0] REF_PERIOD = 32'sd131072;

    reg signed [31:0] v_reg;
    reg signed [31:0] refractory_reg;

    function automatic signed [31:0] mg_block(input signed [31:0] voltage);
        integer mv;
        begin
            mv = voltage >>> 16;
            if (mv < -80) mg_block = 32'sd163;
            else if (mv > 0) mg_block = 32'sd51151;
            else case (mv)
            -80: mg_block = 32'sd1601;
            -79: mg_block = 32'sd1700;
            -78: mg_block = 32'sd1806;
            -77: mg_block = 32'sd1918;
            -76: mg_block = 32'sd2037;
            -75: mg_block = 32'sd2163;
            -74: mg_block = 32'sd2297;
            -73: mg_block = 32'sd2438;
            -72: mg_block = 32'sd2588;
            -71: mg_block = 32'sd2747;
            -70: mg_block = 32'sd2914;
            -69: mg_block = 32'sd3092;
            -68: mg_block = 32'sd3280;
            -67: mg_block = 32'sd3479;
            -66: mg_block = 32'sd3689;
            -65: mg_block = 32'sd3910;
            -64: mg_block = 32'sd4145;
            -63: mg_block = 32'sd4392;
            -62: mg_block = 32'sd4653;
            -61: mg_block = 32'sd4928;
            -60: mg_block = 32'sd5218;
            -59: mg_block = 32'sd5524;
            -58: mg_block = 32'sd5846;
            -57: mg_block = 32'sd6184;
            -56: mg_block = 32'sd6541;
            -55: mg_block = 32'sd6915;
            -54: mg_block = 32'sd7308;
            -53: mg_block = 32'sd7720;
            -52: mg_block = 32'sd8152;
            -51: mg_block = 32'sd8605;
            -50: mg_block = 32'sd9080;
            -49: mg_block = 32'sd9576;
            -48: mg_block = 32'sd10094;
            -47: mg_block = 32'sd10635;
            -46: mg_block = 32'sd11199;
            -45: mg_block = 32'sd11786;
            -44: mg_block = 32'sd12397;
            -43: mg_block = 32'sd13032;
            -42: mg_block = 32'sd13692;
            -41: mg_block = 32'sd14376;
            -40: mg_block = 32'sd15083;
            -39: mg_block = 32'sd15815;
            -38: mg_block = 32'sd16571;
            -37: mg_block = 32'sd17351;
            -36: mg_block = 32'sd18153;
            -35: mg_block = 32'sd18978;
            -34: mg_block = 32'sd19824;
            -33: mg_block = 32'sd20692;
            -32: mg_block = 32'sd21580;
            -31: mg_block = 32'sd22487;
            -30: mg_block = 32'sd23411;
            -29: mg_block = 32'sd24352;
            -28: mg_block = 32'sd25308;
            -27: mg_block = 32'sd26278;
            -26: mg_block = 32'sd27259;
            -25: mg_block = 32'sd28251;
            -24: mg_block = 32'sd29252;
            -23: mg_block = 32'sd30259;
            -22: mg_block = 32'sd31271;
            -21: mg_block = 32'sd32286;
            -20: mg_block = 32'sd33302;
            -19: mg_block = 32'sd34316;
            -18: mg_block = 32'sd35328;
            -17: mg_block = 32'sd36335;
            -16: mg_block = 32'sd37335;
            -15: mg_block = 32'sd38326;
            -14: mg_block = 32'sd39307;
            -13: mg_block = 32'sd40276;
            -12: mg_block = 32'sd41232;
            -11: mg_block = 32'sd42172;
            -10: mg_block = 32'sd43096;
            -9: mg_block = 32'sd44001;
            -8: mg_block = 32'sd44888;
            -7: mg_block = 32'sd45755;
            -6: mg_block = 32'sd46600;
            -5: mg_block = 32'sd47424;
            -4: mg_block = 32'sd48225;
            -3: mg_block = 32'sd49004;
            -2: mg_block = 32'sd49758;
            -1: mg_block = 32'sd50489;
            0: mg_block = 32'sd51196;
                default: mg_block = 32'sd163;
            endcase
        end
    endfunction

    function automatic signed [63:0] derivative(
        input signed [31:0] voltage,
        input signed [31:0] ext_gate,
        input signed [31:0] ampa_gate,
        input signed [31:0] nmda_gate,
        input signed [31:0] gaba_gate
    );
        reg signed [63:0] leak;
        reg signed [63:0] ampa_ext_current;
        reg signed [63:0] ampa_rec_current;
        reg signed [63:0] nmda_current;
        reg signed [63:0] gaba_current;
        reg signed [63:0] synaptic;
        reg signed [31:0] block;
        begin
            leak = (($signed(V_REST - voltage) * $signed(INV_TAU_M)) >>> 16);
            ampa_ext_current = (($signed(G_AMPA_EXT) * $signed(-voltage)) >>> 16);
            ampa_ext_current = (($signed(ampa_ext_current) * $signed(ext_gate)) >>> 16);
            ampa_rec_current = (($signed(G_AMPA_REC) * $signed(-voltage)) >>> 16);
            ampa_rec_current = (($signed(ampa_rec_current) * $signed(ampa_gate)) >>> 16);
            block = mg_block(voltage);
            nmda_current = (($signed(G_NMDA) * $signed(block)) >>> 16);
            nmda_current = (($signed(nmda_current) * $signed(-voltage)) >>> 16);
            nmda_current = (($signed(nmda_current) * $signed(nmda_gate)) >>> 16);
            gaba_current = (($signed(G_GABA) * $signed(V_REST - voltage)) >>> 16);
            gaba_current = (($signed(gaba_current) * $signed(gaba_gate)) >>> 16);
            synaptic = ampa_ext_current + ampa_rec_current + nmda_current + gaba_current;
            derivative = leak + (($signed(synaptic) * $signed(INV_C_M)) >>> 16);
        end
    endfunction

    wire signed [63:0] k1 = derivative(v_reg, s_ampa_ext_t, s_ampa_rec_t, s_nmda_rec_t, s_gaba_t);
    wire signed [63:0] midpoint_raw = $signed(v_reg) + (($signed(k1) * $signed(HALF_DT)) >>> 16);
    wire signed [31:0] midpoint = midpoint_raw[31:0];
    wire signed [63:0] k2 = derivative(midpoint, s_ampa_ext_t, s_ampa_rec_t, s_nmda_rec_t, s_gaba_t);
    wire signed [63:0] candidate_raw = $signed(v_reg) + (($signed(k2) * $signed(DT)) >>> 16);
    wire candidate_event = candidate_raw >= V_THRESHOLD;

    function automatic signed [31:0] bound_voltage(input signed [63:0] value);
        begin
            if (value < V_MIN) bound_voltage = V_MIN;
            else if (value > V_MAX) bound_voltage = V_MAX;
            else bound_voltage = value[31:0];
        end
    endfunction

    wire signed [31:0] refractory_next =
        refractory_reg <= DT ? 32'sd0 : refractory_reg - DT;

    always @(posedge clk) begin
        if (!rst_n) begin
            v_reg <= V_REST;
            refractory_reg <= 0;
            v_out <= V_REST;
            refractory_out <= 0;
            event_out <= 0;
        end else if (refractory_reg > 0) begin
            v_reg <= V_RESET;
            refractory_reg <= refractory_next;
            v_out <= V_RESET;
            refractory_out <= refractory_next;
            event_out <= 0;
        end else if (candidate_event) begin
            v_reg <= V_RESET;
            refractory_reg <= REF_PERIOD;
            v_out <= V_RESET;
            refractory_out <= REF_PERIOD;
            event_out <= 1;
        end else begin
            v_reg <= bound_voltage(candidate_raw);
            refractory_reg <= 0;
            v_out <= bound_voltage(candidate_raw);
            refractory_out <= 0;
            event_out <= 0;
        end
    end
endmodule
