// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wang NMDA-autapse source profile in signed Q16.16

// One rising edge advances one 0.05 ms midpoint-RK2 step at the source
// defaults. current_t is nA and all state outputs are signed Q16.16. The
// Jahr-Stevens magnesium factor is linearly interpolated between 5 mV LUT
// samples. Calcium is retained as an output state even though the enrolled
// source profile has g_ahp=0. Arithmetic wraps only outside the formally
// enrolled current/state envelope. This module does not claim binary64
// equivalence, interpolated spike timing, a full recurrent network, timing,
// PPA, device implementation, or silicon validation.

module sc_nmda_autapse (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] current_t,
    output reg signed [31:0] v_out,
    output reg signed [31:0] x_nmda_out,
    output reg signed [31:0] s_nmda_out,
    output reg signed [31:0] ca_out,
    output reg signed [31:0] refractory_out,
    output reg event_out
);
    localparam signed [31:0] ONE = 32'sd65536;
    localparam signed [31:0] V_REST = -32'sd4587520;
    localparam signed [31:0] V_RESET = -32'sd3866624;
    localparam signed [31:0] V_THRESHOLD = -32'sd3407872;
    localparam signed [31:0] V_MIN = -32'sd7864320;
    localparam signed [31:0] V_MAX = 32'sd5242880;
    localparam signed [31:0] DT = 32'sd3277;
    localparam signed [31:0] HALF_DT = 32'sd1638;
    localparam signed [31:0] G_L = 32'sd1638;
    localparam signed [31:0] G_NMDA = 32'sd6554;
    localparam signed [31:0] INV_C_M = 32'sd131072;
    localparam signed [31:0] INV_TAU_X = 32'sd32768;
    localparam signed [31:0] INV_TAU_S = 32'sd819;
    localparam signed [31:0] INV_TAU_CA = 32'sd819;
    localparam signed [31:0] ALPHA_X = 32'sd65536;
    localparam signed [31:0] ALPHA_CA = 32'sd13107;
    localparam signed [31:0] REF_PERIOD = 32'sd131072;
    localparam signed [31:0] LUT_STEP = 32'sd327680;

    reg signed [31:0] v_reg;
    reg signed [31:0] x_nmda_reg;
    reg signed [31:0] s_nmda_reg;
    reg signed [31:0] ca_reg;
    reg signed [31:0] refractory_reg;

    function automatic signed [31:0] mg_sample(input [5:0] index);
        begin
            case (index)
                6'd0: mg_sample = 32'sd137;
                6'd1: mg_sample = 32'sd187;
                6'd2: mg_sample = 32'sd254;
                6'd3: mg_sample = 32'sd346;
                6'd4: mg_sample = 32'sd471;
                6'd5: mg_sample = 32'sd641;
                6'd6: mg_sample = 32'sd871;
                6'd7: mg_sample = 32'sd1182;
                6'd8: mg_sample = 32'sd1601;
                6'd9: mg_sample = 32'sd2163;
                6'd10: mg_sample = 32'sd2914;
                6'd11: mg_sample = 32'sd3910;
                6'd12: mg_sample = 32'sd5218;
                6'd13: mg_sample = 32'sd6915;
                6'd14: mg_sample = 32'sd9080;
                6'd15: mg_sample = 32'sd11786;
                6'd16: mg_sample = 32'sd15083;
                6'd17: mg_sample = 32'sd18978;
                6'd18: mg_sample = 32'sd23411;
                6'd19: mg_sample = 32'sd28251;
                6'd20: mg_sample = 32'sd33302;
                6'd21: mg_sample = 32'sd38326;
                6'd22: mg_sample = 32'sd43096;
                6'd23: mg_sample = 32'sd47424;
                6'd24: mg_sample = 32'sd51196;
                6'd25: mg_sample = 32'sd54367;
                6'd26: mg_sample = 32'sd56954;
                6'd27: mg_sample = 32'sd59014;
                6'd28: mg_sample = 32'sd60622;
                6'd29: mg_sample = 32'sd61858;
                6'd30: mg_sample = 32'sd62798;
                6'd31: mg_sample = 32'sd63505;
                6'd32: mg_sample = 32'sd64034;
                6'd33: mg_sample = 32'sd64428;
                6'd34: mg_sample = 32'sd64719;
                6'd35: mg_sample = 32'sd64935;
                6'd36: mg_sample = 32'sd65094;
                6'd37: mg_sample = 32'sd65211;
                6'd38: mg_sample = 32'sd65298;
                6'd39: mg_sample = 32'sd65361;
                default: mg_sample = 32'sd65408;
            endcase
        end
    endfunction

    function automatic signed [31:0] mg_block(input signed [31:0] voltage);
        reg signed [63:0] shifted;
        reg signed [63:0] remainder;
        reg signed [63:0] interpolation;
        reg signed [31:0] lower;
        reg signed [31:0] upper;
        integer index;
        begin
            if (voltage <= V_MIN) begin
                mg_block = mg_sample(0);
            end else if (voltage >= V_MAX) begin
                mg_block = mg_sample(40);
            end else begin
                shifted = $signed(voltage) - $signed(V_MIN);
                index = shifted / LUT_STEP;
                remainder = shifted - index * LUT_STEP;
                lower = mg_sample(index[5:0]);
                upper = mg_sample(index[5:0] + 1'b1);
                interpolation = ($signed(upper - lower) * $signed(remainder)) / LUT_STEP;
                mg_block = lower + interpolation[31:0];
            end
        end
    endfunction

    function automatic signed [63:0] dv(
        input signed [31:0] voltage,
        input signed [31:0] nmda,
        input signed [31:0] current
    );
        reg signed [63:0] leak;
        reg signed [63:0] i_nmda;
        reg signed [63:0] total;
        reg signed [31:0] block;
        begin
            leak = (($signed(G_L) * $signed(V_REST - voltage)) >>> 16);
            block = mg_block(voltage);
            i_nmda = (($signed(G_NMDA) * $signed(block)) >>> 16);
            i_nmda = (($signed(i_nmda) * $signed(nmda)) >>> 16);
            i_nmda = (($signed(i_nmda) * $signed(-voltage)) >>> 16);
            total = leak + i_nmda + current;
            dv = (($signed(total) * $signed(INV_C_M)) >>> 16);
        end
    endfunction

    function automatic signed [63:0] d_x(input signed [31:0] value);
        begin d_x = -(($signed(value) * $signed(INV_TAU_X)) >>> 16); end
    endfunction

    function automatic signed [63:0] d_s(
        input signed [31:0] value,
        input signed [31:0] precursor
    );
        reg signed [63:0] opening;
        reg signed [63:0] decay;
        begin
            opening = (($signed(precursor) * $signed(ONE - value)) >>> 16);
            decay = (($signed(value) * $signed(INV_TAU_S)) >>> 16);
            d_s = opening - decay;
        end
    endfunction

    function automatic signed [63:0] d_ca(input signed [31:0] value);
        begin d_ca = -(($signed(value) * $signed(INV_TAU_CA)) >>> 16); end
    endfunction

    wire active = refractory_reg <= 0;
    wire signed [31:0] integration_v = active ? v_reg : V_RESET;
    wire signed [63:0] k1_v = dv(integration_v, s_nmda_reg, current_t);
    wire signed [63:0] k1_x = d_x(x_nmda_reg);
    wire signed [63:0] k1_s = d_s(s_nmda_reg, x_nmda_reg);
    wire signed [63:0] k1_ca = d_ca(ca_reg);
    wire signed [63:0] mid_v_raw = $signed(integration_v) + (($signed(k1_v) * $signed(HALF_DT)) >>> 16);
    wire signed [63:0] mid_x_raw = $signed(x_nmda_reg) + (($signed(k1_x) * $signed(HALF_DT)) >>> 16);
    wire signed [63:0] mid_s_raw = $signed(s_nmda_reg) + (($signed(k1_s) * $signed(HALF_DT)) >>> 16);
    wire signed [63:0] mid_ca_raw = $signed(ca_reg) + (($signed(k1_ca) * $signed(HALF_DT)) >>> 16);
    wire signed [31:0] mid_v = mid_v_raw[31:0];
    wire signed [31:0] mid_x = mid_x_raw[31:0];
    wire signed [31:0] mid_s = mid_s_raw[31:0];
    wire signed [31:0] mid_ca = mid_ca_raw[31:0];
    wire signed [63:0] k2_v = dv(mid_v, mid_s, current_t);
    wire signed [63:0] k2_x = d_x(mid_x);
    wire signed [63:0] k2_s = d_s(mid_s, mid_x);
    wire signed [63:0] k2_ca = d_ca(mid_ca);
    wire signed [63:0] candidate_v = $signed(integration_v) + (($signed(k2_v) * $signed(DT)) >>> 16);
    wire signed [63:0] candidate_x_base = $signed(x_nmda_reg) + (($signed(k2_x) * $signed(DT)) >>> 16);
    wire signed [63:0] candidate_s = $signed(s_nmda_reg) + (($signed(k2_s) * $signed(DT)) >>> 16);
    wire signed [63:0] candidate_ca_base = $signed(ca_reg) + (($signed(k2_ca) * $signed(DT)) >>> 16);
    wire candidate_event = active && candidate_v >= V_THRESHOLD;
    wire signed [63:0] candidate_x = candidate_x_base + (candidate_event ? ALPHA_X : 0);
    wire signed [63:0] candidate_ca = candidate_ca_base + (candidate_event ? ALPHA_CA : 0);
    wire signed [31:0] refractory_next =
        refractory_reg <= DT ? 32'sd0 : refractory_reg - DT;

    function automatic signed [31:0] bound_voltage(input signed [63:0] value);
        begin
            if (value < V_MIN) bound_voltage = V_MIN;
            else if (value > V_MAX) bound_voltage = V_MAX;
            else bound_voltage = value[31:0];
        end
    endfunction

    function automatic signed [31:0] bound_nonnegative(input signed [63:0] value);
        begin bound_nonnegative = value < 0 ? 32'sd0 : value[31:0]; end
    endfunction

    function automatic signed [31:0] bound_gate(input signed [63:0] value);
        begin
            if (value < 0) bound_gate = 32'sd0;
            else if (value > ONE) bound_gate = ONE;
            else bound_gate = value[31:0];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            v_reg <= V_REST;
            x_nmda_reg <= 0;
            s_nmda_reg <= 0;
            ca_reg <= 0;
            refractory_reg <= 0;
            v_out <= V_REST;
            x_nmda_out <= 0;
            s_nmda_out <= 0;
            ca_out <= 0;
            refractory_out <= 0;
            event_out <= 0;
        end else begin
            x_nmda_reg <= bound_nonnegative(candidate_x);
            s_nmda_reg <= bound_gate(candidate_s);
            ca_reg <= bound_nonnegative(candidate_ca);
            x_nmda_out <= bound_nonnegative(candidate_x);
            s_nmda_out <= bound_gate(candidate_s);
            ca_out <= bound_nonnegative(candidate_ca);
            if (!active) begin
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
                v_reg <= bound_voltage(candidate_v);
                refractory_reg <= 0;
                v_out <= bound_voltage(candidate_v);
                refractory_out <= 0;
                event_out <= 0;
            end
        end
    end
endmodule
