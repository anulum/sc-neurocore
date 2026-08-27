// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained WB plus NMDA recurrence in signed Q32.32

// A start pulse begins one 0.5 ms macro-step. The datapath executes the
// retained recurrence's 50 Euler substeps over 50 clocks and pulses ready when
// the complete state and macro-step event are committed. Voltage-dependent
// rates and magnesium block are linearly interpolated between 5 mV Q32.32 LUT
// samples. This source-default profile does not claim configurable parameters,
// binary64 equivalence, timing, PPA, device implementation, or silicon proof.

module sc_wb_nmda_magnesium_block (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire signed [63:0] current_t,
    output reg signed [63:0] v_out,
    output reg signed [63:0] h_out,
    output reg signed [63:0] n_out,
    output reg signed [63:0] s_nmda_out,
    output reg event_out,
    output reg ready,
    output reg busy
);
    localparam signed [63:0] ONE = 64'sd4294967296;
    localparam signed [63:0] V_REST = -64'sd279172874240;
    localparam signed [63:0] V_THRESHOLD = -64'sd85899345920;
    localparam signed [63:0] V_MIN = -64'sd429496729600;
    localparam signed [63:0] V_MAX = 64'sd257698037760;
    localparam signed [63:0] H_INIT = 64'sd2576980378;
    localparam signed [63:0] N_INIT = 64'sd1374389535;
    localparam signed [63:0] SUB_DT = 64'sd42949673;
    localparam signed [63:0] PHI = 64'sd21474836480;
    localparam signed [63:0] G_NA = 64'sd150323855360;
    localparam signed [63:0] G_K = 64'sd38654705664;
    localparam signed [63:0] G_NMDA = 64'sd2147483648;
    localparam signed [63:0] G_L = 64'sd429496730;
    localparam signed [63:0] E_NA = 64'sd236223201280;
    localparam signed [63:0] E_K = -64'sd386547056640;
    localparam signed [63:0] DRIVE_STEP = 64'sd2147483648;
    localparam signed [63:0] DRIVE_MAX = 64'sd85899345920;
    localparam signed [63:0] RISE_STEP = 64'sd214748365;
    localparam signed [63:0] DECAY_STEP = 64'sd21474836;
    localparam signed [63:0] LUT_STEP = 64'sd21474836480;

    reg signed [63:0] v_reg;
    reg signed [63:0] h_reg;
    reg signed [63:0] n_reg;
    reg signed [63:0] s_reg;
    reg signed [63:0] current_reg;
    reg [5:0] substep_count;
    reg event_accum;

    function automatic signed [63:0] qmul(
        input signed [63:0] left,
        input signed [63:0] right
    );
        reg signed [127:0] product;
        begin
            product = $signed(left) * $signed(right);
            qmul = product >>> 32;
        end
    endfunction

    function automatic signed [63:0] drive_sample(input [5:0] index);
        begin
            case (index)
                6'd0: drive_sample = 64'sd0;
                6'd1: drive_sample = 64'sd390451572;
                6'd2: drive_sample = 64'sd715827883;
                6'd3: drive_sample = 64'sd991146299;
                6'd4: drive_sample = 64'sd1227133513;
                6'd5: drive_sample = 64'sd1431655765;
                6'd6: drive_sample = 64'sd1610612736;
                6'd7: drive_sample = 64'sd1768515945;
                6'd8: drive_sample = 64'sd1908874354;
                6'd9: drive_sample = 64'sd2034458193;
                6'd10: drive_sample = 64'sd2147483648;
                6'd11: drive_sample = 64'sd2249744774;
                6'd12: drive_sample = 64'sd2342709434;
                6'd13: drive_sample = 64'sd2427590211;
                6'd14: drive_sample = 64'sd2505397589;
                6'd15: drive_sample = 64'sd2576980378;
                6'd16: drive_sample = 64'sd2643056798;
                6'd17: drive_sample = 64'sd2704238668;
                6'd18: drive_sample = 64'sd2761050405;
                6'd19: drive_sample = 64'sd2813944090;
                6'd20: drive_sample = 64'sd2863311531;
                6'd21: drive_sample = 64'sd2909493975;
                6'd22: drive_sample = 64'sd2952790016;
                6'd23: drive_sample = 64'sd2993462055;
                6'd24: drive_sample = 64'sd3031741621;
                6'd25: drive_sample = 64'sd3067833783;
                6'd26: drive_sample = 64'sd3101920825;
                6'd27: drive_sample = 64'sd3134165324;
                6'd28: drive_sample = 64'sd3164712744;
                6'd29: drive_sample = 64'sd3193693630;
                6'd30: drive_sample = 64'sd3221225472;
                6'd31: drive_sample = 64'sd3247414297;
                6'd32: drive_sample = 64'sd3272356035;
                6'd33: drive_sample = 64'sd3296137692;
                6'd34: drive_sample = 64'sd3318838365;
                6'd35: drive_sample = 64'sd3340530119;
                6'd36: drive_sample = 64'sd3361278753;
                6'd37: drive_sample = 64'sd3381144467;
                6'd38: drive_sample = 64'sd3400182443;
                6'd39: drive_sample = 64'sd3418443358;
                default: drive_sample = 64'sd3435973837;
            endcase
        end
    endfunction

    function automatic signed [63:0] drive_lut(input signed [63:0] current);
        reg signed [127:0] remainder;
        reg signed [127:0] interpolation;
        reg signed [63:0] lower;
        reg signed [63:0] upper;
        integer index;
        begin
            if (current <= 0) begin
                drive_lut = 0;
            end else if (current >= DRIVE_MAX) begin
                drive_lut = drive_sample(40);
            end else begin
                index = current / DRIVE_STEP;
                remainder = current - index * DRIVE_STEP;
                lower = drive_sample(index[5:0]);
                upper = drive_sample(index[5:0] + 1'b1);
                interpolation =
                    ($signed(upper - lower) * $signed(remainder)) / DRIVE_STEP;
                drive_lut = lower + interpolation[63:0];
            end
        end
    endfunction

    function automatic signed [63:0] rate_sample(
        input [2:0] kind,
        input [5:0] index
    );
        begin
            case ({kind, index})
                9'd0: rate_sample = 64'sd1138514;
                9'd1: rate_sample = 64'sd2289123;
                9'd2: rate_sample = 64'sd4572300;
                9'd3: rate_sample = 64'sd9062114;
                9'd4: rate_sample = 64'sd17794521;
                9'd5: rate_sample = 64'sd34545441;
                9'd6: rate_sample = 64'sd66106279;
                9'd7: rate_sample = 64'sd124148325;
                9'd8: rate_sample = 64'sd227343293;
                9'd9: rate_sample = 64'sd402189118;
                9'd10: rate_sample = 64'sd678829842;
                9'd11: rate_sample = 64'sd1077229674;
                9'd12: rate_sample = 64'sd1585773997;
                9'd13: rate_sample = 64'sd2150269499;
                9'd14: rate_sample = 64'sd2693556303;
                9'd15: rate_sample = 64'sd3154025260;
                9'd16: rate_sample = 64'sd3507524731;
                9'd17: rate_sample = 64'sd3761075142;
                9'd18: rate_sample = 64'sd3935583857;
                9'd19: rate_sample = 64'sd4053121586;
                9'd20: rate_sample = 64'sd4131607174;
                9'd21: rate_sample = 64'sd4183979360;
                9'd22: rate_sample = 64'sd4219059125;
                9'd23: rate_sample = 64'sd4242699313;
                9'd24: rate_sample = 64'sd4258742786;
                9'd25: rate_sample = 64'sd4269709574;
                9'd26: rate_sample = 64'sd4277258611;
                9'd27: rate_sample = 64'sd4282489087;
                9'd28: rate_sample = 64'sd4286134928;
                9'd29: rate_sample = 64'sd4288690117;
                9'd30: rate_sample = 64'sd4290489774;
                9'd31: rate_sample = 64'sd4291762943;
                9'd32: rate_sample = 64'sd4292667260;
                9'd64: rate_sample = 64'sd2455140290;
                9'd65: rate_sample = 64'sd1912065180;
                9'd66: rate_sample = 64'sd1489117860;
                9'd67: rate_sample = 64'sd1159726155;
                9'd68: rate_sample = 64'sd903195638;
                9'd69: rate_sample = 64'sd703409470;
                9'd70: rate_sample = 64'sd547815846;
                9'd71: rate_sample = 64'sd426639410;
                9'd72: rate_sample = 64'sd332267106;
                9'd73: rate_sample = 64'sd258769883;
                9'd74: rate_sample = 64'sd201530187;
                9'd75: rate_sample = 64'sd156951868;
                9'd76: rate_sample = 64'sd122234237;
                9'd77: rate_sample = 64'sd95196120;
                9'd78: rate_sample = 64'sd74138813;
                9'd79: rate_sample = 64'sd57739365;
                9'd80: rate_sample = 64'sd44967463;
                9'd81: rate_sample = 64'sd35020695;
                9'd82: rate_sample = 64'sd27274145;
                9'd83: rate_sample = 64'sd21241125;
                9'd84: rate_sample = 64'sd16542605;
                9'd85: rate_sample = 64'sd12883394;
                9'd86: rate_sample = 64'sd10033597;
                9'd87: rate_sample = 64'sd7814173;
                9'd88: rate_sample = 64'sd6085684;
                9'd89: rate_sample = 64'sd4739536;
                9'd90: rate_sample = 64'sd3691154;
                9'd91: rate_sample = 64'sd2874674;
                9'd92: rate_sample = 64'sd2238798;
                9'd93: rate_sample = 64'sd1743578;
                9'd94: rate_sample = 64'sd1357900;
                9'd95: rate_sample = 64'sd1057533;
                9'd96: rate_sample = 64'sd823608;
                9'd128: rate_sample = 64'sd3204169;
                9'd129: rate_sample = 64'sd5280227;
                9'd130: rate_sample = 64'sd8698685;
                9'd131: rate_sample = 64'sd14322888;
                9'd132: rate_sample = 64'sd23563474;
                9'd133: rate_sample = 64'sd38711823;
                9'd134: rate_sample = 64'sd63453983;
                9'd135: rate_sample = 64'sd103624768;
                9'd136: rate_sample = 64'sd168215499;
                9'd137: rate_sample = 64'sd270468505;
                9'd138: rate_sample = 64'sd428425089;
                9'd139: rate_sample = 64'sd663423262;
                9'd140: rate_sample = 64'sd994178485;
                9'd141: rate_sample = 64'sd1425122667;
                9'd142: rate_sample = 64'sd1933448259;
                9'd143: rate_sample = 64'sd2467211823;
                9'd144: rate_sample = 64'sd2963417832;
                9'd145: rate_sample = 64'sd3375135552;
                9'd146: rate_sample = 64'sd3685721611;
                9'd147: rate_sample = 64'sd3903597159;
                9'd148: rate_sample = 64'sd4048761835;
                9'd149: rate_sample = 64'sd4142190201;
                9'd150: rate_sample = 64'sd4200987953;
                9'd151: rate_sample = 64'sd4237470922;
                9'd152: rate_sample = 64'sd4259909320;
                9'd153: rate_sample = 64'sd4273635045;
                9'd154: rate_sample = 64'sd4282003297;
                9'd155: rate_sample = 64'sd4287094883;
                9'd156: rate_sample = 64'sd4290188990;
                9'd157: rate_sample = 64'sd4292067838;
                9'd158: rate_sample = 64'sd4293208218;
                9'd159: rate_sample = 64'sd4293900190;
                9'd160: rate_sample = 64'sd4294320000;
                9'd192: rate_sample = 64'sd3861459;
                9'd193: rate_sample = 64'sd5889366;
                9'd194: rate_sample = 64'sd8927045;
                9'd195: rate_sample = 64'sd13436435;
                9'd196: rate_sample = 64'sd20060910;
                9'd197: rate_sample = 64'sd29675228;
                9'd198: rate_sample = 64'sd43434408;
                9'd199: rate_sample = 64'sd62809837;
                9'd200: rate_sample = 64'sd89595232;
                9'd201: rate_sample = 64'sd125861254;
                9'd202: rate_sample = 64'sd173839902;
                9'd203: rate_sample = 64'sd235732168;
                9'd204: rate_sample = 64'sd313455960;
                9'd205: rate_sample = 64'sd408379747;
                9'd206: rate_sample = 64'sd521107486;
                9'd207: rate_sample = 64'sd651377307;
                9'd208: rate_sample = 64'sd798105918;
                9'd209: rate_sample = 64'sd959564528;
                9'd210: rate_sample = 64'sd1133633018;
                9'd211: rate_sample = 64'sd1318064679;
                9'd212: rate_sample = 64'sd1510706083;
                9'd213: rate_sample = 64'sd1709643702;
                9'd214: rate_sample = 64'sd1913275544;
                9'd215: rate_sample = 64'sd2120323137;
                9'd216: rate_sample = 64'sd2329805093;
                9'd217: rate_sample = 64'sd2540991611;
                9'd218: rate_sample = 64'sd2753353925;
                9'd219: rate_sample = 64'sd2966517047;
                9'd220: rate_sample = 64'sd3180219717;
                9'd221: rate_sample = 64'sd3394282572;
                9'd222: rate_sample = 64'sd3608583981;
                9'd223: rate_sample = 64'sd3823042314;
                9'd224: rate_sample = 64'sd4037603265;
                9'd256: rate_sample = 64'sd1081125253;
                9'd257: rate_sample = 64'sd1015623185;
                9'd258: rate_sample = 64'sd954089687;
                9'd259: rate_sample = 64'sd896284315;
                9'd260: rate_sample = 64'sd841981193;
                9'd261: rate_sample = 64'sd790968132;
                9'd262: rate_sample = 64'sd743045795;
                9'd263: rate_sample = 64'sd698026926;
                9'd264: rate_sample = 64'sd655735613;
                9'd265: rate_sample = 64'sd616006600;
                9'd266: rate_sample = 64'sd578684647;
                9'd267: rate_sample = 64'sd543623917;
                9'd268: rate_sample = 64'sd510687409;
                9'd269: rate_sample = 64'sd479746423;
                9'd270: rate_sample = 64'sd450680056;
                9'd271: rate_sample = 64'sd423374732;
                9'd272: rate_sample = 64'sd397723754;
                9'd273: rate_sample = 64'sd373626890;
                9'd274: rate_sample = 64'sd350989981;
                9'd275: rate_sample = 64'sd329724573;
                9'd276: rate_sample = 64'sd309747571;
                9'd277: rate_sample = 64'sd290980914;
                9'd278: rate_sample = 64'sd273351272;
                9'd279: rate_sample = 64'sd256789756;
                9'd280: rate_sample = 64'sd241231651;
                9'd281: rate_sample = 64'sd226616164;
                9'd282: rate_sample = 64'sd212886185;
                9'd283: rate_sample = 64'sd199988063;
                9'd284: rate_sample = 64'sd187871399;
                9'd285: rate_sample = 64'sd176488846;
                9'd286: rate_sample = 64'sd165795927;
                9'd287: rate_sample = 64'sd155750860;
                9'd288: rate_sample = 64'sd146314392;
                9'd320: rate_sample = 64'sd30893502;
                9'd321: rate_sample = 64'sd42011155;
                9'd322: rate_sample = 64'sd57076167;
                9'd323: rate_sample = 64'sd77445052;
                9'd324: rate_sample = 64'sd104903086;
                9'd325: rate_sample = 64'sd141769084;
                9'd326: rate_sample = 64'sd191000289;
                9'd327: rate_sample = 64'sd256272768;
                9'd328: rate_sample = 64'sd341992650;
                9'd329: rate_sample = 64'sd453167506;
                9'd330: rate_sample = 64'sd595042775;
                9'd331: rate_sample = 64'sd772405317;
                9'd332: rate_sample = 64'sd988509565;
                9'd333: rate_sample = 64'sd1243728097;
                9'd334: rate_sample = 64'sd1534264269;
                9'd335: rate_sample = 64'sd1851486863;
                9'd336: rate_sample = 64'sd2182447600;
                9'd337: rate_sample = 64'sd2511755294;
                9'd338: rate_sample = 64'sd2824320589;
                9'd339: rate_sample = 64'sd3107989458;
                9'd340: rate_sample = 64'sd3355149507;
                9'd341: rate_sample = 64'sd3562965549;
                9'd342: rate_sample = 64'sd3732531800;
                9'd343: rate_sample = 64'sd3867530715;
                9'd344: rate_sample = 64'sd3972922202;
                9'd345: rate_sample = 64'sd4053947111;
                9'd346: rate_sample = 64'sd4115507391;
                9'd347: rate_sample = 64'sd4161860486;
                9'd348: rate_sample = 64'sd4196527315;
                9'd349: rate_sample = 64'sd4222322981;
                9'd350: rate_sample = 64'sd4241445244;
                9'd351: rate_sample = 64'sd4255580881;
                9'd352: rate_sample = 64'sd4266008664;
                default: rate_sample = 64'sd0;
            endcase
        end
    endfunction

    function automatic signed [63:0] rate_lut(
        input [2:0] kind,
        input signed [63:0] voltage
    );
        reg signed [127:0] shifted;
        reg signed [127:0] remainder;
        reg signed [127:0] interpolation;
        reg signed [63:0] lower;
        reg signed [63:0] upper;
        integer index;
        begin
            if (voltage <= V_MIN) begin
                rate_lut = rate_sample(kind, 0);
            end else if (voltage >= V_MAX) begin
                rate_lut = rate_sample(kind, 32);
            end else begin
                shifted = $signed(voltage) - $signed(V_MIN);
                index = shifted / LUT_STEP;
                remainder = shifted - index * LUT_STEP;
                lower = rate_sample(kind, index[5:0]);
                upper = rate_sample(kind, index[5:0] + 1'b1);
                interpolation =
                    ($signed(upper - lower) * $signed(remainder)) / LUT_STEP;
                rate_lut = lower + interpolation[63:0];
            end
        end
    endfunction

    function automatic signed [63:0] bound_gate(input signed [63:0] value);
        begin
            if (value < 0) bound_gate = 64'sd0;
            else if (value > ONE) bound_gate = ONE;
            else bound_gate = value;
        end
    endfunction

    function automatic signed [63:0] bound_voltage(input signed [63:0] value);
        begin
            if (value < V_MIN) bound_voltage = V_MIN;
            else if (value > V_MAX) bound_voltage = V_MAX;
            else bound_voltage = value;
        end
    endfunction

    wire signed [63:0] drive_start = drive_lut(current_t);
    wire signed [63:0] gate_step = drive_start > s_reg ? RISE_STEP : DECAY_STEP;
    wire signed [63:0] s_start =
        bound_gate(s_reg + qmul(gate_step, drive_start - s_reg));

    wire signed [63:0] m_inf = rate_lut(0, v_reg);
    wire signed [63:0] alpha_h = rate_lut(1, v_reg);
    wire signed [63:0] beta_h = rate_lut(2, v_reg);
    wire signed [63:0] alpha_n = rate_lut(3, v_reg);
    wire signed [63:0] beta_n = rate_lut(4, v_reg);
    wire signed [63:0] mg_block = rate_lut(5, v_reg);
    wire signed [63:0] dh = qmul(
        PHI, qmul(alpha_h, ONE - h_reg) - qmul(beta_h, h_reg)
    );
    wire signed [63:0] dn = qmul(
        PHI, qmul(alpha_n, ONE - n_reg) - qmul(beta_n, n_reg)
    );
    wire signed [63:0] h_next = h_reg + qmul(SUB_DT, dh);
    wire signed [63:0] n_next = n_reg + qmul(SUB_DT, dn);
    wire signed [63:0] m_squared = qmul(m_inf, m_inf);
    wire signed [63:0] m_cubed = qmul(m_squared, m_inf);
    wire signed [63:0] n_squared = qmul(n_next, n_next);
    wire signed [63:0] n_fourth = qmul(n_squared, n_squared);
    wire signed [63:0] i_na =
        qmul(qmul(G_NA, qmul(m_cubed, h_next)), v_reg - E_NA);
    wire signed [63:0] i_k =
        qmul(qmul(G_K, n_fourth), v_reg - E_K);
    wire signed [63:0] i_nmda =
        qmul(qmul(qmul(G_NMDA, s_reg), mg_block), v_reg);
    wire signed [63:0] i_l = qmul(G_L, v_reg - V_REST);
    wire signed [63:0] v_candidate =
        v_reg + qmul(SUB_DT, -i_na - i_k - i_nmda - i_l + current_reg);
    wire substep_event = v_candidate >= V_THRESHOLD;
    wire signed [63:0] v_next = substep_event ? V_REST : v_candidate;

    always @(posedge clk) begin
        if (!rst_n) begin
            v_reg <= V_REST;
            h_reg <= H_INIT;
            n_reg <= N_INIT;
            s_reg <= 0;
            current_reg <= 0;
            substep_count <= 0;
            event_accum <= 0;
            v_out <= V_REST;
            h_out <= H_INIT;
            n_out <= N_INIT;
            s_nmda_out <= 0;
            event_out <= 0;
            ready <= 0;
            busy <= 0;
        end else begin
            ready <= 0;
            event_out <= 0;
            if (start && !busy) begin
                current_reg <= current_t;
                s_reg <= s_start;
                substep_count <= 0;
                event_accum <= 0;
                busy <= 1;
            end else if (busy) begin
                v_reg <= v_next;
                h_reg <= h_next;
                n_reg <= n_next;
                event_accum <= event_accum | substep_event;
                if (substep_count == 6'd49) begin
                    v_reg <= bound_voltage(v_next);
                    h_reg <= bound_gate(h_next);
                    n_reg <= bound_gate(n_next);
                    v_out <= bound_voltage(v_next);
                    h_out <= bound_gate(h_next);
                    n_out <= bound_gate(n_next);
                    s_nmda_out <= s_reg;
                    event_out <= event_accum | substep_event;
                    ready <= 1;
                    busy <= 0;
                end else begin
                    substep_count <= substep_count + 1'b1;
                end
            end
        end
    end
endmodule
