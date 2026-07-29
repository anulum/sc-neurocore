// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained SC adaptive-threshold-map Q8.24 RTL

// One-cycle simultaneous project recurrence with a 256-entry sigmoid LUT.
// Every public numeric port and parameter uses signed Q8.24.
`timescale 1ns / 1ps

module sc_adaptive_threshold_map #(
    parameter signed [31:0] P_K = 32'sd25165824,
    parameter signed [31:0] P_BETA = 32'sd15938355,
    parameter signed [31:0] P_GAMMA = 32'sd5033165,
    parameter signed [31:0] P_THETA_SPIKE = 32'sd13421773,
    parameter signed [31:0] P_X_THRESHOLD = 32'sd13421773
)(
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] I_t,
    output reg spike_out,
    output reg signed [31:0] x_out,
    output reg signed [31:0] theta_out
);
    localparam signed [31:0] Q_FIVE = 32'sd83886080;
    reg signed [31:0] x_reg;
    reg signed [31:0] theta_reg;

    wire signed [63:0] sigmoid_arg_wide =
        ($signed({{32{x_reg[31]}}, x_reg}) -
         $signed({{32{theta_reg[31]}}, theta_reg})) <<< 2;
    wire signed [31:0] _sigmoid_lut2_arg =
        sigmoid_arg_wide > 64'sd2147483647 ? 32'sd2147483647 :
        sigmoid_arg_wide < -64'sd2147483648 ? -32'sd2147483648 :
        sigmoid_arg_wide[31:0];
// _sigmoid_lut lookup table (256 entries over [-16.0, 16.0), step 0.125)

wire signed [32:0] _sigmoid_lut2_raw = ($signed({{_sigmoid_lut2_arg[31]}, _sigmoid_lut2_arg}) + 33'sd268435456) >>> 21;
// Explicit signed casts are required here: an unsized literal can otherwise
// turn the negative lower-bound comparison unsigned in Verilog-2005.
wire [7:0] _sigmoid_lut2_idx = ($signed(_sigmoid_lut2_raw) < $signed(33'sd0)) ? 8'd0 : (($signed(_sigmoid_lut2_raw) > $signed(33'sd255)) ? 8'd255 : _sigmoid_lut2_raw[7:0]);
reg signed [31:0] _sigmoid_lut2_out;
always @(*) case (_sigmoid_lut2_idx)
    8'd0: _sigmoid_lut2_out = 32'sd2;
    8'd1: _sigmoid_lut2_out = 32'sd2;
    8'd2: _sigmoid_lut2_out = 32'sd2;
    8'd3: _sigmoid_lut2_out = 32'sd3;
    8'd4: _sigmoid_lut2_out = 32'sd3;
    8'd5: _sigmoid_lut2_out = 32'sd4;
    8'd6: _sigmoid_lut2_out = 32'sd4;
    8'd7: _sigmoid_lut2_out = 32'sd5;
    8'd8: _sigmoid_lut2_out = 32'sd5;
    8'd9: _sigmoid_lut2_out = 32'sd6;
    8'd10: _sigmoid_lut2_out = 32'sd7;
    8'd11: _sigmoid_lut2_out = 32'sd7;
    8'd12: _sigmoid_lut2_out = 32'sd8;
    8'd13: _sigmoid_lut2_out = 32'sd10;
    8'd14: _sigmoid_lut2_out = 32'sd11;
    8'd15: _sigmoid_lut2_out = 32'sd12;
    8'd16: _sigmoid_lut2_out = 32'sd14;
    8'd17: _sigmoid_lut2_out = 32'sd16;
    8'd18: _sigmoid_lut2_out = 32'sd18;
    8'd19: _sigmoid_lut2_out = 32'sd20;
    8'd20: _sigmoid_lut2_out = 32'sd23;
    8'd21: _sigmoid_lut2_out = 32'sd26;
    8'd22: _sigmoid_lut2_out = 32'sd30;
    8'd23: _sigmoid_lut2_out = 32'sd33;
    8'd24: _sigmoid_lut2_out = 32'sd38;
    8'd25: _sigmoid_lut2_out = 32'sd43;
    8'd26: _sigmoid_lut2_out = 32'sd49;
    8'd27: _sigmoid_lut2_out = 32'sd55;
    8'd28: _sigmoid_lut2_out = 32'sd63;
    8'd29: _sigmoid_lut2_out = 32'sd71;
    8'd30: _sigmoid_lut2_out = 32'sd80;
    8'd31: _sigmoid_lut2_out = 32'sd91;
    8'd32: _sigmoid_lut2_out = 32'sd103;
    8'd33: _sigmoid_lut2_out = 32'sd117;
    8'd34: _sigmoid_lut2_out = 32'sd132;
    8'd35: _sigmoid_lut2_out = 32'sd150;
    8'd36: _sigmoid_lut2_out = 32'sd170;
    8'd37: _sigmoid_lut2_out = 32'sd193;
    8'd38: _sigmoid_lut2_out = 32'sd218;
    8'd39: _sigmoid_lut2_out = 32'sd247;
    8'd40: _sigmoid_lut2_out = 32'sd280;
    8'd41: _sigmoid_lut2_out = 32'sd318;
    8'd42: _sigmoid_lut2_out = 32'sd360;
    8'd43: _sigmoid_lut2_out = 32'sd408;
    8'd44: _sigmoid_lut2_out = 32'sd462;
    8'd45: _sigmoid_lut2_out = 32'sd523;
    8'd46: _sigmoid_lut2_out = 32'sd593;
    8'd47: _sigmoid_lut2_out = 32'sd672;
    8'd48: _sigmoid_lut2_out = 32'sd762;
    8'd49: _sigmoid_lut2_out = 32'sd863;
    8'd50: _sigmoid_lut2_out = 32'sd978;
    8'd51: _sigmoid_lut2_out = 32'sd1108;
    8'd52: _sigmoid_lut2_out = 32'sd1256;
    8'd53: _sigmoid_lut2_out = 32'sd1423;
    8'd54: _sigmoid_lut2_out = 32'sd1612;
    8'd55: _sigmoid_lut2_out = 32'sd1827;
    8'd56: _sigmoid_lut2_out = 32'sd2070;
    8'd57: _sigmoid_lut2_out = 32'sd2346;
    8'd58: _sigmoid_lut2_out = 32'sd2658;
    8'd59: _sigmoid_lut2_out = 32'sd3012;
    8'd60: _sigmoid_lut2_out = 32'sd3413;
    8'd61: _sigmoid_lut2_out = 32'sd3867;
    8'd62: _sigmoid_lut2_out = 32'sd4382;
    8'd63: _sigmoid_lut2_out = 32'sd4965;
    8'd64: _sigmoid_lut2_out = 32'sd5626;
    8'd65: _sigmoid_lut2_out = 32'sd6375;
    8'd66: _sigmoid_lut2_out = 32'sd7224;
    8'd67: _sigmoid_lut2_out = 32'sd8185;
    8'd68: _sigmoid_lut2_out = 32'sd9274;
    8'd69: _sigmoid_lut2_out = 32'sd10508;
    8'd70: _sigmoid_lut2_out = 32'sd11906;
    8'd71: _sigmoid_lut2_out = 32'sd13490;
    8'd72: _sigmoid_lut2_out = 32'sd15285;
    8'd73: _sigmoid_lut2_out = 32'sd17318;
    8'd74: _sigmoid_lut2_out = 32'sd19621;
    8'd75: _sigmoid_lut2_out = 32'sd22230;
    8'd76: _sigmoid_lut2_out = 32'sd25186;
    8'd77: _sigmoid_lut2_out = 32'sd28533;
    8'd78: _sigmoid_lut2_out = 32'sd32325;
    8'd79: _sigmoid_lut2_out = 32'sd36620;
    8'd80: _sigmoid_lut2_out = 32'sd41484;
    8'd81: _sigmoid_lut2_out = 32'sd46992;
    8'd82: _sigmoid_lut2_out = 32'sd53229;
    8'd83: _sigmoid_lut2_out = 32'sd60291;
    8'd84: _sigmoid_lut2_out = 32'sd68286;
    8'd85: _sigmoid_lut2_out = 32'sd77336;
    8'd86: _sigmoid_lut2_out = 32'sd87579;
    8'd87: _sigmoid_lut2_out = 32'sd99171;
    8'd88: _sigmoid_lut2_out = 32'sd112287;
    8'd89: _sigmoid_lut2_out = 32'sd127125;
    8'd90: _sigmoid_lut2_out = 32'sd143906;
    8'd91: _sigmoid_lut2_out = 32'sd162881;
    8'd92: _sigmoid_lut2_out = 32'sd184330;
    8'd93: _sigmoid_lut2_out = 32'sd208568;
    8'd94: _sigmoid_lut2_out = 32'sd235949;
    8'd95: _sigmoid_lut2_out = 32'sd266865;
    8'd96: _sigmoid_lut2_out = 32'sd301759;
    8'd97: _sigmoid_lut2_out = 32'sd341120;
    8'd98: _sigmoid_lut2_out = 32'sd385496;
    8'd99: _sigmoid_lut2_out = 32'sd435492;
    8'd100: _sigmoid_lut2_out = 32'sd491778;
    8'd101: _sigmoid_lut2_out = 32'sd555091;
    8'd102: _sigmoid_lut2_out = 32'sd626241;
    8'd103: _sigmoid_lut2_out = 32'sd706115;
    8'd104: _sigmoid_lut2_out = 32'sd795674;
    8'd105: _sigmoid_lut2_out = 32'sd895959;
    8'd106: _sigmoid_lut2_out = 32'sd1008087;
    8'd107: _sigmoid_lut2_out = 32'sd1133245;
    8'd108: _sigmoid_lut2_out = 32'sd1272689;
    8'd109: _sigmoid_lut2_out = 32'sd1427725;
    8'd110: _sigmoid_lut2_out = 32'sd1599699;
    8'd111: _sigmoid_lut2_out = 32'sd1789971;
    8'd112: _sigmoid_lut2_out = 32'sd1999893;
    8'd113: _sigmoid_lut2_out = 32'sd2230770;
    8'd114: _sigmoid_lut2_out = 32'sd2483820;
    8'd115: _sigmoid_lut2_out = 32'sd2760128;
    8'd116: _sigmoid_lut2_out = 32'sd3060592;
    8'd117: _sigmoid_lut2_out = 32'sd3385864;
    8'd118: _sigmoid_lut2_out = 32'sd3736288;
    8'd119: _sigmoid_lut2_out = 32'sd4111844;
    8'd120: _sigmoid_lut2_out = 32'sd4512088;
    8'd121: _sigmoid_lut2_out = 32'sd4936108;
    8'd122: _sigmoid_lut2_out = 32'sd5382488;
    8'd123: _sigmoid_lut2_out = 32'sd5849295;
    8'd124: _sigmoid_lut2_out = 32'sd6334081;
    8'd125: _sigmoid_lut2_out = 32'sd6833920;
    8'd126: _sigmoid_lut2_out = 32'sd7345459;
    8'd127: _sigmoid_lut2_out = 32'sd7865002;
    8'd128: _sigmoid_lut2_out = 32'sd8388608;
    8'd129: _sigmoid_lut2_out = 32'sd8912214;
    8'd130: _sigmoid_lut2_out = 32'sd9431757;
    8'd131: _sigmoid_lut2_out = 32'sd9943296;
    8'd132: _sigmoid_lut2_out = 32'sd10443135;
    8'd133: _sigmoid_lut2_out = 32'sd10927921;
    8'd134: _sigmoid_lut2_out = 32'sd11394728;
    8'd135: _sigmoid_lut2_out = 32'sd11841108;
    8'd136: _sigmoid_lut2_out = 32'sd12265128;
    8'd137: _sigmoid_lut2_out = 32'sd12665372;
    8'd138: _sigmoid_lut2_out = 32'sd13040928;
    8'd139: _sigmoid_lut2_out = 32'sd13391352;
    8'd140: _sigmoid_lut2_out = 32'sd13716624;
    8'd141: _sigmoid_lut2_out = 32'sd14017088;
    8'd142: _sigmoid_lut2_out = 32'sd14293396;
    8'd143: _sigmoid_lut2_out = 32'sd14546446;
    8'd144: _sigmoid_lut2_out = 32'sd14777323;
    8'd145: _sigmoid_lut2_out = 32'sd14987245;
    8'd146: _sigmoid_lut2_out = 32'sd15177517;
    8'd147: _sigmoid_lut2_out = 32'sd15349491;
    8'd148: _sigmoid_lut2_out = 32'sd15504527;
    8'd149: _sigmoid_lut2_out = 32'sd15643971;
    8'd150: _sigmoid_lut2_out = 32'sd15769129;
    8'd151: _sigmoid_lut2_out = 32'sd15881257;
    8'd152: _sigmoid_lut2_out = 32'sd15981542;
    8'd153: _sigmoid_lut2_out = 32'sd16071101;
    8'd154: _sigmoid_lut2_out = 32'sd16150975;
    8'd155: _sigmoid_lut2_out = 32'sd16222125;
    8'd156: _sigmoid_lut2_out = 32'sd16285438;
    8'd157: _sigmoid_lut2_out = 32'sd16341724;
    8'd158: _sigmoid_lut2_out = 32'sd16391720;
    8'd159: _sigmoid_lut2_out = 32'sd16436096;
    8'd160: _sigmoid_lut2_out = 32'sd16475457;
    8'd161: _sigmoid_lut2_out = 32'sd16510351;
    8'd162: _sigmoid_lut2_out = 32'sd16541267;
    8'd163: _sigmoid_lut2_out = 32'sd16568648;
    8'd164: _sigmoid_lut2_out = 32'sd16592886;
    8'd165: _sigmoid_lut2_out = 32'sd16614335;
    8'd166: _sigmoid_lut2_out = 32'sd16633310;
    8'd167: _sigmoid_lut2_out = 32'sd16650091;
    8'd168: _sigmoid_lut2_out = 32'sd16664929;
    8'd169: _sigmoid_lut2_out = 32'sd16678045;
    8'd170: _sigmoid_lut2_out = 32'sd16689637;
    8'd171: _sigmoid_lut2_out = 32'sd16699880;
    8'd172: _sigmoid_lut2_out = 32'sd16708930;
    8'd173: _sigmoid_lut2_out = 32'sd16716925;
    8'd174: _sigmoid_lut2_out = 32'sd16723987;
    8'd175: _sigmoid_lut2_out = 32'sd16730224;
    8'd176: _sigmoid_lut2_out = 32'sd16735732;
    8'd177: _sigmoid_lut2_out = 32'sd16740596;
    8'd178: _sigmoid_lut2_out = 32'sd16744891;
    8'd179: _sigmoid_lut2_out = 32'sd16748683;
    8'd180: _sigmoid_lut2_out = 32'sd16752030;
    8'd181: _sigmoid_lut2_out = 32'sd16754986;
    8'd182: _sigmoid_lut2_out = 32'sd16757595;
    8'd183: _sigmoid_lut2_out = 32'sd16759898;
    8'd184: _sigmoid_lut2_out = 32'sd16761931;
    8'd185: _sigmoid_lut2_out = 32'sd16763726;
    8'd186: _sigmoid_lut2_out = 32'sd16765310;
    8'd187: _sigmoid_lut2_out = 32'sd16766708;
    8'd188: _sigmoid_lut2_out = 32'sd16767942;
    8'd189: _sigmoid_lut2_out = 32'sd16769031;
    8'd190: _sigmoid_lut2_out = 32'sd16769992;
    8'd191: _sigmoid_lut2_out = 32'sd16770841;
    8'd192: _sigmoid_lut2_out = 32'sd16771590;
    8'd193: _sigmoid_lut2_out = 32'sd16772251;
    8'd194: _sigmoid_lut2_out = 32'sd16772834;
    8'd195: _sigmoid_lut2_out = 32'sd16773349;
    8'd196: _sigmoid_lut2_out = 32'sd16773803;
    8'd197: _sigmoid_lut2_out = 32'sd16774204;
    8'd198: _sigmoid_lut2_out = 32'sd16774558;
    8'd199: _sigmoid_lut2_out = 32'sd16774870;
    8'd200: _sigmoid_lut2_out = 32'sd16775146;
    8'd201: _sigmoid_lut2_out = 32'sd16775389;
    8'd202: _sigmoid_lut2_out = 32'sd16775604;
    8'd203: _sigmoid_lut2_out = 32'sd16775793;
    8'd204: _sigmoid_lut2_out = 32'sd16775960;
    8'd205: _sigmoid_lut2_out = 32'sd16776108;
    8'd206: _sigmoid_lut2_out = 32'sd16776238;
    8'd207: _sigmoid_lut2_out = 32'sd16776353;
    8'd208: _sigmoid_lut2_out = 32'sd16776454;
    8'd209: _sigmoid_lut2_out = 32'sd16776544;
    8'd210: _sigmoid_lut2_out = 32'sd16776623;
    8'd211: _sigmoid_lut2_out = 32'sd16776693;
    8'd212: _sigmoid_lut2_out = 32'sd16776754;
    8'd213: _sigmoid_lut2_out = 32'sd16776808;
    8'd214: _sigmoid_lut2_out = 32'sd16776856;
    8'd215: _sigmoid_lut2_out = 32'sd16776898;
    8'd216: _sigmoid_lut2_out = 32'sd16776936;
    8'd217: _sigmoid_lut2_out = 32'sd16776969;
    8'd218: _sigmoid_lut2_out = 32'sd16776998;
    8'd219: _sigmoid_lut2_out = 32'sd16777023;
    8'd220: _sigmoid_lut2_out = 32'sd16777046;
    8'd221: _sigmoid_lut2_out = 32'sd16777066;
    8'd222: _sigmoid_lut2_out = 32'sd16777084;
    8'd223: _sigmoid_lut2_out = 32'sd16777099;
    8'd224: _sigmoid_lut2_out = 32'sd16777113;
    8'd225: _sigmoid_lut2_out = 32'sd16777125;
    8'd226: _sigmoid_lut2_out = 32'sd16777136;
    8'd227: _sigmoid_lut2_out = 32'sd16777145;
    8'd228: _sigmoid_lut2_out = 32'sd16777153;
    8'd229: _sigmoid_lut2_out = 32'sd16777161;
    8'd230: _sigmoid_lut2_out = 32'sd16777167;
    8'd231: _sigmoid_lut2_out = 32'sd16777173;
    8'd232: _sigmoid_lut2_out = 32'sd16777178;
    8'd233: _sigmoid_lut2_out = 32'sd16777183;
    8'd234: _sigmoid_lut2_out = 32'sd16777186;
    8'd235: _sigmoid_lut2_out = 32'sd16777190;
    8'd236: _sigmoid_lut2_out = 32'sd16777193;
    8'd237: _sigmoid_lut2_out = 32'sd16777196;
    8'd238: _sigmoid_lut2_out = 32'sd16777198;
    8'd239: _sigmoid_lut2_out = 32'sd16777200;
    8'd240: _sigmoid_lut2_out = 32'sd16777202;
    8'd241: _sigmoid_lut2_out = 32'sd16777204;
    8'd242: _sigmoid_lut2_out = 32'sd16777205;
    8'd243: _sigmoid_lut2_out = 32'sd16777206;
    8'd244: _sigmoid_lut2_out = 32'sd16777208;
    8'd245: _sigmoid_lut2_out = 32'sd16777209;
    8'd246: _sigmoid_lut2_out = 32'sd16777209;
    8'd247: _sigmoid_lut2_out = 32'sd16777210;
    8'd248: _sigmoid_lut2_out = 32'sd16777211;
    8'd249: _sigmoid_lut2_out = 32'sd16777211;
    8'd250: _sigmoid_lut2_out = 32'sd16777212;
    8'd251: _sigmoid_lut2_out = 32'sd16777212;
    8'd252: _sigmoid_lut2_out = 32'sd16777213;
    8'd253: _sigmoid_lut2_out = 32'sd16777213;
    8'd254: _sigmoid_lut2_out = 32'sd16777214;
    8'd255: _sigmoid_lut2_out = 32'sd16777214;
    default: _sigmoid_lut2_out = 32'sd0;
endcase

    wire signed [63:0] gain_product = P_K * _sigmoid_lut2_out;
    wire signed [63:0] gain_term = gain_product >>> 24;
    wire signed [63:0] x_candidate = -$signed({{32{x_reg[31]}}, x_reg}) +
        gain_term + $signed({{32{I_t[31]}}, I_t});
    wire signed [31:0] x_saturated =
        x_candidate > 64'sd2147483647 ? 32'sd2147483647 :
        x_candidate < -64'sd2147483648 ? -32'sd2147483648 :
        x_candidate[31:0];
    wire signed [31:0] x_next =
        x_saturated > Q_FIVE ? Q_FIVE :
        x_saturated < -Q_FIVE ? -Q_FIVE : x_saturated;

    wire signed [63:0] theta_product = P_BETA * theta_reg;
    wire signed [63:0] theta_decay = theta_product >>> 24;
    wire signed [63:0] theta_candidate = theta_decay +
        (x_reg >= P_THETA_SPIKE ? $signed({{32{P_GAMMA[31]}}, P_GAMMA}) : 64'sd0);
    wire signed [31:0] theta_saturated =
        theta_candidate > 64'sd2147483647 ? 32'sd2147483647 :
        theta_candidate < -64'sd2147483648 ? -32'sd2147483648 :
        theta_candidate[31:0];
    wire signed [31:0] theta_next =
        theta_saturated > Q_FIVE ? Q_FIVE :
        theta_saturated < -Q_FIVE ? -Q_FIVE : theta_saturated;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_reg <= 32'sd0;
            theta_reg <= 32'sd0;
            x_out <= 32'sd0;
            theta_out <= 32'sd0;
            spike_out <= 1'b0;
        end else begin
            x_reg <= x_next;
            theta_reg <= theta_next;
            x_out <= x_next;
            theta_out <= theta_next;
            spike_out <= (x_reg < P_X_THRESHOLD) && (x_next >= P_X_THRESHOLD);
        end
    end
endmodule
