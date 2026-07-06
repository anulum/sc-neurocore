// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fixed-point scaled-dot-product softmax attention core

//hdl/sc_softmax_attention.v
//
// Combinational single-head scaled-dot-product attention:
//
//   attn[i] = softmax( Q[i] . K^T * inv_temp ) . V
//
// implementing the reference StochasticAttention::forward_softmax
// (engine/src/attention.rs): the scaled scores are made numerically stable by
// subtracting the row maximum before exp, normalised by their sum, and used to
// weight the V rows. `inv_temp` is 1/sqrt(dim_k) (the Vaswani et al. 2017
// scaling), baked as a Q8.16 parameter by the emitter.
//
// Fixed-point format: Q8.16 signed, 24-bit. Q, K and V are row-major
// (Q is Q_ROWS x DIM_K, K is K_ROWS x DIM_K, V is K_ROWS x V_COLS); the output is
// Q_ROWS x V_COLS. Word 0 sits at the bus LSB. Rounding is deferred throughout:
// the Q.K^T dot products and the scores.V products accumulate in wide Q(2*16)
// accumulators shifted once, and the softmax normalisation is a single integer
// division per weight.
//
// exp() is a 256-entry lookup over the symmetric [-16, 16) grid at 0.125
// spacing (mirroring compiler/expr_lut_tables.py + c_fixed_emitter.py): the
// argument is offset by 16*2^16, arithmetic-shifted right by FRACTION-3, clamped
// to [0, 255], and indexes exp(x)*2^16 saturated to 2^(24-1)-1. After the
// row-max subtraction the argument is non-positive, so the used range is
// [-16, 0] with exp(0) landing exactly on 1.0. The fixed-point result is defined
// solely by this module and mirrored bit-for-bit by the co-simulation oracle;
// the coarse 0.125 exp grid bounds its divergence from the ideal float softmax.

`timescale 1ns / 1ps

module sc_softmax_attention #(
    parameter integer Q_ROWS = 2,
    parameter integer K_ROWS = 2,
    parameter integer DIM_K = 2,
    parameter integer V_COLS = 2,
    parameter integer DATA_WIDTH = 24,
    parameter integer FRACTION = 16,
    // Q(FRACTION) softmax scaling 1/sqrt(dim_k); default dim_k = 2.
    parameter signed [DATA_WIDTH-1:0] INV_TEMP = 24'sd46341,
    // exp LUT index geometry: FRACTION - log2(1/step) and 16*2^FRACTION.
    parameter integer EXP_SHIFT = 13,
    parameter integer EXP_MIN_ABS = 1048576
)(
    input  wire signed [Q_ROWS*DIM_K*DATA_WIDTH-1:0]  q_in,
    input  wire signed [K_ROWS*DIM_K*DATA_WIDTH-1:0]  k_in,
    input  wire signed [K_ROWS*V_COLS*DATA_WIDTH-1:0] v_in,
    output wire signed [Q_ROWS*V_COLS*DATA_WIDTH-1:0] attn_out
);
    localparam integer SCORE_ACC_W = 2*DATA_WIDTH + 8;   // Q(2F) dot accumulator
    localparam integer SCORE_MUL_W = SCORE_ACC_W + DATA_WIDTH;
    localparam integer SUM_W = DATA_WIDTH + 8;           // sum of exp values
    localparam integer OUT_ACC_W = 2*DATA_WIDTH + 8;     // Q(2F) weighted-V accumulator

    // Q8.16 exp() lookup over the [-16, 16) grid, saturated to the signed word max.
    function automatic signed [DATA_WIDTH-1:0] exp_lut;
        input signed [DATA_WIDTH-1:0] arg;
        reg signed [DATA_WIDTH:0] offset;
        reg signed [DATA_WIDTH:0] raw;
        reg [31:0] idx;
        begin
            offset = $signed({arg[DATA_WIDTH-1], arg}) + EXP_MIN_ABS;
            raw = offset >>> EXP_SHIFT;
            if (raw < 0) idx = 0;
            else if (raw > 255) idx = 255;
            else idx = raw;
            case (idx)
                   0: exp_lut = 24'sd0;
                   1: exp_lut = 24'sd0;
                   2: exp_lut = 24'sd0;
                   3: exp_lut = 24'sd0;
                   4: exp_lut = 24'sd0;
                   5: exp_lut = 24'sd0;
                   6: exp_lut = 24'sd0;
                   7: exp_lut = 24'sd0;
                   8: exp_lut = 24'sd0;
                   9: exp_lut = 24'sd0;
                  10: exp_lut = 24'sd0;
                  11: exp_lut = 24'sd0;
                  12: exp_lut = 24'sd0;
                  13: exp_lut = 24'sd0;
                  14: exp_lut = 24'sd0;
                  15: exp_lut = 24'sd0;
                  16: exp_lut = 24'sd0;
                  17: exp_lut = 24'sd0;
                  18: exp_lut = 24'sd0;
                  19: exp_lut = 24'sd0;
                  20: exp_lut = 24'sd0;
                  21: exp_lut = 24'sd0;
                  22: exp_lut = 24'sd0;
                  23: exp_lut = 24'sd0;
                  24: exp_lut = 24'sd0;
                  25: exp_lut = 24'sd0;
                  26: exp_lut = 24'sd0;
                  27: exp_lut = 24'sd0;
                  28: exp_lut = 24'sd0;
                  29: exp_lut = 24'sd0;
                  30: exp_lut = 24'sd0;
                  31: exp_lut = 24'sd0;
                  32: exp_lut = 24'sd0;
                  33: exp_lut = 24'sd0;
                  34: exp_lut = 24'sd1;
                  35: exp_lut = 24'sd1;
                  36: exp_lut = 24'sd1;
                  37: exp_lut = 24'sd1;
                  38: exp_lut = 24'sd1;
                  39: exp_lut = 24'sd1;
                  40: exp_lut = 24'sd1;
                  41: exp_lut = 24'sd1;
                  42: exp_lut = 24'sd1;
                  43: exp_lut = 24'sd2;
                  44: exp_lut = 24'sd2;
                  45: exp_lut = 24'sd2;
                  46: exp_lut = 24'sd2;
                  47: exp_lut = 24'sd3;
                  48: exp_lut = 24'sd3;
                  49: exp_lut = 24'sd3;
                  50: exp_lut = 24'sd4;
                  51: exp_lut = 24'sd4;
                  52: exp_lut = 24'sd5;
                  53: exp_lut = 24'sd6;
                  54: exp_lut = 24'sd6;
                  55: exp_lut = 24'sd7;
                  56: exp_lut = 24'sd8;
                  57: exp_lut = 24'sd9;
                  58: exp_lut = 24'sd10;
                  59: exp_lut = 24'sd12;
                  60: exp_lut = 24'sd13;
                  61: exp_lut = 24'sd15;
                  62: exp_lut = 24'sd17;
                  63: exp_lut = 24'sd19;
                  64: exp_lut = 24'sd22;
                  65: exp_lut = 24'sd25;
                  66: exp_lut = 24'sd28;
                  67: exp_lut = 24'sd32;
                  68: exp_lut = 24'sd36;
                  69: exp_lut = 24'sd41;
                  70: exp_lut = 24'sd47;
                  71: exp_lut = 24'sd53;
                  72: exp_lut = 24'sd60;
                  73: exp_lut = 24'sd68;
                  74: exp_lut = 24'sd77;
                  75: exp_lut = 24'sd87;
                  76: exp_lut = 24'sd99;
                  77: exp_lut = 24'sd112;
                  78: exp_lut = 24'sd127;
                  79: exp_lut = 24'sd143;
                  80: exp_lut = 24'sd162;
                  81: exp_lut = 24'sd184;
                  82: exp_lut = 24'sd209;
                  83: exp_lut = 24'sd236;
                  84: exp_lut = 24'sd268;
                  85: exp_lut = 24'sd303;
                  86: exp_lut = 24'sd344;
                  87: exp_lut = 24'sd390;
                  88: exp_lut = 24'sd442;
                  89: exp_lut = 24'sd500;
                  90: exp_lut = 24'sd567;
                  91: exp_lut = 24'sd642;
                  92: exp_lut = 24'sd728;
                  93: exp_lut = 24'sd825;
                  94: exp_lut = 24'sd935;
                  95: exp_lut = 24'sd1059;
                  96: exp_lut = 24'sd1200;
                  97: exp_lut = 24'sd1360;
                  98: exp_lut = 24'sd1541;
                  99: exp_lut = 24'sd1746;
                 100: exp_lut = 24'sd1979;
                 101: exp_lut = 24'sd2243;
                 102: exp_lut = 24'sd2541;
                 103: exp_lut = 24'sd2879;
                 104: exp_lut = 24'sd3263;
                 105: exp_lut = 24'sd3697;
                 106: exp_lut = 24'sd4190;
                 107: exp_lut = 24'sd4747;
                 108: exp_lut = 24'sd5380;
                 109: exp_lut = 24'sd6096;
                 110: exp_lut = 24'sd6907;
                 111: exp_lut = 24'sd7827;
                 112: exp_lut = 24'sd8869;
                 113: exp_lut = 24'sd10050;
                 114: exp_lut = 24'sd11388;
                 115: exp_lut = 24'sd12905;
                 116: exp_lut = 24'sd14623;
                 117: exp_lut = 24'sd16570;
                 118: exp_lut = 24'sd18776;
                 119: exp_lut = 24'sd21276;
                 120: exp_lut = 24'sd24109;
                 121: exp_lut = 24'sd27319;
                 122: exp_lut = 24'sd30957;
                 123: exp_lut = 24'sd35079;
                 124: exp_lut = 24'sd39750;
                 125: exp_lut = 24'sd45042;
                 126: exp_lut = 24'sd51039;
                 127: exp_lut = 24'sd57835;
                 128: exp_lut = 24'sd65536;
                 129: exp_lut = 24'sd74262;
                 130: exp_lut = 24'sd84150;
                 131: exp_lut = 24'sd95354;
                 132: exp_lut = 24'sd108051;
                 133: exp_lut = 24'sd122437;
                 134: exp_lut = 24'sd138740;
                 135: exp_lut = 24'sd157213;
                 136: exp_lut = 24'sd178145;
                 137: exp_lut = 24'sd201865;
                 138: exp_lut = 24'sd228743;
                 139: exp_lut = 24'sd259200;
                 140: exp_lut = 24'sd293712;
                 141: exp_lut = 24'sd332819;
                 142: exp_lut = 24'sd377134;
                 143: exp_lut = 24'sd427348;
                 144: exp_lut = 24'sd484249;
                 145: exp_lut = 24'sd548726;
                 146: exp_lut = 24'sd621788;
                 147: exp_lut = 24'sd704578;
                 148: exp_lut = 24'sd798392;
                 149: exp_lut = 24'sd904697;
                 150: exp_lut = 24'sd1025156;
                 151: exp_lut = 24'sd1161653;
                 152: exp_lut = 24'sd1316326;
                 153: exp_lut = 24'sd1491592;
                 154: exp_lut = 24'sd1690196;
                 155: exp_lut = 24'sd1915243;
                 156: exp_lut = 24'sd2170254;
                 157: exp_lut = 24'sd2459220;
                 158: exp_lut = 24'sd2786662;
                 159: exp_lut = 24'sd3157701;
                 160: exp_lut = 24'sd3578144;
                 161: exp_lut = 24'sd4054569;
                 162: exp_lut = 24'sd4594428;
                 163: exp_lut = 24'sd5206169;
                 164: exp_lut = 24'sd5899363;
                 165: exp_lut = 24'sd6684854;
                 166: exp_lut = 24'sd7574932;
                 167: exp_lut = 24'sd8388607;
                 168: exp_lut = 24'sd8388607;
                 169: exp_lut = 24'sd8388607;
                 170: exp_lut = 24'sd8388607;
                 171: exp_lut = 24'sd8388607;
                 172: exp_lut = 24'sd8388607;
                 173: exp_lut = 24'sd8388607;
                 174: exp_lut = 24'sd8388607;
                 175: exp_lut = 24'sd8388607;
                 176: exp_lut = 24'sd8388607;
                 177: exp_lut = 24'sd8388607;
                 178: exp_lut = 24'sd8388607;
                 179: exp_lut = 24'sd8388607;
                 180: exp_lut = 24'sd8388607;
                 181: exp_lut = 24'sd8388607;
                 182: exp_lut = 24'sd8388607;
                 183: exp_lut = 24'sd8388607;
                 184: exp_lut = 24'sd8388607;
                 185: exp_lut = 24'sd8388607;
                 186: exp_lut = 24'sd8388607;
                 187: exp_lut = 24'sd8388607;
                 188: exp_lut = 24'sd8388607;
                 189: exp_lut = 24'sd8388607;
                 190: exp_lut = 24'sd8388607;
                 191: exp_lut = 24'sd8388607;
                 192: exp_lut = 24'sd8388607;
                 193: exp_lut = 24'sd8388607;
                 194: exp_lut = 24'sd8388607;
                 195: exp_lut = 24'sd8388607;
                 196: exp_lut = 24'sd8388607;
                 197: exp_lut = 24'sd8388607;
                 198: exp_lut = 24'sd8388607;
                 199: exp_lut = 24'sd8388607;
                 200: exp_lut = 24'sd8388607;
                 201: exp_lut = 24'sd8388607;
                 202: exp_lut = 24'sd8388607;
                 203: exp_lut = 24'sd8388607;
                 204: exp_lut = 24'sd8388607;
                 205: exp_lut = 24'sd8388607;
                 206: exp_lut = 24'sd8388607;
                 207: exp_lut = 24'sd8388607;
                 208: exp_lut = 24'sd8388607;
                 209: exp_lut = 24'sd8388607;
                 210: exp_lut = 24'sd8388607;
                 211: exp_lut = 24'sd8388607;
                 212: exp_lut = 24'sd8388607;
                 213: exp_lut = 24'sd8388607;
                 214: exp_lut = 24'sd8388607;
                 215: exp_lut = 24'sd8388607;
                 216: exp_lut = 24'sd8388607;
                 217: exp_lut = 24'sd8388607;
                 218: exp_lut = 24'sd8388607;
                 219: exp_lut = 24'sd8388607;
                 220: exp_lut = 24'sd8388607;
                 221: exp_lut = 24'sd8388607;
                 222: exp_lut = 24'sd8388607;
                 223: exp_lut = 24'sd8388607;
                 224: exp_lut = 24'sd8388607;
                 225: exp_lut = 24'sd8388607;
                 226: exp_lut = 24'sd8388607;
                 227: exp_lut = 24'sd8388607;
                 228: exp_lut = 24'sd8388607;
                 229: exp_lut = 24'sd8388607;
                 230: exp_lut = 24'sd8388607;
                 231: exp_lut = 24'sd8388607;
                 232: exp_lut = 24'sd8388607;
                 233: exp_lut = 24'sd8388607;
                 234: exp_lut = 24'sd8388607;
                 235: exp_lut = 24'sd8388607;
                 236: exp_lut = 24'sd8388607;
                 237: exp_lut = 24'sd8388607;
                 238: exp_lut = 24'sd8388607;
                 239: exp_lut = 24'sd8388607;
                 240: exp_lut = 24'sd8388607;
                 241: exp_lut = 24'sd8388607;
                 242: exp_lut = 24'sd8388607;
                 243: exp_lut = 24'sd8388607;
                 244: exp_lut = 24'sd8388607;
                 245: exp_lut = 24'sd8388607;
                 246: exp_lut = 24'sd8388607;
                 247: exp_lut = 24'sd8388607;
                 248: exp_lut = 24'sd8388607;
                 249: exp_lut = 24'sd8388607;
                 250: exp_lut = 24'sd8388607;
                 251: exp_lut = 24'sd8388607;
                 252: exp_lut = 24'sd8388607;
                 253: exp_lut = 24'sd8388607;
                 254: exp_lut = 24'sd8388607;
                default: exp_lut = 24'sd8388607;
            endcase
        end
    endfunction

    genvar gi, gc;
    generate
        for (gi = 0; gi < Q_ROWS; gi = gi + 1) begin : qrow
            integer j, d, c;
            reg signed [SCORE_ACC_W-1:0] dot_acc;
            reg signed [SCORE_MUL_W-1:0] score_mul;
            reg signed [DATA_WIDTH-1:0]  score  [0:K_ROWS-1];
            reg signed [DATA_WIDTH-1:0]  maxsc;
            reg signed [DATA_WIDTH-1:0]  e_val  [0:K_ROWS-1];
            reg signed [SUM_W-1:0]       sum_e;
            reg signed [SCORE_MUL_W-1:0] num_w;
            reg signed [DATA_WIDTH-1:0]  w_val  [0:K_ROWS-1];
            reg signed [OUT_ACC_W-1:0]   out_acc;
            reg signed [DATA_WIDTH-1:0]  q_e, k_e, v_e;
            reg signed [DATA_WIDTH-1:0]  qrow_out [0:V_COLS-1];

            always @* begin
                // scores[j] = (sum_d Q[i][d]*K[j][d]) * inv_temp, rounding deferred.
                for (j = 0; j < K_ROWS; j = j + 1) begin
                    dot_acc = {SCORE_ACC_W{1'b0}};
                    for (d = 0; d < DIM_K; d = d + 1) begin
                        q_e = q_in[(gi*DIM_K + d)*DATA_WIDTH +: DATA_WIDTH];
                        k_e = k_in[(j*DIM_K + d)*DATA_WIDTH +: DATA_WIDTH];
                        dot_acc = dot_acc + (q_e * k_e);
                    end
                    score_mul = dot_acc * INV_TEMP;
                    score[j] = score_mul >>> (2*FRACTION);
                end
                // numerically stable softmax: subtract the row max before exp.
                maxsc = score[0];
                for (j = 1; j < K_ROWS; j = j + 1)
                    if (score[j] > maxsc) maxsc = score[j];
                sum_e = {SUM_W{1'b0}};
                for (j = 0; j < K_ROWS; j = j + 1) begin
                    e_val[j] = exp_lut(score[j] - maxsc);
                    sum_e = sum_e + e_val[j];
                end
                // normalise (widen the numerator before the FRACTION shift).
                for (j = 0; j < K_ROWS; j = j + 1) begin
                    num_w = e_val[j];
                    num_w = num_w <<< FRACTION;
                    w_val[j] = num_w / sum_e;
                end
                // attn[i][c] = sum_j softmax[j] * V[j][c].
                for (c = 0; c < V_COLS; c = c + 1) begin
                    out_acc = {OUT_ACC_W{1'b0}};
                    for (j = 0; j < K_ROWS; j = j + 1) begin
                        v_e = v_in[(j*V_COLS + c)*DATA_WIDTH +: DATA_WIDTH];
                        out_acc = out_acc + (w_val[j] * v_e);
                    end
                    qrow_out[c] = out_acc >>> FRACTION;
                end
            end

            for (gc = 0; gc < V_COLS; gc = gc + 1) begin : ocol
                assign attn_out[(gi*V_COLS + gc)*DATA_WIDTH +: DATA_WIDTH] = qrow_out[gc];
            end
        end
    endgenerate

endmodule
