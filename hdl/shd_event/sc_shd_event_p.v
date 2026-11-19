// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Event-driven integer SHD core with P shared lanes
//
// Provenance: imported from the group's earlier design study as Golden Reference #0,
// bit-exact with tools/shd_int_reference.py.
// Delayed events are (due time, address[, count]) entries in a 32-bucket time wheel per
// layer (bucket = due time mod 32; delays 0..30, so no aliasing). One logical step t:
//   PUSH1  (t < T)       walk non-zero channels of frame t into wheel 1, (t + d1[i]) mod 32
//   RD1    (t < T + 15)  read bucket t mod 32 once into buffer eb1 (dropped for t >= T + 15)
//   L1     (t < T + 15)  for each group g of P neurons, accumulate count * Q1[i] and run LIF
//   PUSH2  (t < T + 15)  walk the 128 layer-1 spikes into wheel 2, (t + d2[j]) mod 32
//   RD2, L2              as RD1/L1 with wheel 2, Q2 and layer 2; group spikes go to s2
//   OUT                  for each pass h of the 20 classes (P per pass), accumulate Q3[j]
// Buckets and buffers hold CAP1 / CAP2 entries. Pushing into a full bucket drops the event
// and raises sticky `overflow`. Simulation counters exist only outside synthesis.
// Group-word layout matches sc_shd_dense_p; clock gating uses sc_cg.
// Target selection is performed exclusively via the target filelist.
// No target preprocessor conditionals exist in the core or leaf modules.
`timescale 1ns / 1ps
module sc_shd_event_p #(
    parameter integer P         = 8,
    parameter integer FRAC_BITS = 0,
    parameter integer THETA1    = 1,
    parameter integer THETA2    = 1,
    parameter integer V1_WIDTH  = 22,
    parameter integer V2_WIDTH  = 17,
    parameter integer CAP1      = 62,
    parameter integer CAP2      = 39,
    parameter integer WSHAPE    = 6,
    parameter integer QSHAPE    = 12
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              load_we,
    input  wire [2:0]        load_sel,
    input  wire [7:0]        load_addr,
    input  wire [6:0]        load_grp,      // group (Q1/Q2) or class pass (Q3) of the word
    input  wire [8*P-1:0]    load_data,     // one group word, lane 0 in the low byte
    input  wire              start,
    input  wire [15:0]       t_orig,
    input  wire [140*4-1:0]  frame,
    input  wire              frame_valid,
    output wire              frame_ready,
    output reg               busy,
    output reg               done,
    output reg               error,
    output reg               overflow,
    output reg  [20*32-1:0]  out_packed
);
`include "sc_shapes.vh"
    localparam integer NI = 140, NH = 128, NO = 20;
    localparam integer T_MAX = 1024, HALF = 15;
    localparam integer G  = NH / P;
    localparam integer GW = (G > 1) ? $clog2(G) : 1;
    localparam integer WB = sc_shape_bits(WSHAPE);
    localparam integer W8 = (8 * P + WB - 1) / WB;
    localparam integer PC = (NO + P - 1) / P;
    localparam integer LP = NO - (PC - 1) * P;
    localparam integer W3 = (8 * P + WB - 1) / WB;
    localparam integer W3L = (8 * LP + WB - 1) / WB;
    localparam integer QB = sc_shape_bits(QSHAPE);
    localparam integer CE1 = (12 + QB - 1) / QB;
    localparam integer CE2 = (7 + QB - 1) / QB;
    localparam integer CW1 = $clog2(CAP1 + 1);
    localparam integer CW2 = $clog2(CAP2 + 1);
    localparam integer KW  = (CW1 > CW2) ? CW1 : CW2;

    localparam [3:0] S_IDLE = 0, S_WAIT = 1, S_PUSH1 = 2, S_RD1 = 3, S_L1 = 4, S_LIF1 = 5,
                     S_PUSH2 = 6, S_RD2 = 7, S_L2 = 8, S_LIF2 = 9, S_OUT = 10, S_NEXT = 11;
    reg [3:0]   state;
    reg [15:0]  t_len;
    reg [10:0]  t;
    reg [6:0]   g;
    reg [KW-1:0] k;             // RD: next entry to read; L: next buffered entry to issue
    reg [KW-1:0] nbuf;          // entries in the buffer of the current layer
    reg [NI*4-1:0] frame_r;
    reg [NI-1:0]   pend1;
    reg [NH-1:0]   pend, s1, s2;
    reg            lif_fin;
    reg [10:0]     k_it;

    wire [4:0] slot     = t[4:0];
    wire       in_input = ({5'd0, t} < t_len);
    wire       in_l1    = ({5'd0, t} < t_len + HALF);
    assign frame_ready = (state == S_WAIT) && in_input;

    wire cclk;
    wire waiting = (state == S_IDLE && !start && !load_we)
                || (state == S_WAIT && in_input && !frame_valid);
    sc_cg #(1) cg_root (.clk(clk), .en(!waiting), .gclk(cclk));

    reg [4:0] d1 [0:NI-1];
    reg [4:0] d2 [0:NH-1];
    always @(posedge cclk) begin
        if (!busy && load_we && load_sel == 3) d1[load_addr] <= load_data[4:0];
        if (!busy && load_we && load_sel == 4) d2[load_addr[6:0]] <= load_data[4:0];
    end

    // ------------------------------------------------------------------ priority encoders
    function [7:0] ffs140(input [NI-1:0] m);
        integer q;
        begin ffs140 = 0; for (q = NI - 1; q >= 0; q = q - 1) if (m[q]) ffs140 = q; end
    endfunction
    function [6:0] ffs128(input [NH-1:0] m);
        integer q;
        begin ffs128 = 0; for (q = NH - 1; q >= 0; q = q - 1) if (m[q]) ffs128 = q; end
    endfunction
    wire [7:0] e1 = ffs140(pend1);
    wire [6:0] e2 = ffs128(pend);
    wire [3:0] e1_cnt = frame_r[4*e1 +: 4];
    wire [4:0] b1 = slot + d1[e1];
    wire [4:0] b2 = slot + d2[e2];

    // ------------------------------------------------------------------ wheels
    reg [CW1-1:0] cnt1 [0:31];
    reg [CW2-1:0] cnt2 [0:31];
    wire push1 = (state == S_PUSH1) && pend1 != 0;
    wire push2 = (state == S_PUSH2) && pend != 0;
    wire full1 = cnt1[b1] == CAP1;
    wire full2 = cnt2[b2] == CAP2;
    wire [CW1-1:0] n1 = cnt1[slot];
    wire [CW2-1:0] n2 = cnt2[slot];
    wire rd1 = (state == S_RD1) && in_l1 && k < n1;
    wire rd2 = (state == S_RD2) && k < n2;
    wire [11:0] w1_q;
    wire [6:0]  w2_q;
    sc_mem #(.DEPTH(32*CAP1), .WIDTH(12), .SHAPE(QSHAPE)) w1 (
        .clk(cclk), .we(push1 && !full1), .re(rd1),
        .addr(push1 ? b1 * CAP1 + cnt1[b1] : slot * CAP1 + k),
        .wdata({e1_cnt, e1}), .rmask({CE1{1'b1}}), .rdata(w1_q));
    sc_mem #(.DEPTH(32*CAP2), .WIDTH(7), .SHAPE(QSHAPE)) w2 (
        .clk(cclk), .we(push2 && !full2), .re(rd2),
        .addr(push2 ? b2 * CAP2 + cnt2[b2] : slot * CAP2 + k),
        .wdata(e2), .rmask({CE2{1'b1}}), .rdata(w2_q));

    // ------------------------------------------------------------------ entry buffers
    reg [11:0] eb1 [0:CAP1-1];
    reg [6:0]  eb2 [0:CAP2-1];
    reg        rp_v;
    reg [KW-1:0] rp_k;

    // ------------------------------------------------------------------ weight rows
    // Group-word layout as in sc_shd_dense_p: group g of row i at address i*G + g, class
    // pass h of neuron j at address j*PC + h; the last pass reads only its W3L words.
    function [W3-1:0] cmask(input integer hi);
        integer q;
        begin
            cmask = 0;
            for (q = 0; q < W3; q = q + 1) if (q < ((hi == PC - 1) ? W3L : W3)) cmask[q] = 1'b1;
        end
    endfunction
    wire [GW-1:0] gg = g[GW-1:0];
    wire issue_l = ((state == S_L1) || (state == S_L2)) && k < nbuf;
    wire issue_o = (state == S_OUT) && pend != 0;
    reg        p1_v;
    reg [3:0]  p1_y;
    wire [11:0] ent1 = eb1[k];
    wire [6:0]  ent2 = eb2[k];
    wire [W8*WB-1:0] q1_row, q2_row;
    wire [W3*WB-1:0] q3_row;
    wire ld = !busy && load_we;
    sc_mem #(.DEPTH(NI*G), .WIDTH(W8*WB), .SHAPE(WSHAPE)) q1 (
        .clk(cclk), .we(ld && load_sel == 0), .re(issue_l && state == S_L1),
        .addr(busy ? ent1[7:0] * G + gg : load_addr * G + load_grp), .wdata(load_data),
        .rmask({W8{1'b1}}), .rdata(q1_row));
    sc_mem #(.DEPTH(NH*G), .WIDTH(W8*WB), .SHAPE(WSHAPE)) q2 (
        .clk(cclk), .we(ld && load_sel == 1), .re(issue_l && state == S_L2),
        .addr(busy ? ent2 * G + gg : load_addr[6:0] * G + load_grp), .wdata(load_data),
        .rmask({W8{1'b1}}), .rdata(q2_row));
    sc_mem #(.DEPTH(NH*PC), .WIDTH(W3*WB), .SHAPE(WSHAPE)) q3 (
        .clk(cclk), .we(ld && load_sel == 2), .re(issue_o),
        .addr(busy ? e2 * PC + g : load_addr[6:0] * PC + load_grp), .wdata(load_data),
        .rmask(cmask(g)), .rdata(q3_row));
    wire [8*P-1:0] wq = (state == S_L1) ? q1_row[8*P-1:0] : q2_row[8*P-1:0];
    wire [8*P-1:0] w3 = q3_row[8*P-1:0];

    // ------------------------------------------------------------------ lanes
    wire start_ok = (state == S_IDLE) && start && t_orig != 0 && t_orig <= T_MAX;
    wire in_lif   = (state == S_LIF1) || (state == S_LIF2);
    wire [P-1:0] spk, it_v, upd_v;
    wire any_it;
    sc_lanes #(.P(P), .FRAC_BITS(FRAC_BITS), .THETA1(THETA1), .THETA2(THETA2),
               .V1_WIDTH(V1_WIDTH), .V2_WIDTH(V2_WIDTH), .LAZY(1), .GATING(1)) lanes (
        .clk(cclk), .init(start_ok), .layer(state == S_L2 || state == S_LIF2), .grp(g),
        .clr(((state == S_L1) || (state == S_L2)) && k == 0 && !p1_v),
        .add(p1_v && (state == S_L1 || state == S_L2)), .y(p1_y), .w(wq),
        .leak(in_lif && !lif_fin), .lif(in_lif && lif_fin), .t(t), .k_it(k_it), .in_l1(in_l1),
        .spk(spk), .any_it(any_it), .it_v(it_v), .upd_v(upd_v));

    wire [NO*32-1:0] out_now;
    genvar c;
    generate
        for (c = 0; c < NO; c = c + 1) begin : cls
            reg signed [31:0] acc;
            wire hit = (state == S_OUT) && p1_v && (g == c / P);
            wire gc;
            sc_cg #(1) cg (.clk(cclk), .en(start_ok | hit), .gclk(gc));
            always @(posedge gc) begin
                if (start_ok) acc <= 0;
                else if (hit) acc <= acc + $signed(w3[8*(c % P) +: 8]);
            end
            assign out_now[32*c +: 32] = acc;
        end
    endgenerate

    // the full spike vector with group gi's spikes in place
    function [NH-1:0] merge(input [NH-1:0] s, input [P-1:0] sp, input integer gi);
        integer m;
        begin merge = s; for (m = 0; m < P; m = m + 1) merge[gi * P + m] = sp[m]; end
    endfunction

    integer q;
    always @(posedge cclk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; error <= 1'b0; overflow <= 1'b0;
            t <= 0; t_len <= 0; g <= 0; k <= 0; nbuf <= 0; out_packed <= 0; frame_r <= 0;
            pend1 <= 0; pend <= 0; s1 <= 0; s2 <= 0; rp_v <= 1'b0; rp_k <= 0;
            p1_v <= 1'b0; p1_y <= 0; lif_fin <= 1'b0; k_it <= 0;
            for (q = 0; q < 32; q = q + 1) begin cnt1[q] <= 0; cnt2[q] <= 0; end
        end else begin
            done <= 1'b0;
            p1_v <= 1'b0;
            rp_v <= 1'b0;
            case (state)
            S_IDLE: if (start) begin
                if (!start_ok) error <= 1'b1;
                else begin
                    error <= 1'b0; overflow <= 1'b0; t_len <= t_orig; t <= 0; busy <= 1'b1;
                    for (q = 0; q < 32; q = q + 1) begin cnt1[q] <= 0; cnt2[q] <= 0; end
                    state <= S_WAIT;
                end
            end
            S_WAIT: begin
                k <= 0; s1 <= 0;
                if (!in_input) begin frame_r <= 0; pend1 <= 0; state <= S_RD1; end
                else if (frame_valid) begin
                    frame_r <= frame; state <= S_PUSH1;
                    for (q = 0; q < NI; q = q + 1) pend1[q] <= frame[4*q +: 4] != 0;
                end
            end
            S_PUSH1: begin
                if (pend1 != 0) begin
                    if (full1) overflow <= 1'b1; else cnt1[b1] <= cnt1[b1] + 1;
                    pend1[e1] <= 1'b0;
                end else begin k <= 0; state <= S_RD1; end
            end
            S_RD1, S_RD2: begin
                if ((state == S_RD1) ? rd1 : rd2) begin
                    rp_v <= 1'b1; rp_k <= k; k <= k + 1;
                end
                if (rp_v) begin
                    if (state == S_RD1) eb1[rp_k] <= w1_q; else eb2[rp_k] <= w2_q;
                end
                if (!((state == S_RD1) ? rd1 : rd2) && !rp_v) begin
                    nbuf <= (state == S_RD1) ? (in_l1 ? n1 : 0) : n2;
                    if (state == S_RD1) cnt1[slot] <= 0; else cnt2[slot] <= 0;
                    k <= 0; g <= 0;
                    if (state == S_RD1) state <= in_l1 ? S_L1 : S_RD2;
                    else state <= S_L2;
                end
            end
            S_L1, S_L2: begin
                if (issue_l) k <= k + 1;
                p1_v <= issue_l;
                p1_y <= (state == S_L1) ? ent1[11:8] : 4'd1;
                if (!issue_l && !p1_v) begin
                    // the last read's data arrives with p1_v; drained one cycle later
                    k <= 0; lif_fin <= 1'b0; k_it <= 0;
                    state <= (state == S_L1) ? S_LIF1 : S_LIF2;
                end
            end
            S_LIF1, S_LIF2: begin
                if (!lif_fin) begin
                    if (any_it) k_it <= k_it + 1; else lif_fin <= 1'b1;
                end else begin
                    if (state == S_LIF1) s1[g*P +: P] <= spk; else s2[g*P +: P] <= spk;
                    lif_fin <= 1'b0; k_it <= 0; k <= 0;
                    if (g == G - 1) begin
                        g <= 0;
                        if (state == S_LIF1) begin
                            pend <= merge(s1, spk, g);
                            state <= S_PUSH2;
                        end else begin
                            pend <= merge(s2, spk, g);
                            state <= S_OUT;
                        end
                    end else begin
                        g <= g + 1;
                        state <= (state == S_LIF1) ? S_L1 : S_L2;
                    end
                end
            end
            S_PUSH2: begin
                if (pend != 0) begin
                    if (full2) overflow <= 1'b1; else cnt2[b2] <= cnt2[b2] + 1;
                    pend[e2] <= 1'b0;
                end else begin k <= 0; state <= S_RD2; end
            end
            S_OUT: begin
                if (issue_o) pend[e2] <= 1'b0;
                p1_v <= issue_o;
                if (!issue_o && !p1_v) begin
                    if (g == PC - 1) state <= S_NEXT;
                    else begin g <= g + 1; pend <= s2; end
                end
            end
            S_NEXT: begin
                if ({5'd0, t} == t_len + 2 * HALF - 1) begin
                    out_packed <= out_now; busy <= 1'b0; done <= 1'b1; state <= S_IDLE;
                end else begin
                    t <= t + 1; state <= S_WAIT;
                end
            end
            default: state <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    integer cycles = 0, l1_rows = 0, l2_rows = 0, l3_rows = 0, mem_rw = 0, leak_it = 0;
    integer pushes = 0, pops = 0, n;
    always @(posedge cclk) begin
        if (start_ok) begin
            cycles = 0; l1_rows = 0; l2_rows = 0; l3_rows = 0; mem_rw = 0; leak_it = 0;
            pushes = 0; pops = 0;
        end else if (busy) begin
            cycles = cycles + 1;
            if (issue_l && state == S_L1 && g == 0) l1_rows = l1_rows + 1;
            if (issue_l && state == S_L2 && g == 0) l2_rows = l2_rows + 1;
            if (issue_o && g == 0) l3_rows = l3_rows + 1;
            if (push1 || push2) pushes = pushes + 1;
            if (rd1 || rd2) pops = pops + 1;
            for (n = 0; n < P; n = n + 1) begin
                leak_it = leak_it + it_v[n];
                mem_rw = mem_rw + 2 * upd_v[n];
            end
        end
    end
`endif
endmodule
