// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bit-exact integer SHD core (140 counts -> 128 -> 128 -> 20), event-driven
//
// Computes exactly what tools/shd_int_reference.py defines. One logical step t runs a fixed
// phase order; every read in a phase comes after the write it depends on, which is what makes
// the result independent of clock timing:
//
//   FILL  frame t (zeros once t >= T) into input history H1, slot t mod 32
//   A     layer 1: for i in 0..139, y = H1[(t - d1[i]) mod 32][i] if t >= d1[i] else 0;
//         if y != 0 (or ZERO_SKIP == 0) read row Q1[i] and add y * Q1[i][j] into acc1[j]
//   B     LIF 1: u = v - (v >>> 2) + (acc1 <<< F); spike = u >= THETA1; v = spike ? 0 : u;
//         s1 (zeros once t >= T + 15) into spike history H2, slot t mod 32
//   C     layer 2: for j in 0..127, b = H2[(t - d2[j]) mod 32][j] if t >= d2[j] else 0;
//         if b (or ZERO_SKIP == 0) read row Q2[j] and add b * Q2[j][k] into acc2[k]
//   D     LIF 2 as B with THETA2
//   E     layer 3: for j with s2[j] (or every j) read row Q3[j] and add s2[j] * Q3[j][k] into out[k]
//
// Layer 1 is active for t < T + 15 and layers 2 and 3 for t < T + 30, as the PyTorch delay
// padding defines. The guard t >= d makes every slot written before this sample read as zero, so
// nothing of the previous sample is visible. Accumulators and membranes cover the bounds proven
// for T <= 1024, counts <= 15 and int8 weights; nothing saturates. ZERO_SKIP changes only which
// rows are read, never a result.
//
// Every memory has a registered read, so synthesis maps the weight ROMs and histories to block
// RAM. ROMs are initialised by $readmemh in a form synthesis keeps, so the netlist carries the
// network. Row files: Q1 140 lines of 128 int8 (element 0 in the low byte), Q2 128 lines of 128
// int8, Q3 128 lines of 20 int8; D1 140 and D2 128 lines of a delay 0..30.

`timescale 1ns / 1ps

module sc_shd_int_core #(
    parameter integer FRAC_BITS = 0,
    parameter integer THETA1    = 1,
    parameter integer THETA2    = 1,
    parameter integer V1_WIDTH  = 22,
    parameter integer V2_WIDTH  = 17,
    parameter integer ZERO_SKIP = 1,
    parameter         Q1_FILE   = "q1.hex",
    parameter         Q2_FILE   = "q2.hex",
    parameter         Q3_FILE   = "q3.hex",
    parameter         D1_FILE   = "d1.hex",
    parameter         D2_FILE   = "d2.hex"
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [15:0]       t_orig,
    input  wire [140*4-1:0]  frame,         // counts of input step t, channel i at [4i+3:4i]
    input  wire              frame_valid,
    output wire              frame_ready,
    output reg               busy,
    output reg               done,
    output reg               error,
    output reg  [20*32-1:0]  out_packed,    // class k at [32k+31:32k], two's complement
    output reg  [31:0]       cycles,        // clocks of the last sample
    output reg  [31:0]       row_reads      // Q1 + Q2 + Q3 row reads of the last sample
);
    localparam integer NI = 140, NH = 128, NO = 20;
    localparam integer T_MAX = 1024;
    localparam integer HALF = 15;
    localparam integer A1W = 20;   // |acc1| <= 140 * 15 * 127 = 266 700 < 2^19
    localparam integer A2W = 15;   // |acc2| <= 128 * 127 = 16 256 < 2^14

    // ------------------------------------------------------------------ ROMs
    reg [NH*8-1:0] q1_rom [0:NI-1];
    reg [NH*8-1:0] q2_rom [0:NH-1];
    reg [NO*8-1:0] q3_rom [0:NH-1];
    reg [4:0]      d1_rom [0:NI-1];
    reg [4:0]      d2_rom [0:NH-1];
    initial begin
        $readmemh(Q1_FILE, q1_rom);
        $readmemh(Q2_FILE, q2_rom);
        $readmemh(Q3_FILE, q3_rom);
        $readmemh(D1_FILE, d1_rom);
        $readmemh(D2_FILE, d2_rom);
    end

    // ------------------------------------------------------------------ control
    localparam [3:0] S_IDLE = 0, S_WAIT = 1, S_FILL = 2, S_L1 = 3, S_LIF1 = 4, S_L2 = 5,
                     S_LIF2 = 6, S_L3 = 7, S_NEXT = 8;
    reg [3:0]   state;
    reg [15:0]  t_len;
    reg [10:0]  t;              // logical step, t < T + 30 <= 1054
    reg [7:0]   idx;            // issue index of the current phase
    reg [NI*4-1:0] frame_r;

    wire [4:0] slot     = t[4:0];
    wire       in_input = ({5'd0, t} < t_len);
    wire       in_l1    = ({5'd0, t} < t_len + HALF);

    assign frame_ready = (state == S_WAIT) && in_input;

    // ------------------------------------------------------------------ input history H1
    reg  [3:0]  h1 [0:32*NI-1];
    reg  [3:0]  h1_q;
    wire        h1_we    = (state == S_FILL);
    wire [12:0] h1_waddr = slot * NI + idx;
    wire [4:0]  d1_i     = d1_rom[idx];
    wire [4:0]  rd1_slot = slot - d1_i;
    wire [12:0] h1_raddr = rd1_slot * NI + idx;
    always @(posedge clk) begin
        if (h1_we) h1[h1_waddr] <= frame_r[4*idx +: 4];
        h1_q <= h1[h1_raddr];
    end

    // ------------------------------------------------------------------ spike history H2
    reg  [NH-1:0] h2 [0:31];
    reg  [NH-1:0] h2_q;
    wire [NH-1:0] s1_vec;
    wire          h2_we    = (state == S_LIF1);
    wire [4:0]    d2_i     = d2_rom[idx[6:0]];
    wire [4:0]    rd2_slot = slot - d2_i;
    always @(posedge clk) begin
        if (h2_we) h2[slot] <= s1_vec;
        h2_q <= h2[rd2_slot];
    end

    // ------------------------------------------------------------------ weight rows
    // Stage 0 issues index idx; stage 1 decides the row read; stage 2 accumulates.
    reg        p0_v, p0_g;
    reg [7:0]  p0_idx;
    reg        p1_v;
    reg [3:0]  p1_y;
    wire [3:0] st1_y  = (state == S_L1) ? (p0_g ? h1_q : 4'd0)
                      : (state == S_L2) ? {3'd0, p0_g & h2_q[p0_idx[6:0]]}
                      :                   {3'd0, p0_g};
    wire       st1_en = p0_v && (ZERO_SKIP == 0 || st1_y != 0);

    reg [NH*8-1:0] q1_row, q2_row;
    reg [NO*8-1:0] q3_row;
    always @(posedge clk) begin
        if (st1_en && state == S_L1) q1_row <= q1_rom[p0_idx];
        if (st1_en && state == S_L2) q2_row <= q2_rom[p0_idx[6:0]];
        if (st1_en && state == S_L3) q3_row <= q3_rom[p0_idx[6:0]];
    end

    // ------------------------------------------------------------------ control strobes
    // The FSM below raises one strobe per datapath action; each lane acts on it in the same clock.
    wire start_ok   = (state == S_IDLE) && start && t_orig != 0 && t_orig <= T_MAX;
    wire clr_acc1   = (state == S_FILL) && (idx == NI - 1);
    wire add_acc1   = (state == S_L1) && p1_v;
    wire lif1       = (state == S_LIF1);
    wire add_acc2   = (state == S_L2) && p1_v && p1_y[0];
    wire lif2       = (state == S_LIF2);
    wire add_out    = (state == S_L3) && p1_v && p1_y[0];

    // ------------------------------------------------------------------ neuron lanes
    reg  [NH-1:0] s2;
    wire [NH-1:0] s2_vec;
    genvar g;
    generate
        for (g = 0; g < NH; g = g + 1) begin : lane
            reg  signed [A1W-1:0]      acc1;
            reg  signed [A2W-1:0]      acc2;
            reg  signed [V1_WIDTH-1:0] v1;
            reg  signed [V2_WIDTH-1:0] v2;
            wire signed [V1_WIDTH-1:0] a1 = acc1;
            wire signed [V2_WIDTH-1:0] a2 = acc2;
            wire signed [V1_WIDTH-1:0] u1 = v1 - (v1 >>> 2) + (a1 <<< FRAC_BITS);
            wire signed [V2_WIDTH-1:0] u2 = v2 - (v2 >>> 2) + (a2 <<< FRAC_BITS);
            // y * w for a 4-bit unsigned y, as a shift-add (no multiplier).
            wire signed [A1W-1:0] w1 = $signed(q1_row[8*g +: 8]);
            wire signed [A1W-1:0] prod1 = (p1_y[0] ? w1 : 0) + (p1_y[1] ? (w1 <<< 1) : 0)
                                        + (p1_y[2] ? (w1 <<< 2) : 0) + (p1_y[3] ? (w1 <<< 3) : 0);
            wire signed [A2W-1:0] w2 = $signed(q2_row[8*g +: 8]);
            assign s1_vec[g] = in_l1 && (u1 >= THETA1);
            assign s2_vec[g] = (u2 >= THETA2);
            always @(posedge clk) begin
                if (start_ok) begin
                    v1 <= 0;
                    v2 <= 0;
                end else begin
                    if (lif1) v1 <= s1_vec[g] ? {V1_WIDTH{1'b0}} : u1;
                    if (lif2) v2 <= s2_vec[g] ? {V2_WIDTH{1'b0}} : u2;
                end
                if (clr_acc1)      acc1 <= 0;
                else if (add_acc1) acc1 <= acc1 + prod1;
                if (lif1)          acc2 <= 0;
                else if (add_acc2) acc2 <= acc2 + w2;
            end
        end
    endgenerate

    // ------------------------------------------------------------------ output lanes
    wire [NO*32-1:0] out_now;
    generate
        for (g = 0; g < NO; g = g + 1) begin : cls
            reg signed [31:0] acc;
            always @(posedge clk) begin
                if (start_ok)     acc <= 0;
                else if (add_out) acc <= acc + $signed(q3_row[8*g +: 8]);
            end
            assign out_now[32*g +: 32] = acc;
        end
    endgenerate

    // Phase lengths including the two-stage pipeline drain.
    wire [7:0] last_issue = (state == S_L1) ? NI - 1 : NH - 1;
    wire [7:0] phase_end  = last_issue + 2;

    // ------------------------------------------------------------------ FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; error <= 1'b0;
            t <= 0; idx <= 0; t_len <= 0; cycles <= 0; row_reads <= 0; out_packed <= 0;
            p0_v <= 1'b0; p0_g <= 1'b0; p0_idx <= 0; p1_v <= 1'b0; p1_y <= 0; s2 <= 0;
            frame_r <= 0;
        end else begin
            done <= 1'b0;
            if (busy) cycles <= cycles + 1;
            p0_v <= 1'b0;
            p1_v <= 1'b0;
            case (state)
            S_IDLE: if (start) begin
                if (!start_ok) begin
                    error <= 1'b1;
                end else begin
                    error <= 1'b0; t_len <= t_orig; t <= 0; busy <= 1'b1;
                    cycles <= 0; row_reads <= 0;
                    state <= S_WAIT;
                end
            end
            S_WAIT: begin
                if (!in_input) begin
                    frame_r <= 0; idx <= 0; state <= S_FILL;
                end else if (frame_valid) begin
                    frame_r <= frame; idx <= 0; state <= S_FILL;
                end
            end
            S_FILL: begin
                if (idx == NI - 1) begin
                    idx <= 0;
                    state <= in_l1 ? S_L1 : S_LIF1;
                end else idx <= idx + 1;
            end
            S_L1, S_L2, S_L3: begin
                // Stage 0: issue index idx.
                if (idx <= last_issue) begin
                    p0_v <= 1'b1; p0_idx <= idx;
                    p0_g <= (state == S_L1) ? (t >= d1_i) : (state == S_L2) ? (t >= d2_i) : s2[idx[6:0]];
                end
                // Stage 1: the row read is enabled by st1_en; carry the value to stage 2.
                p1_v <= st1_en; p1_y <= st1_y;
                if (st1_en) row_reads <= row_reads + 1;
                // Stage 2 runs in the lanes (add_acc1, add_acc2, add_out).
                if (idx == phase_end) begin
                    idx <= 0;
                    case (state)
                    S_L1:    state <= S_LIF1;
                    S_L2:    state <= S_LIF2;
                    default: state <= S_NEXT;
                    endcase
                end else idx <= idx + 1;
            end
            S_LIF1: begin
                idx <= 0; state <= S_L2;
            end
            S_LIF2: begin
                s2 <= s2_vec;
                idx <= 0; state <= S_L3;
            end
            S_NEXT: begin
                if ({5'd0, t} == t_len + 2 * HALF - 1) begin
                    out_packed <= out_now;
                    busy <= 1'b0; done <= 1'b1; state <= S_IDLE;
                end else begin
                    t <= t + 1; state <= S_WAIT;
                end
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
