// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Tiled single-port SRAM array over 1P macro blocks
//
// Provenance: imported from the group's earlier design study as part of Golden Reference #0.
// Splits WIDTH across C = ceil(WIDTH / BITS) column macros and DEPTH across
// K = ceil(DEPTH / WORDS) banks, enabling only columns selected by rmask on reads.
// Outside synthesis, counts enabled macro reads and writes per access.
`timescale 1ns / 1ps
module sc_mem #(
    parameter integer DEPTH = 32,
    parameter integer WIDTH = 64,
    parameter integer SHAPE = 0
)(
    clk, we, re, addr, wdata, rmask, rdata
);
`include "sc_shapes.vh"
    localparam integer WORDS = sc_shape_words(SHAPE);
    localparam integer BITS  = sc_shape_bits(SHAPE);
    localparam integer C     = (WIDTH + BITS - 1) / BITS;
    localparam integer K     = (DEPTH + WORDS - 1) / WORDS;
    localparam integer MAW   = $clog2(WORDS);
    localparam integer LAW   = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam integer KW    = (K > 1) ? $clog2(K) : 1;

    input  wire             clk;
    input  wire             we;
    input  wire             re;
    input  wire [LAW-1:0]   addr;
    input  wire [WIDTH-1:0] wdata;
    input  wire [C-1:0]     rmask;
    output wire [WIDTH-1:0] rdata;

    wire [C*BITS-1:0] wpad = wdata;   // zero-extended to whole columns
    wire [KW-1:0]     bank = (K > 1) ? addr / WORDS : 0;
    wire [MAW-1:0]    row  = addr % WORDS;
    reg  [KW-1:0]     rbank;
    always @(posedge clk) if (re) rbank <= bank;

    wire [K*C*BITS-1:0] all_out;
    genvar b, c;
    generate
        for (b = 0; b < K; b = b + 1) begin : bk
            for (c = 0; c < C; c = c + 1) begin : col
                wire sel_w = we && bank == b;
                wire sel_r = re && bank == b && rmask[c];
                sc_sram_macro #(.SHAPE(SHAPE), .AW(MAW), .BW(BITS)) m (
                    .clk(clk), .men(sel_w | sel_r), .wen(sel_w), .ren(sel_r), .addr(row),
                    .din(wpad[c*BITS +: BITS]), .bm({BITS{1'b1}}),
                    .dout(all_out[(b*C+c)*BITS +: BITS])
                );
            end
        end
    endgenerate
    wire [C*BITS-1:0] sel_out = all_out[rbank*C*BITS +: C*BITS];
    assign rdata = sel_out[WIDTH-1:0];

`ifndef SYNTHESIS
    integer reads = 0, writes = 0, i;
    always @(posedge clk) begin
        if (we) writes = writes + C;
        if (re) for (i = 0; i < C; i = i + 1) if (rmask[i]) reads = reads + 1;
    end
`endif
endmodule
