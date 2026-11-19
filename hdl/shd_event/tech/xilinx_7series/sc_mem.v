// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Xilinx 7-Series block-RAM-inferring sc_mem implementation
//
// Behavioural single-port bank/column memory matching tech/generic/sc_mem.v cycle-for-cycle
// at its ports while inferring Xilinx 7-Series RAMB36E1 / RAMB18E1 primitives in Vivado and
// Yosys synth_xilinx (registered read per bank/column with column rmask enable).
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

    wire [C*BITS-1:0] wpad      = {{(C*BITS-WIDTH){1'b0}}, wdata};
    wire [31:0]       addr_ext  = {{(32-LAW){1'b0}}, addr};
    wire [31:0]       bank_full = (K > 1) ? (addr_ext / WORDS) : 32'd0;
    wire [31:0]       row_full  = addr_ext % WORDS;
    wire [KW-1:0]     bank      = bank_full[KW-1:0];
    wire [MAW-1:0]    row       = row_full[MAW-1:0];

    reg [KW-1:0] rbank = {KW{1'b0}};
    always @(posedge clk) begin
        if (re) begin
            rbank <= bank;
        end
    end

    wire [K*C*BITS-1:0] all_out;
    genvar b, c;
    generate
        for (b = 0; b < K; b = b + 1) begin : bk
            for (c = 0; c < C; c = c + 1) begin : col
                (* ram_style = "block" *) reg [BITS-1:0] ram [0:WORDS-1];
                reg [BITS-1:0] ram_dout_r = {BITS{1'b0}};
                reg [BITS-1:0] bypass_r   = {BITS{1'b0}};
                reg            use_byp_r  = 1'b0;
`ifndef SYNTHESIS
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < WORDS; init_idx = init_idx + 1) begin
                        ram[init_idx] = {BITS{1'b0}};
                    end
                end
`endif
                wire sel_w      = we && (bank == b[KW-1:0]);
                wire sel_r      = re && (bank == b[KW-1:0]) && rmask[c];
                wire sel_r_bram = sel_r && !sel_w;
                always @(posedge clk) begin
                    if (sel_w) begin
                        ram[row] <= wpad[c*BITS +: BITS];
                    end
                    if (sel_r_bram) begin
                        ram_dout_r <= ram[row];
                    end
                    if (sel_r) begin
                        use_byp_r <= sel_w;
                        if (sel_w) begin
                            bypass_r <= wpad[c*BITS +: BITS];
                        end
                    end
                end
                assign all_out[(b*C+c)*BITS +: BITS] = use_byp_r ? bypass_r : ram_dout_r;
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
