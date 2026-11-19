// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — First-party behavioural 1P SRAM macro model (SHAPE 0..19)
//
// Behavioural contract matching the 20 single-port macro shapes in sc_shapes.vh:
//   1. Power-up state: all 2**AW words and the registered output dout start at 0.
//   2. Write: on posedge clk with men && wen, shapes 1..18 apply bit-enable mask bm
//      ((mem[addr] & ~bm) | (din & bm)), while shapes 0 and 19 have no bit-mask hardware
//      and write the full word din regardless of bm.
//   3. Read: on posedge clk with men && !wen && ren, dout registers mem[addr].
//   4. Read-during-write at one address (men && wen && ren): write-through — dout
//      registers the newly written post-mask word on that same clock edge.
//   5. Hold: when !men or !ren (including write-only cycles with ren == 0), dout retains
//      its previous registered value.
//   6. Bounds: SHAPE outside 0..19 reports an error and finishes elaboration/simulation.
`timescale 1ns / 1ps
module sc_sram_macro #(
    parameter integer SHAPE = 0,
    parameter integer AW    = 6,
    parameter integer BW    = 16
) (
    input  wire          clk,
    input  wire          men,
    input  wire          wen,
    input  wire          ren,
    input  wire [AW-1:0] addr,
    input  wire [BW-1:0] din,
    input  wire [BW-1:0] bm,
    output wire [BW-1:0] dout
);
    localparam integer DEPTH    = (1 << AW);
    localparam integer HAS_MASK = (SHAPE >= 1 && SHAPE <= 18) ? 1 : 0;

    generate
        if (SHAPE < 0 || SHAPE > 19) begin : invalid_shape
            initial begin
                $display("sc_sram_macro: SHAPE %0d outside 0..19", SHAPE);
                $finish;
            end
            assign dout = {BW{1'b0}};
        end else begin : valid_shape
            reg [BW-1:0] mem [0:DEPTH-1];
            reg [BW-1:0] dout_r;
            integer idx;
            initial begin
                dout_r = {BW{1'b0}};
                for (idx = 0; idx < DEPTH; idx = idx + 1) begin
                    mem[idx] = {BW{1'b0}};
                end
            end

            wire [BW-1:0] eff_bm = HAS_MASK ? bm : {BW{1'b1}};
            wire [BW-1:0] wr_val = (mem[addr] & ~eff_bm) | (din & eff_bm);

            always @(posedge clk) begin
                if (men && wen) begin
                    mem[addr] <= wr_val;
                    if (ren) begin
                        dout_r <= wr_val;
                    end
                end else if (men && ren) begin
                    dout_r <= mem[addr];
                end
            end

            assign dout = dout_r;
        end
    endgenerate
endmodule
