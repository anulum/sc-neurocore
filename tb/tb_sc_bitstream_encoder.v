// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// tb/tb_sc_bitstream_encoder.v
`timescale 1ns / 1ps

module tb_sc_bitstream_encoder;

localparam DATA_WIDTH = 16;
localparam LFSR_WIDTH = 16;

reg clk;
reg rst_n;
reg [DATA_WIDTH-1:0] x_value;
reg [31:0] t_index;
wire bit_out;

sc_bitstream_encoder #(
    .DATA_WIDTH(DATA_WIDTH),
    .LFSR_WIDTH(LFSR_WIDTH)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .x_value(x_value),
    .t_index(t_index),
    .bit_out(bit_out)
);

initial begin
    clk = 0;
    forever #5 clk = ~clk; // 100 MHz clock
end

initial begin
    $monitor("time=%0t, clk=%b, rst_n=%b, x_value=%h, t_index=%d, bit_out=%b",
             $time, clk, rst_n, x_value, t_index, bit_out);
    rst_n = 0;
    x_value = 16'h8000; // 0.5
    t_index = 0;
    #50;
    rst_n = 1;

    #1000;

    $finish;
end

endmodule
