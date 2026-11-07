// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench streaming SHD samples through sc_shd_int_core
//
// Writes the 20 accumulators and the two counters of each sample to outputs.txt, or
// "REFUSED <error>" for a refused length. Define GATE_LEVEL to drive a synthesised netlist,
// which carries its parameters and ROM contents.
//
// stimulus.hex: first word = number of samples; per sample one word T (or an invalid T, which
// must be refused), then T lines of 140 hex nibbles (channel 139 first, channel 0 last).
// Driven by tools/cosim_shd_int_core.py.

`timescale 1ns / 1ps

module tb_sc_shd_int_core;
    parameter integer FRAC_BITS = 0;
    parameter integer THETA1 = 1;
    parameter integer THETA2 = 1;
    parameter integer V1_WIDTH = 22;
    parameter integer V2_WIDTH = 17;
    parameter integer ZERO_SKIP = 1;
    parameter integer MAX_WORDS = 4000000;

    reg clk = 0, rst_n = 0, start = 0, frame_valid = 0;
    reg [15:0] t_orig = 0;
    reg [559:0] frame = 0;
    wire frame_ready, busy, done, error;
    wire [639:0] out_packed;
    wire [31:0] cycles, row_reads;

`ifdef GATE_LEVEL
    // The synthesised netlist has its parameters and ROM contents built in.
    sc_shd_int_core dut (
`else
    sc_shd_int_core #(.FRAC_BITS(FRAC_BITS), .THETA1(THETA1), .THETA2(THETA2), .V1_WIDTH(V1_WIDTH),
                      .V2_WIDTH(V2_WIDTH), .ZERO_SKIP(ZERO_SKIP)) dut (
`endif
        .clk(clk), .rst_n(rst_n), .start(start), .t_orig(t_orig), .frame(frame),
        .frame_valid(frame_valid), .frame_ready(frame_ready), .busy(busy), .done(done),
        .error(error), .out_packed(out_packed), .cycles(cycles), .row_reads(row_reads));

    always #5 clk = ~clk;

    reg [559:0] stim [0:MAX_WORDS-1];
    integer fd, n, s, p, f, k, ptr;

    initial begin
        $readmemh("stimulus.hex", stim);
        fd = $fopen("outputs.txt", "w");
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        n = stim[0]; ptr = 1;
        for (s = 0; s < n; s = s + 1) begin
            t_orig = stim[ptr]; ptr = ptr + 1;
            @(negedge clk); start = 1; @(negedge clk); start = 0;
            if (t_orig == 0 || t_orig > 1024) begin
                @(negedge clk);
                $fwrite(fd, "REFUSED %0d\n", error);
                $fflush(fd);
            end else begin
                f = 0;
                while (!done) begin
                    @(negedge clk);
                    if (frame_ready && f < t_orig) begin
                        frame = stim[ptr]; frame_valid = 1;
                    end else frame_valid = 0;
                    @(posedge clk); #1;
                    if (frame_valid && frame_ready === 1'b0) begin
                        f = f + 1; ptr = ptr + 1; frame_valid = 0;
                    end
                end
                for (k = 0; k < 20; k = k + 1)
                    $fwrite(fd, "%0d ", $signed(out_packed[32*k +: 32]));
                $fwrite(fd, "| %0d %0d\n", cycles, row_reads);
                $fflush(fd);
            end
        end
        $fclose(fd);
        $finish;
    end
endmodule
