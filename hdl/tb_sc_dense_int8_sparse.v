// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for sc_dense_int8_sparse.v
//
// File-based cosim driver for arbitrary IN_F × OUT_F dense layers.
// IN_F and OUT_F are local parameters that can be overridden at compile
// time via `iverilog -Ptb_sc_dense_int8_sparse.IN_F=...` so a single
// testbench source covers all SHD layer sizes (140×128, 128×128, 128×20)
// plus tiny smoke-test sizes (e.g. 8×4).
//
// Cosim contract:
//   1. The Python harness writes `weights.hex` (one signed-int8 per line,
//      row-major OUT_F × IN_F) into a working directory.
//   2. The harness writes `spikes.hex` (one IN_F-bit hex value per line,
//      N samples).
//   3. The harness writes the DUT scale (Q16.16 signed integer) into
//      `scale.txt`.
//   4. The harness runs `vvp tb +N=<n>` from inside the working directory.
//   5. The testbench drives the DUT one sample per cycle, captures the
//      registered output bus on the next rising edge, and writes one row
//      per sample to `outputs.txt`:
//
//        # step <out0> <out1> ... <out_{OUT_F-1}>
//
//      where each value is a signed decimal Q8.8 integer.
//
// Plus-args:
//   +N=<num_samples>          number of stimulus samples to drive

`timescale 1ns / 1ps

module tb_sc_dense_int8_sparse;

    // Compile-time-overridable layer dimensions
    parameter integer IN_F  = 140;
    parameter integer OUT_F = 128;

    localparam integer MAX_SAMPLES = 256;

    reg                 clk;
    reg                 rst_n;
    reg signed [31:0]   scale_q16_16;
    reg [IN_F-1:0]      spikes_in;
    wire [OUT_F*16-1:0] out_q88_packed;

    // Stimulus storage: N samples × IN_F bits
    reg [IN_F-1:0] stim_mem [0:MAX_SAMPLES-1];

    integer n_samples;
    integer scale_int;
    integer scale_fd;
    integer out_fd;
    integer i;
    integer j;
    reg signed [15:0] out_word;

    sc_dense_int8_sparse #(
        .IN_FEATURES (IN_F),
        .OUT_FEATURES(OUT_F),
        .WEIGHT_FILE ("weights.hex")
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .scale_q16_16   (scale_q16_16),
        .spikes_in      (spikes_in),
        .out_q88_packed (out_q88_packed)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        n_samples = 0;
        if (!$value$plusargs("N=%d", n_samples)) n_samples = 0;
        if (n_samples <= 0 || n_samples > MAX_SAMPLES) begin
            $display("ERROR: invalid +N=%0d (must be 1..%0d)",
                     n_samples, MAX_SAMPLES);
            $finish;
        end

        // Read scale from a small text file (one decimal int)
        scale_fd = $fopen("scale.txt", "r");
        if (scale_fd == 0) begin
            $display("ERROR: cannot open scale.txt");
            $finish;
        end
        if ($fscanf(scale_fd, "%d", scale_int) != 1) begin
            $display("ERROR: cannot parse scale.txt");
            $finish;
        end
        $fclose(scale_fd);

        // Load stimulus
        $readmemh("spikes.hex", stim_mem, 0, n_samples - 1);

        // Open output file
        out_fd = $fopen("outputs.txt", "w");
        if (out_fd == 0) begin
            $display("ERROR: cannot open outputs.txt");
            $finish;
        end
        $fwrite(out_fd, "# step out0 out1 ... out%0d\n", OUT_F - 1);

        // Reset hold + first-sample handover (same race-free idiom as
        // the Vmin_LIF testbench).
        scale_q16_16 = scale_int;
        rst_n        = 1'b0;
        spikes_in    = {IN_F{1'b0}};
        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n     = 1'b1;
        spikes_in = stim_mem[0];
        @(posedge clk);
        #1;
        $fwrite(out_fd, "%0d", 0);
        for (j = 0; j < OUT_F; j = j + 1) begin
            out_word = $signed(out_q88_packed[16*j +: 16]);
            $fwrite(out_fd, " %0d", out_word);
        end
        $fwrite(out_fd, "\n");

        for (i = 1; i < n_samples; i = i + 1) begin
            @(negedge clk);
            spikes_in = stim_mem[i];
            @(posedge clk);
            #1;
            $fwrite(out_fd, "%0d", i);
            for (j = 0; j < OUT_F; j = j + 1) begin
                out_word = $signed(out_q88_packed[16*j +: 16]);
                $fwrite(out_fd, " %0d", out_word);
            end
            $fwrite(out_fd, "\n");
        end

        $fclose(out_fd);
        $display("tb_sc_dense_int8_sparse: drove %0d samples (IN=%0d OUT=%0d)",
                 n_samples, IN_F, OUT_F);
        $finish;
    end

    initial begin
        if ($test$plusargs("VCD")) begin
            $dumpfile("tb_sc_dense_int8_sparse.vcd");
            $dumpvars(0, tb_sc_dense_int8_sparse);
        end
    end

endmodule
