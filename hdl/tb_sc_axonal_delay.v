// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for sc_axonal_delay.v
//
// File-based cosim driver:
//   1. Read N spike samples (one bit per line, '0' or '1') from
//      `INPUT_FILE`.
//   2. Drive the DUT with one spike per cycle and a fixed read_offset
//      (provided via +READ_OFFSET=<n>).
//   3. Capture spike_out at the same cycle the input is presented (the
//      Python reference is logically combinational — the same cycle
//      observes the new write).
//   4. Write step + spike_out to `OUTPUT_FILE`, one row per sample.
//
// Plus-args:
//   +N=<num_samples>          number of stimulus samples to drive
//   +READ_OFFSET=<n>          fixed delay tap (0..DEPTH-1)
//   +INPUT_FILE=<path>        $readmemh source file (one bit per line)
//   +OUTPUT_FILE=<path>       text file written by $fwrite

`timescale 1ns / 1ps

module tb_sc_axonal_delay;

    localparam integer MAX_SAMPLES = 4096;
    localparam integer DEPTH       = 31;
    localparam integer PTR_WIDTH   = 5;

    reg                 clk;
    reg                 rst_n;
    reg                 spike_in;
    reg [PTR_WIDTH-1:0] read_offset;
    wire                spike_out;

    reg [0:0] stim_mem [0:MAX_SAMPLES-1];

    integer n_samples;
    integer offset_arg;
    reg [1023:0] input_file;
    reg [1023:0] output_file;
    integer out_fd;
    integer i;

    sc_axonal_delay #(
        .DEPTH(DEPTH),
        .PTR_WIDTH(PTR_WIDTH)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .spike_in    (spike_in),
        .read_offset (read_offset),
        .spike_out   (spike_out)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        n_samples   = 0;
        offset_arg  = 0;
        input_file  = "";
        output_file = "";

        if (!$value$plusargs("N=%d", n_samples)) n_samples = 0;
        if (!$value$plusargs("READ_OFFSET=%d", offset_arg)) offset_arg = 0;
        if (!$value$plusargs("INPUT_FILE=%s", input_file))
            input_file = "tb_axdelay_input.hex";
        if (!$value$plusargs("OUTPUT_FILE=%s", output_file))
            output_file = "tb_axdelay_output.txt";

        if (n_samples <= 0 || n_samples > MAX_SAMPLES) begin
            $display("ERROR: invalid +N=%0d", n_samples);
            $finish;
        end
        if (offset_arg < 0 || offset_arg >= DEPTH) begin
            $display("ERROR: invalid +READ_OFFSET=%0d (must be 0..%0d)",
                     offset_arg, DEPTH - 1);
            $finish;
        end

        $readmemh(input_file, stim_mem, 0, n_samples - 1);

        out_fd = $fopen(output_file, "w");
        if (out_fd == 0) begin
            $display("ERROR: cannot open output file %0s", output_file);
            $finish;
        end
        $fwrite(out_fd, "# step spike_out\n");

        // Reset hold — same idiom as tb_vmin_lif_neuron.v: drop reset
        // between posedges so the very next clock edge consumes sample[0]
        // while the buffer is still empty (head=0).
        rst_n       = 1'b0;
        spike_in    = 1'b0;
        read_offset = offset_arg[PTR_WIDTH-1:0];
        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n    = 1'b1;
        spike_in = stim_mem[0][0];
        // The output for sample 0 is combinational: it reflects the new
        // write at head=0 with the configured read_offset.
        #1;
        $fwrite(out_fd, "%0d %0d\n", 0, spike_out);

        // Latch sample 0 into the buffer.
        @(posedge clk);

        for (i = 1; i < n_samples; i = i + 1) begin
            @(negedge clk);
            spike_in = stim_mem[i][0];
            #1;
            $fwrite(out_fd, "%0d %0d\n", i, spike_out);
            @(posedge clk);
        end

        $fclose(out_fd);
        $display("tb_sc_axonal_delay: drove %0d samples (offset=%0d) -> %0s",
                 n_samples, offset_arg, output_file);
        $finish;
    end

    initial begin
        if ($test$plusargs("VCD")) begin
            $dumpfile("tb_sc_axonal_delay.vcd");
            $dumpvars(0, tb_sc_axonal_delay);
        end
    end

endmodule
