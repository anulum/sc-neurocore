// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for sc_vmin_lif_neuron.v
//
// File-based cosim driver:
//   1. Reads N input current samples (Q8.8, signed) from `INPUT_FILE` via
//      $readmemh. One token per line, lower 16 bits of each line treated
//      as the signed Q8.8 sample.
//   2. Drives one sample per clock cycle into the DUT.
//   3. Captures (spike, v_out) at every cycle and writes them to
//      `OUTPUT_FILE` as one decimal pair per line: `<spike> <v_out_decimal>`.
//   4. Stops simulation when all samples have been driven and 2 trailing
//      flush cycles have elapsed (covers the 1-cycle output register).
//
// The expected vectors come from tools/gen_vmin_lif_lut.py vmin_lif_step_q88(),
// the bit-true Python reference. The Python harness in
// tools/cosim_vmin_lif_verilog.py generates the input file, runs iverilog,
// loads the output file, and asserts spike-by-spike equality.
//
// Plus-args:
//   +N=<num_samples>          number of stimulus samples to drive
//   +INPUT_FILE=<path>        $readmemh source file with Q8.8 currents
//   +OUTPUT_FILE=<path>       text file written by $fwrite
//
// Defaults are set so the testbench can be run standalone for smoke
// testing without external files.

`timescale 1ns / 1ps

module tb_vmin_lif_neuron;

    // Maximum number of samples we ever store in memory. The Python
    // harness keeps stimulus sizes far below this for fast iteration.
    localparam integer MAX_SAMPLES = 4096;

    reg                 clk;
    reg                 rst_n;
    reg  signed [15:0]  x_in;
    wire                spike;
    wire signed [15:0]  v_out;

    // Stimulus storage (16 bits wide, MAX_SAMPLES deep)
    reg [15:0] stim_mem [0:MAX_SAMPLES-1];

    // Plus-arg state
    integer n_samples;
    reg [1023:0] input_file;
    reg [1023:0] output_file;
    integer out_fd;
    integer i;

    // Instantiate the DUT with default parameters (matching Vmin_LIFNode
    // tau=4, v_thr=1, v_reset=0, v_inf=-5, beta=1).
    sc_vmin_lif_neuron #(
        .DECAY    (16'sd192),
        .V_THRESH (16'sd256),
        .V_RESET  (16'sd0),
        .V_INF    (-16'sd1280)
    ) dut (
        .clk   (clk),
        .rst_n (rst_n),
        .x_in  (x_in),
        .spike (spike),
        .v_out (v_out)
    );

    // 100 MHz clock (10 ns period)
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        // Defaults
        n_samples   = 0;
        input_file  = "";
        output_file = "";

        // Plus-arg parsing
        if (!$value$plusargs("N=%d", n_samples)) begin
            n_samples = 0;
        end
        if (!$value$plusargs("INPUT_FILE=%s", input_file)) begin
            input_file = "tb_vmin_lif_input.hex";
        end
        if (!$value$plusargs("OUTPUT_FILE=%s", output_file)) begin
            output_file = "tb_vmin_lif_output.txt";
        end

        if (n_samples <= 0 || n_samples > MAX_SAMPLES) begin
            $display("ERROR: invalid +N=%0d (must be 1..%0d)", n_samples, MAX_SAMPLES);
            $finish;
        end

        // Load stimulus
        $readmemh(input_file, stim_mem, 0, n_samples - 1);

        // Open output file
        out_fd = $fopen(output_file, "w");
        if (out_fd == 0) begin
            $display("ERROR: cannot open output file %0s", output_file);
            $finish;
        end
        $fwrite(out_fd, "# step spike v_out\n");

        // Reset sequence — hold rst_n=0 for several cycles so v_reg latches 0.
        // We MUST release rst_n in the same negedge window in which we apply
        // sample[0], because as soon as rst_n=1 the next posedge would otherwise
        // execute one "idle" step on x_in=0 — and the vmin floor would push
        // v_reg from 0 toward softplus(5)-5 ≈ Q8.8 2, breaking bit-true match
        // with the Python reference (which starts at v=0 and immediately
        // consumes sample[0]).
        rst_n = 1'b0;
        x_in  = 16'sd0;
        repeat (3) @(posedge clk);

        // Move into the first sample window: set up sample[0] AND release
        // reset between posedges, so the very next posedge computes
        //   v_next = step(v_reg=0, x_in=stim[0]).
        @(negedge clk);
        rst_n = 1'b1;
        x_in  = stim_mem[0];
        @(posedge clk);
        #1;
        $fwrite(out_fd, "%0d %0d %0d\n", 0, spike, v_out);

        // Remaining samples — one per clock cycle, sampling after the
        // registered outputs update.
        for (i = 1; i < n_samples; i = i + 1) begin
            @(negedge clk);
            x_in = stim_mem[i];
            @(posedge clk);
            #1;
            $fwrite(out_fd, "%0d %0d %0d\n", i, spike, v_out);
        end

        // Flush: 2 trailing cycles in case of any pipeline lag
        x_in = 16'sd0;
        @(posedge clk);
        @(posedge clk);

        $fclose(out_fd);
        $display("tb_vmin_lif_neuron: drove %0d samples, output -> %0s",
                 n_samples, output_file);
        $finish;
    end

    // VCD dump for debugging (only when +VCD plusarg is set)
    initial begin
        if ($test$plusargs("VCD")) begin
            $dumpfile("tb_vmin_lif_neuron.vcd");
            $dumpvars(0, tb_vmin_lif_neuron);
        end
    end

endmodule
