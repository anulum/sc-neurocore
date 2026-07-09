// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for SC LIF Neuron

// hdl/tb_sc_lif_neuron.v
//
// Co-simulation testbench for sc_lif_neuron.
//
// Reads stimuli from "stimuli.txt" (one line per clock cycle):
//   leak_k  gain_k  I_t  noise_in   (all as decimal integers, Q8.8)
//
// Writes results to "results_verilog.txt" (one line per cycle):
//   spike  v_out   (decimal integers)
//
// Compare results_verilog.txt against the Python FixedPointLIFNeuron
// model to verify bit-true equivalence.

`timescale 1ns / 1ps

module tb_sc_lif_neuron;

    parameter integer DATA_WIDTH = 16;
    parameter integer FRACTION   = 8;
    parameter integer NUM_STEPS  = 1000;

    // Clock and reset
    reg clk;
    reg rst_n;

    // Neuron I/O
    reg signed [DATA_WIDTH-1:0] leak_k;
    reg signed [DATA_WIDTH-1:0] gain_k;
    reg signed [DATA_WIDTH-1:0] I_t;
    reg signed [DATA_WIDTH-1:0] noise_in;
    wire                        spike_out;
    wire signed [DATA_WIDTH-1:0] v_out;

    // Instantiate DUT
    sc_lif_neuron #(
        .DATA_WIDTH       (DATA_WIDTH),
        .FRACTION         (FRACTION),
        .V_REST           (0),
        .V_RESET          (0),
        .V_THRESHOLD      (1 << FRACTION),   // 1.0 in Q8.8 = 256
        .REFRACTORY_PERIOD(2)
    ) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .leak_k   (leak_k),
        .gain_k   (gain_k),
        .I_t      (I_t),
        .noise_in (noise_in),
        .spike_out(spike_out),
        .v_out    (v_out)
    );

    // Clock generation: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // File handles
    integer stim_file;
    integer result_file;
    integer scan_ret;
    integer step;

    // Stimulus values (read as 32-bit, then cast)
    integer stim_leak, stim_gain, stim_it, stim_noise;

    initial begin
        // Open files
        stim_file = $fopen("stimuli.txt", "r");
        if (stim_file == 0) begin
            $display("ERROR: Cannot open stimuli.txt");
            $finish;
        end

        result_file = $fopen("results_verilog.txt", "w");
        if (result_file == 0) begin
            $display("ERROR: Cannot open results_verilog.txt for writing");
            $finish;
        end

        // Reset
        rst_n    = 0;
        leak_k   = 0;
        gain_k   = 0;
        I_t      = 0;
        noise_in = 0;

        // Hold reset for 2 cycles
        @(posedge clk);
        @(posedge clk);
        rst_n = 1;

        // Process each stimulus line
        for (step = 0; step < NUM_STEPS; step = step + 1) begin
            scan_ret = $fscanf(stim_file, "%d %d %d %d\n",
                               stim_leak, stim_gain, stim_it, stim_noise);
            if (scan_ret != 4) begin
                $display("INFO: End of stimuli at step %0d", step);
                step = NUM_STEPS; // break
            end else begin
                leak_k   = stim_leak[DATA_WIDTH-1:0];
                gain_k   = stim_gain[DATA_WIDTH-1:0];
                I_t      = stim_it[DATA_WIDTH-1:0];
                noise_in = stim_noise[DATA_WIDTH-1:0];

                @(posedge clk);
                // Sample outputs after the rising edge
                #1;
                $fwrite(result_file, "%0d %0d\n", spike_out, v_out);
            end
        end

        $fclose(stim_file);
        $fclose(result_file);
        $display("Co-simulation complete. Results written to results_verilog.txt");
        $finish;
    end

endmodule
