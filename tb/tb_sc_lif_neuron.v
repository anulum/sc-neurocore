// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

`timescale 1ns / 1ps

module tb_sc_lif_neuron;

    // Parameters
    parameter DATA_WIDTH = 16;
    parameter FRACTION = 8;
    parameter V_REST = 0;
    parameter V_RESET = 0;
    parameter V_THRESHOLD = (1 << FRACTION); // 1.0 in Q8
    parameter REFRACTORY_PERIOD = 2;

    // Inputs
    reg clk;
    reg rst_n;
    reg signed [DATA_WIDTH-1:0] leak_k;
    reg signed [DATA_WIDTH-1:0] gain_k;
    reg signed [DATA_WIDTH-1:0] I_t;
    reg signed [DATA_WIDTH-1:0] noise_in;

    // Outputs
    wire spike_out;
    wire signed [DATA_WIDTH-1:0] v_out;

    // File I/O
    integer input_file, output_file;
    integer scan_status;

    // Instantiate the Unit Under Test (UUT)
    sc_lif_neuron #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRACTION(FRACTION),
        .V_REST(V_REST),
        .V_RESET(V_RESET),
        .V_THRESHOLD(V_THRESHOLD),
        .REFRACTORY_PERIOD(REFRACTORY_PERIOD)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .leak_k(leak_k),
        .gain_k(gain_k),
        .I_t(I_t),
        .noise_in(noise_in),
        .spike_out(spike_out),
        .v_out(v_out)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period
    end

    // Test Stimulus
    initial begin
        // Initialize Inputs
        rst_n = 0;
        leak_k = 0;
        gain_k = 0;
        I_t = 0;
        noise_in = 0;

        // Open files
        input_file = $fopen("stimuli.txt", "r");
        if (input_file == 0) begin
            $display("Error: Could not open stimuli.txt");
            $finish;
        end

        output_file = $fopen("results_verilog.txt", "w");

        // Reset
        #20;
        rst_n = 1;

        // Read and Apply Loop
        while (!$feof(input_file)) begin
            @(negedge clk); // Apply inputs on negedge to be stable at posedge
            scan_status = $fscanf(input_file, "%d %d %d %d\n", leak_k, gain_k, I_t, noise_in);
            if (scan_status == 4) begin
                // Capture output just before next rising edge (or at posedge)
                // We'll write output corresponding to the result of THIS cycle
                @(posedge clk);
                #1; // Wait a tiny bit for propagation
                $fwrite(output_file, "%d %d\n", spike_out, v_out);
            end
        end

        $fclose(input_file);
        $fclose(output_file);
        $display("Simulation finished.");
        $finish;
    end

endmodule
`
