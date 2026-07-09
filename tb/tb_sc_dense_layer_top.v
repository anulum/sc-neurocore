// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// tb/tb_sc_dense_layer_core.v
`timescale 1ns / 1ps

module tb_sc_dense_layer_core;

localparam N_INPUTS = 3;
localparam N_NEURONS = 7;
localparam DATA_WIDTH = 16;
localparam FRACTION = 8;

reg clk;
reg rst_n;

// Config inputs (could be driven by AXI in real system)
reg         start_pulse;
reg [15:0]  x_input_fp [0:N_INPUTS-1];
reg [15:0]  weight_fp [0:N_INPUTS-1];
reg [15:0]  y_min_fp;
reg [15:0]  y_max_fp;
reg [31:0]  stream_len;

wire [DATA_WIDTH-1:0] I_t;
wire [N_NEURONS-1:0]  spikes;
wire                  running;
wire                  step_valid;
wire                  run_done;

integer i;

// DUT
sc_dense_layer_top #(
    .N_INPUTS(N_INPUTS),
    .N_NEURONS(N_NEURONS),
    .STREAM_LEN(2500),
    .DATA_WIDTH(DATA_WIDTH)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_pulse),
    .done(run_done),
    .x_inputs({x_input_fp[2], x_input_fp[1], x_input_fp[0]}),
    .weight_values({weight_fp[2], weight_fp[1], weight_fp[0]}),
    .y_min(y_min_fp),
    .y_max(y_max_fp),
    .spike_out(spikes),
    .spike_valid(step_valid)
);

// Simple spike counters in testbench
integer spike_count [0:N_NEURONS-1];

initial begin
    clk = 0;
    forever #5 clk = ~clk; // 100 MHz clock
end

initial begin
    $monitor("time=%0t, clk=%b, rst_n=%b, start_pulse=%b, run_done=%b, step_valid=%b, pre_bits_t=%b, w_bits_t=%b, post_bits_t=%b, I_t=%h, spikes=%b",
             $time, clk, rst_n, start_pulse, run_done, step_valid, dut.pre_bits_t, dut.w_bits_t, dut.post_bits_t, I_t, spikes);
    rst_n = 0;
    start_pulse = 0;
    for (i = 0; i < N_NEURONS; i = i + 1)
        spike_count[i] = 0;

    // Wait a bit
    #50;
    rst_n = 1;

    // Set config: pattern A, weights, etc.
    // Q8.8 encoding
    x_input_fp[0] = $rtoi(0.02 * (1<<FRACTION)); // ~0x005
    x_input_fp[1] = $rtoi(0.05 * (1<<FRACTION)); // ~0x00D
    x_input_fp[2] = $rtoi(0.08 * (1<<FRACTION)); // ~0x014

    weight_fp[0] = $rtoi(0.3 * (1<<FRACTION));
    weight_fp[1] = $rtoi(0.6 * (1<<FRACTION));
    weight_fp[2] = $rtoi(0.9 * (1<<FRACTION));

    y_min_fp = $rtoi(0.0 * (1<<FRACTION));
    y_max_fp = $rtoi(0.1 * (1<<FRACTION));

    stream_len = 32'd2500;

    // Start run
    @(posedge clk);
    start_pulse = 1;
    @(posedge clk);
    start_pulse = 0;

    // Count spikes while running
    while (!run_done) begin
        @(posedge clk);
        if (step_valid) begin
            for (i = 0; i < N_NEURONS; i = i + 1)
                if (spikes[i])
                    spike_count[i] = spike_count[i] + 1;
        end
    end

    $display("Run done.");
    for (i = 0; i < N_NEURONS; i = i + 1)
        $display("Neuron %0d spikes = %0d", i, spike_count[i]);

    #100;
    $finish;
end

endmodule
