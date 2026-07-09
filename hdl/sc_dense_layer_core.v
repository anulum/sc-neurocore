// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Core logic for an SC-based dense layer

// hdl/sc_dense_layer_core.v
//
// Core logic for an SC-based dense layer.
//
// Matches instantiation in sc_neurocore_top.v

`timescale 1ns / 1ps

module sc_dense_layer_core #(
    parameter integer N_INPUTS = 3,
    parameter integer N_NEURONS = 5,
    parameter integer DATA_WIDTH = 16 // fixed-point width for x_inputs/weights
)(
    input wire          clk,
    input wire          rst_n,

    // Control
    input wire          start_pulse,
    input wire [31:0]   stream_len, // Runtime configurable length

    // Scalar inputs (Packed fixed-point bus)
    // x_input_fp[i] is slice [i*DATA_WIDTH +: DATA_WIDTH]
    input wire [N_INPUTS*DATA_WIDTH-1:0] x_input_fp,

    // Scalar weights (Packed fixed-point bus)
    input wire [N_INPUTS*DATA_WIDTH-1:0] weight_fp,

    // Configuration for mapping probability -> current range
    input wire [DATA_WIDTH-1:0]    y_min_fp,
    input wire [DATA_WIDTH-1:0]    y_max_fp,

    // Neuron parameters
    input wire [DATA_WIDTH-1:0]    cfg_leak,
    input wire [DATA_WIDTH-1:0]    cfg_gain,

    // Debug output: Current I_t
    output wire [DATA_WIDTH-1:0]   I_t,

    // Spike outputs
    output wire [N_NEURONS-1:0]  spikes,
    output wire                  step_valid,
    output reg                   run_done,
    output reg                   running
);

// ----------------------------------------------------------------
// Internal parameters / signals
// ----------------------------------------------------------------
// Counter width must be sufficient for max stream_len (e.g. 2^32)
reg [31:0] t_counter;

// Unpacked scalar inputs and weights (for readability)
wire [DATA_WIDTH-1:0] x_inputs_arr [0:N_INPUTS-1];
wire [DATA_WIDTH-1:0] weight_arr [0:N_INPUTS-1];

genvar i;
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : UNPACK_INPUTS
        // Note: Assumes little-endian packing: [0] at LSB
        // Slice: [i*W +: W]
        assign x_inputs_arr[i] = x_input_fp[i*DATA_WIDTH +: DATA_WIDTH];
        assign weight_arr[i]   = weight_fp[i*DATA_WIDTH +: DATA_WIDTH];
    end
endgenerate


// ----------------------------------------------------------------
// Submodules: encoders, synapses, neurons
// ----------------------------------------------------------------

// Bitstream buses
wire [N_INPUTS-1:0] pre_bits_t;
wire [N_INPUTS-1:0] w_bits_t;
wire [N_INPUTS-1:0] post_bits_t;

// Spike outputs (from neuron cores)
wire [N_NEURONS-1:0] neuron_spikes_t;

assign spikes = neuron_spikes_t;
assign step_valid = running;


// ----------------------------------------------------------------
// Time-stepping control logic
// ----------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        t_counter <= 32'b0;
        running   <= 1'b0;
        run_done  <= 1'b0;
    end else begin
        if (start_pulse && !running) begin
            // Start new run
            running   <= 1'b1;
            run_done  <= 1'b0;
            t_counter <= 32'b0;
        end else if (running) begin
            if (t_counter == stream_len - 1) begin
                running <= 1'b0;
                run_done <= 1'b1;
            end else begin
                t_counter <= t_counter + 1'b1;
            end
        end else begin
            run_done <= 1'b0;
        end
    end
end


// ----------------------------------------------------------------
// Bitstream encoders for x_inputs
// ----------------------------------------------------------------
// Each encoder receives a unique SEED_INIT derived from a prime-stride
// sequence so that LFSRs start in distinct states and produce
// decorrelated bitstreams even when x_value inputs are identical.
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : ENC
        sc_bitstream_encoder #(
            .DATA_WIDTH(DATA_WIDTH),
            .SEED_INIT (16'hACE1 + i * 7)   // prime stride per input channel
        ) u_encoder (
            .clk        (clk),
            .rst_n      (rst_n),
            .x_value    (x_inputs_arr[i]),
            .t_index    (t_counter),
            .bit_out    (pre_bits_t[i])
        );
    end
endgenerate


// ----------------------------------------------------------------
// Bitstream encoders for weights
// ----------------------------------------------------------------
// Weight encoders use a different base seed (0xBEEF) with a second
// prime stride (13) to guarantee full decorrelation from input
// encoders AND from each other.
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : W_ENC
        sc_bitstream_encoder #(
            .DATA_WIDTH(DATA_WIDTH),
            .SEED_INIT (16'hBEEF + i * 13)  // different base + prime stride
        ) u_w_encoder (
            .clk        (clk),
            .rst_n      (rst_n),
            .x_value    (weight_arr[i]),
            .t_index    (t_counter),
            .bit_out    (w_bits_t[i])
        );
    end
endgenerate

// ----------------------------------------------------------------
// SC synapses (AND)
// ----------------------------------------------------------------
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : SYN
        sc_bitstream_synapse u_syn (
            .pre_bit    (pre_bits_t[i]),
            .w_bit      (w_bits_t[i]),
            .post_bit   (post_bits_t[i])
        );
    end
endgenerate

// ----------------------------------------------------------------
// Dot-product -> current I_t
// ----------------------------------------------------------------
sc_dotproduct_to_current #(
    .N_INPUTS(N_INPUTS),
    .DATA_WIDTH(DATA_WIDTH)
) u_dot (
    .post_bits (post_bits_t),
    .y_min     (y_min_fp),
    .y_max     (y_max_fp),
    .I_t       (I_t)
);


// ----------------------------------------------------------------
// Neuron cores (LIF)
// ----------------------------------------------------------------
// Noise input bus — one wire per neuron (tie to 0 for deterministic mode,
// or connect to an external noise generator for stochastic operation).
wire signed [DATA_WIDTH-1:0] noise_bus [0:N_NEURONS-1];
// Debug: per-neuron membrane potential
wire signed [DATA_WIDTH-1:0] v_out_bus [0:N_NEURONS-1];

generate
    for (i = 0; i < N_NEURONS; i = i + 1) begin : LIFs
        // Tie noise to zero — deterministic baseline.
        // Replace with an external LFSR/TRNG source for stochastic runs.
        assign noise_bus[i] = {DATA_WIDTH{1'b0}};

        sc_lif_neuron #(
            .DATA_WIDTH(DATA_WIDTH)
        ) u_neuron (
            .clk      (clk),
            .rst_n    (rst_n),
            .leak_k   (cfg_leak),
            .gain_k   (cfg_gain),
            .I_t      (I_t),
            .noise_in (noise_bus[i]),
            .spike_out(neuron_spikes_t[i]),
            .v_out    (v_out_bus[i])
        );
    end
endgenerate

endmodule
