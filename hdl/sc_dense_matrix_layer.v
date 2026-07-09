// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense matrix layer: N_INPUTS × N_NEURONS with per-neuron

// hdl/sc_dense_matrix_layer.v
//
// Dense matrix layer: N_INPUTS × N_NEURONS with per-neuron weights.
// Each neuron j computes I_j = dot(x, W[j,:]) via stochastic bitstreams
// and integrates through an LIF neuron.
//
// Port interface uses packed buses (Verilog-2001 compatible).
// Internally unpacks via generate blocks.

`timescale 1ns / 1ps

module sc_dense_matrix_layer #(
    parameter integer N_INPUTS   = 16,
    parameter integer N_NEURONS  = 10,
    parameter integer DATA_WIDTH = 16,
    parameter integer FRACTION   = 8
)(
    input wire                                       clk,
    input wire                                       rst_n,

    // Control
    input wire                                       start_pulse,
    input wire [31:0]                                stream_len,

    // Pixel/input values — packed bus: pixel[i] at [i*DATA_WIDTH +: DATA_WIDTH]
    input wire [N_INPUTS*DATA_WIDTH-1:0]             x_input_fp,

    // Weight matrix — packed bus: W[j][i] at [(j*N_INPUTS+i)*DATA_WIDTH +: DATA_WIDTH]
    input wire [N_NEURONS*N_INPUTS*DATA_WIDTH-1:0]   weight_fp,

    // Current mapping range
    input wire [DATA_WIDTH-1:0]                      y_min_fp,
    input wire [DATA_WIDTH-1:0]                      y_max_fp,

    // Neuron parameters
    input wire [DATA_WIDTH-1:0]                      cfg_leak,
    input wire [DATA_WIDTH-1:0]                      cfg_gain,

    // Outputs
    output wire [N_NEURONS-1:0]                      spikes,
    output wire                                      step_valid,
    output reg                                       run_done,
    output reg                                       running
);

// ----------------------------------------------------------------
// Time-stepping control
// ----------------------------------------------------------------
reg [31:0] t_counter;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        t_counter <= 32'b0;
        running   <= 1'b0;
        run_done  <= 1'b0;
    end else begin
        if (start_pulse && !running) begin
            running   <= 1'b1;
            run_done  <= 1'b0;
            t_counter <= 32'b0;
        end else if (running) begin
            if (t_counter == stream_len - 1) begin
                running  <= 1'b0;
                run_done <= 1'b1;
            end else begin
                t_counter <= t_counter + 1'b1;
            end
        end else begin
            run_done <= 1'b0;
        end
    end
end

assign step_valid = running;

// ----------------------------------------------------------------
// Shared input encoders (one per pixel)
// ----------------------------------------------------------------
wire [N_INPUTS-1:0] pre_bits;

genvar i, j;
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : IN_ENC
        sc_bitstream_encoder #(
            .DATA_WIDTH (DATA_WIDTH),
            .SEED_INIT  (16'hACE1 + i * 7)
        ) u_enc (
            .clk     (clk),
            .rst_n   (rst_n),
            .x_value (x_input_fp[i*DATA_WIDTH +: DATA_WIDTH]),
            .t_index (t_counter),
            .bit_out (pre_bits[i])
        );
    end
endgenerate

// ----------------------------------------------------------------
// Per-neuron: weight encoders + synapses + dot-product + LIF
// ----------------------------------------------------------------
generate
    for (j = 0; j < N_NEURONS; j = j + 1) begin : NEURON
        // Weight encoders for neuron j
        wire [N_INPUTS-1:0] w_bits;
        for (i = 0; i < N_INPUTS; i = i + 1) begin : W_ENC
            sc_bitstream_encoder #(
                .DATA_WIDTH (DATA_WIDTH),
                .SEED_INIT  (16'hBEEF + j * 13 * N_INPUTS + i * 13)
            ) u_w_enc (
                .clk     (clk),
                .rst_n   (rst_n),
                .x_value (weight_fp[(j*N_INPUTS + i)*DATA_WIDTH +: DATA_WIDTH]),
                .t_index (t_counter),
                .bit_out (w_bits[i])
            );
        end

        // AND synapses
        wire [N_INPUTS-1:0] post_bits;
        for (i = 0; i < N_INPUTS; i = i + 1) begin : SYN
            sc_bitstream_synapse u_syn (
                .pre_bit  (pre_bits[i]),
                .w_bit    (w_bits[i]),
                .post_bit (post_bits[i])
            );
        end

        // Dot-product → current
        wire [DATA_WIDTH-1:0] I_t_j;
        sc_dotproduct_to_current #(
            .N_INPUTS  (N_INPUTS),
            .DATA_WIDTH(DATA_WIDTH)
        ) u_dot (
            .post_bits (post_bits),
            .y_min     (y_min_fp),
            .y_max     (y_max_fp),
            .I_t       (I_t_j)
        );

        // LIF neuron
        wire signed [DATA_WIDTH-1:0] v_out_j;
        sc_lif_neuron #(
            .DATA_WIDTH(DATA_WIDTH),
            .FRACTION  (FRACTION)
        ) u_lif (
            .clk       (clk),
            .rst_n     (rst_n),
            .leak_k    (cfg_leak),
            .gain_k    (cfg_gain),
            .I_t       (I_t_j),
            .noise_in  ({DATA_WIDTH{1'b0}}),
            .spike_out (spikes[j]),
            .v_out     (v_out_j)
        );
    end
endgenerate

endmodule
