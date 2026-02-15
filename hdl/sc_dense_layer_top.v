// hdl/sc_dense_layer_top.v
//
// Top-level skeleton for an SC-based dense layer,
// conceptually mirroring the Python SCDenseLayer interface.
//
// This is NOT a full implementation yet.
// It defines the ports and the internal structure to be wired up
// to actual SC submodules (encoders, synapses, neuron cores).

`timescale 1ns / 1ps

module sc_dense_layer_top #(
    parameter integer N_INPUTS = 3,
    parameter integer N_NEURONS = 5,
    parameter integer STREAM_LEN = 4096,
    parameter integer DATA_WIDTH = 16 // fixed-point width for x_inputs/weights
)(
    input wire          clk,
    input wire          rst_n,

    // Control
    input wire          start,
    output reg          done,

    // Scalar inputs (Python: x_inputs[] in [x_min, x_max])
    // Packed fixed-point bus: x_inputs[i] is slice [i*DATA_WIDTH +: DATA_WIDTH]
    input wire [N_INPUTS*DATA_WIDTH-1:0] x_inputs,

    // Scalar weights (Python: weight_values[])
    input wire [N_INPUTS*DATA_WIDTH-1:0] weight_values,

    // Configuration for mapping probability -> current range
    // Corresponds to y_min, y_max in Python; here as fixed-point.
    input wire [DATA_WIDTH-1:0]    y_min,
    input wire [DATA_WIDTH-1:0]    y_max,

    // Spike outputs: one bit per neuron, streamed over time.
    // In a PYNQ integration this would likely be connected to BRAM/FIFO.
    output wire [N_NEURONS-1:0]  spike_out,
    output wire                  spike_valid
);

// ----------------------------------------------------------------
// Internal parameters / signals
// ----------------------------------------------------------------
localparam integer CNT_WIDTH = $clog2(STREAM_LEN);

reg [CNT_WIDTH-1:0] t_counter;
reg                 running;

// Unpacked scalar inputs and weights (for readability)
wire [DATA_WIDTH-1:0] x_inputs_arr [0:N_INPUTS-1];
wire [DATA_WIDTH-1:0] weight_arr [0:N_INPUTS-1];

genvar i;
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : UNPACK_INPUTS
        assign x_inputs_arr[i] = x_inputs[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        assign weight_arr[i] = weight_values[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
    end
endgenerate


// ----------------------------------------------------------------
// Submodules: encoders, synapses, neurons (placeholders)
// ----------------------------------------------------------------

// The intended mapping from Python SCDenseLayer:
// - Bitstream encoders: BitstreamEncoder(x_inputs[i], ...)
// - Synapses:           BitstreamSynapse(weight_values[i], ...)
// - Dot-product:        BitstreamDotProduct(...)
// - Current source:     BitstreamCurrentSource(...)
// - Neurons:            StochasticLIFNeuron(...)
//
// Here, we sketch their interfaces so they can be implemented later.

// Example bitstream bus: bits for each input channel at current time t
wire [N_INPUTS-1:0] pre_bits_t;
wire [N_INPUTS-1:0] post_bits_t;

// Single current value for all neurons at time t
wire [DATA_WIDTH-1:0] I_t;

// Spike outputs (from neuron cores)
wire [N_NEURONS-1:0] neuron_spikes_t;

// Valid when a new time step is processed
assign spike_out = neuron_spikes_t;
assign spike_valid = running;


// ----------------------------------------------------------------
// Time-stepping control logic (STREAM_LEN steps per run)
// ----------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        t_counter <= {CNT_WIDTH{1'b0}};
        running   <= 1'b0;
        done      <= 1'b0;
    end else begin
        if (start && !running) begin
            // Start new run
            running   <= 1'b1;
            done      <= 1'b0;
            t_counter <= {CNT_WIDTH{1'b0}};
        end else if (running) begin
            if (t_counter == STREAM_LEN - 1) begin
                running <= 1'b0;
                done    <= 1'b1;
            end else begin
                t_counter <= t_counter + 1'b1;
            end
        end else begin
            done <= 1'b0;
        end
    end
end


// ----------------------------------------------------------------
// PLACEHOLDER: Bitstream encoders for x_inputs[i]
// ----------------------------------------------------------------
// Each encoder converts fixed-point x_inputs[i] into a time-varying
// Bernoulli bit pre_bits_t[i]. For now, we leave this as a black box.
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : ENC
        sc_bitstream_encoder #(
            .DATA_WIDTH(DATA_WIDTH)
        ) u_encoder (
            .clk        (clk),
            .rst_n      (rst_n),
            .x_value    (x_inputs_arr[i]),
            .t_index    ({20'b0, t_counter}),
            .bit_out    (pre_bits_t[i])
        );
    end
endgenerate


// ----------------------------------------------------------------
// PLACEHOLDER: SC synapses (AND with weight bitstreams)
// ----------------------------------------------------------------
wire [N_INPUTS-1:0] w_bits_t;

generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : W_ENC
        sc_bitstream_encoder #(
            .DATA_WIDTH(DATA_WIDTH)
        ) u_w_encoder (
            .clk        (clk),
            .rst_n      (rst_n),
            .x_value    (weight_arr[i]),
            .t_index    ({20'b0, t_counter}),
            .bit_out    (w_bits_t[i])
        );
    end
endgenerate

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
// PLACEHOLDER: Dot-product -> current I_t
// ----------------------------------------------------------------
// In hardware, this could be:
// - count ones in post_bits_t
// - normalize to [0,1]
// - map to current range [y_min, y_max]
sc_dotproduct_to_current #(
    .N_INPUTS(N_INPUTS),
    .DATA_WIDTH(DATA_WIDTH)
) u_dot (
    .post_bits (post_bits_t),
    .y_min     (y_min),
    .y_max     (y_max),
    .I_t       (I_t)
);


// ----------------------------------------------------------------
// PLACEHOLDER: Neuron cores (LIF-like), one per neuron
// ----------------------------------------------------------------
generate
    for (i = 0; i < N_NEURONS; i = i + 1) begin : LIFs
        sc_lif_neuron #(
            .DATA_WIDTH(DATA_WIDTH)
        ) u_neuron (
            .clk      (clk),
            .rst_n    (rst_n),
            .I_t      (I_t),
            .spike_out(neuron_spikes_t[i])
        );
    end
endgenerate

endmodule
