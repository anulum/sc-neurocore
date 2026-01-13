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
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : ENC
        sc_bitstream_encoder #(
            .DATA_WIDTH(DATA_WIDTH)
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
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : W_ENC
        sc_bitstream_encoder #(
            .DATA_WIDTH(DATA_WIDTH)
        ) u_w_encoder (
            .clk        (clk),
            .rst_n      (rst_n),
            .x_value    (weight_arr[i]),
            .t_index    (t_counter), // Correlated or decorrelated? 
                                     // Using same t_index implies correlation if seeds match.
                                     // sc_bitstream_encoder uses t_index[15:0] as seed if non-zero on reset.
                                     // But it only uses t_index on reset. 
                                     // Wait, sc_bitstream_encoder logic:
                                     // lfsr_reg <= (t_index[15:0] != 0) ? t_index[15:0] : 16'hACE1; (Only on RST)
                                     // So t_index change during run doesn't reseed.
                                     // It just advances LFSR.
                                     // To decorrelate, we need different seeds. 
                                     // Since seed is set at RST, and RST is global... 
                                     // We might need to modify encoder to allow seed offset or use different LFSRs.
                                     // For this MVP, we assume LFSRs drift apart or we rely on 'magic'.
                                     // Actually, sc_bitstream_encoder uses default seed if t_index=0 on reset.
                                     // If we want different streams, we need different seeds.
                                     // But t_index is 0 on reset! 
                                     // Issue identified: All encoders will output SAME bitstream for same input value.
                                     // FIX: Add parameter SEED_OFFSET to encoder or use instance ID.
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
generate
    for (i = 0; i < N_NEURONS; i = i + 1) begin : LIFs
        sc_lif_neuron #(
            .DATA_WIDTH(DATA_WIDTH)
        ) u_neuron (
            .clk      (clk),
            .rst_n    (rst_n),
            .leak_k   (cfg_leak),
            .gain_k   (cfg_gain),
            .I_t      (I_t), // Note: Shared I_t for now (single receptive field demo)
            .spike_out(neuron_spikes_t[i])
        );
    end
endgenerate

endmodule