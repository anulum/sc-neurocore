// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// SC-NeuroCore — AER spike router: distribute events to target neurons
//
// Takes AER events from source populations and routes them to target
// neurons based on a connectivity lookup table. Each event is weighted
// by the synaptic weight from the lookup table.
//
// For sparse connectivity, this is far more efficient than dense
// matrix-vector multiplication — only active connections are processed.
//
// Lookup table format: for each source neuron, a list of
// (target_id, weight) pairs stored in BRAM.

module sc_aer_router #(
    parameter N_SRC            = 128,
    parameter N_TGT            = 128,
    parameter MAX_FANOUT       = 32,     // max targets per source
    parameter NEURON_ID_WIDTH  = $clog2(N_TGT),
    parameter SRC_ID_WIDTH     = $clog2(N_SRC),
    parameter DATA_WIDTH       = 16,
    parameter TIMESTAMP_WIDTH  = 16
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // AER input (from sc_aer_encoder or upstream router)
    input  wire                         in_event_valid,
    input  wire [SRC_ID_WIDTH-1:0]      in_neuron_id,
    input  wire [TIMESTAMP_WIDTH-1:0]   in_timestamp,

    // Connectivity table (BRAM interface)
    // fanout_count[src] = number of targets
    // target_id[src][k] = k-th target neuron
    // target_weight[src][k] = k-th synapse weight
    input  wire [$clog2(MAX_FANOUT)-1:0] fanout_count [0:N_SRC-1],
    input  wire [NEURON_ID_WIDTH-1:0]    target_id    [0:N_SRC-1][0:MAX_FANOUT-1],
    input  wire signed [DATA_WIDTH-1:0]  target_weight[0:N_SRC-1][0:MAX_FANOUT-1],

    // AER output to target neurons (one per cycle)
    output reg                          out_event_valid,
    output reg  [NEURON_ID_WIDTH-1:0]   out_target_id,
    output reg  signed [DATA_WIDTH-1:0] out_weight,
    output reg  [TIMESTAMP_WIDTH-1:0]   out_timestamp,
    output wire                         busy
);

    reg processing;
    reg [SRC_ID_WIDTH-1:0]      current_src;
    reg [$clog2(MAX_FANOUT)-1:0] fan_idx;
    reg [$clog2(MAX_FANOUT)-1:0] fan_count;
    reg [TIMESTAMP_WIDTH-1:0]    stored_ts;

    assign busy = processing;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            processing      <= 0;
            out_event_valid <= 0;
            fan_idx         <= 0;
            fan_count       <= 0;
            current_src     <= 0;
            stored_ts       <= 0;
            out_target_id   <= 0;
            out_weight      <= 0;
            out_timestamp   <= 0;
        end else begin
            out_event_valid <= 0;

            if (processing) begin
                if (fan_idx < fan_count) begin
                    out_event_valid <= 1;
                    out_target_id   <= target_id[current_src][fan_idx];
                    out_weight      <= target_weight[current_src][fan_idx];
                    out_timestamp   <= stored_ts;
                    fan_idx         <= fan_idx + 1;
                end else begin
                    processing <= 0;
                end
            end else if (in_event_valid) begin
                current_src <= in_neuron_id;
                fan_count   <= fanout_count[in_neuron_id];
                fan_idx     <= 0;
                stored_ts   <= in_timestamp;
                processing  <= 1;
            end
        end
    end

endmodule
