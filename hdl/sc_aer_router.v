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
    parameter TIMESTAMP_WIDTH  = 16,
    parameter PRIO_WIDTH       = 1,
    parameter PRIORITY_ENABLED = 0,
    parameter QUEUE_DEPTH      = 16,
    parameter MAX_LATENCY_CYCLES = 16
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // AER input (from sc_aer_encoder or upstream router)
    input  wire                         in_event_valid,
    output wire                         in_event_ready,
    input  wire [SRC_ID_WIDTH-1:0]      in_neuron_id,
    input  wire [TIMESTAMP_WIDTH-1:0]   in_timestamp,
    input  wire [PRIO_WIDTH-1:0]        in_priority,

    // Connectivity table (BRAM interface)
    // fanout_count[src] = number of targets
    // target_id[src][k] = k-th target neuron
    // target_weight[src][k] = k-th synapse weight
    input  wire [$clog2(MAX_FANOUT)-1:0] fanout_count [0:N_SRC-1],
    input  wire [NEURON_ID_WIDTH-1:0]    target_id    [0:N_SRC-1][0:MAX_FANOUT-1],
    input  wire signed [DATA_WIDTH-1:0]  target_weight[0:N_SRC-1][0:MAX_FANOUT-1],

    // AER output to target neurons (one per cycle)
    output reg                          out_event_valid,
    input  wire                         out_event_ready,
    output reg  [NEURON_ID_WIDTH-1:0]   out_target_id,
    output reg  signed [DATA_WIDTH-1:0] out_weight,
    output reg  [TIMESTAMP_WIDTH-1:0]   out_timestamp,
    output reg  [PRIO_WIDTH-1:0]        out_priority,
    output wire                         busy,
    output wire                         queue_full,
    output wire                         dropped_event,
    output wire                         critical_deadline_violation
);

    reg processing;
    reg [SRC_ID_WIDTH-1:0]      current_src;
    reg [$clog2(MAX_FANOUT)-1:0] fan_idx;
    reg [$clog2(MAX_FANOUT)-1:0] fan_count;
    reg [TIMESTAMP_WIDTH-1:0]    stored_ts;
    reg [PRIO_WIDTH-1:0]         stored_priority;

    wire output_ready;
    assign output_ready = (out_event_ready !== 1'b0);

    wire queue_in_valid;
    wire queue_in_ready;
    wire queue_out_valid;
    wire [NEURON_ID_WIDTH-1:0] queue_out_target_id;
    wire signed [DATA_WIDTH-1:0] queue_out_weight;
    wire [TIMESTAMP_WIDTH-1:0] queue_out_timestamp;
    wire [PRIO_WIDTH-1:0] queue_out_priority;
    wire queue_busy;
    wire queue_internal_full;
    wire queue_internal_empty;
    wire queue_dropped_event;
    wire queue_deadline_violation;
    wire [$clog2(QUEUE_DEPTH + 1)-1:0] queue_occupancy;

    assign busy = processing
        || (PRIORITY_ENABLED != 0 && (queue_busy || !queue_internal_empty || (|queue_occupancy)));
    assign in_event_ready = !processing;
    assign queue_full = (PRIORITY_ENABLED != 0) ? queue_internal_full : 1'b0;
    assign dropped_event = (PRIORITY_ENABLED != 0) ? queue_dropped_event : 1'b0;
    assign critical_deadline_violation =
        (PRIORITY_ENABLED != 0) ? queue_deadline_violation : 1'b0;
    assign queue_in_valid = (PRIORITY_ENABLED != 0) && processing && (fan_idx < fan_count);

    sc_aer_priority_queue #(
        .NEURON_ID_WIDTH(NEURON_ID_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TIMESTAMP_WIDTH(TIMESTAMP_WIDTH),
        .PRIO_WIDTH(PRIO_WIDTH),
        .QUEUE_DEPTH(QUEUE_DEPTH),
        .MAX_LATENCY_CYCLES(MAX_LATENCY_CYCLES)
    ) priority_queue (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(queue_in_valid),
        .in_ready(queue_in_ready),
        .in_target_id(target_id[current_src][fan_idx]),
        .in_weight(target_weight[current_src][fan_idx]),
        .in_timestamp(stored_ts),
        .in_priority(stored_priority),
        .out_valid(queue_out_valid),
        .out_ready(output_ready),
        .out_target_id(queue_out_target_id),
        .out_weight(queue_out_weight),
        .out_timestamp(queue_out_timestamp),
        .out_priority(queue_out_priority),
        .full(queue_internal_full),
        .empty(queue_internal_empty),
        .busy(queue_busy),
        .dropped_event(queue_dropped_event),
        .critical_deadline_violation(queue_deadline_violation),
        .occupancy(queue_occupancy)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            processing      <= 0;
            out_event_valid <= 0;
            fan_idx         <= 0;
            fan_count       <= 0;
            current_src     <= 0;
            stored_ts       <= 0;
            stored_priority <= 0;
            out_target_id   <= 0;
            out_weight      <= 0;
            out_timestamp   <= 0;
            out_priority    <= 0;
        end else begin
            if (PRIORITY_ENABLED != 0) begin
                out_event_valid <= queue_out_valid;
                out_target_id   <= queue_out_target_id;
                out_weight      <= queue_out_weight;
                out_timestamp   <= queue_out_timestamp;
                out_priority    <= queue_out_priority;

                if (processing) begin
                    if (fan_idx < fan_count) begin
                        if (queue_in_ready) begin
                            fan_idx <= fan_idx + 1;
                        end
                    end else begin
                        processing <= 0;
                    end
                end else if (in_event_valid) begin
                    current_src     <= in_neuron_id;
                    fan_count       <= fanout_count[in_neuron_id];
                    fan_idx         <= 0;
                    stored_ts       <= in_timestamp;
                    stored_priority <= in_priority;
                    processing      <= 1;
                end
            end else begin
                if (out_event_valid && !output_ready) begin
                    out_event_valid <= out_event_valid;
                end else begin
                    out_event_valid <= 0;
                end

                if (processing && !(out_event_valid && !output_ready)) begin
                if (fan_idx < fan_count) begin
                    out_event_valid <= 1;
                    out_target_id   <= target_id[current_src][fan_idx];
                    out_weight      <= target_weight[current_src][fan_idx];
                    out_timestamp   <= stored_ts;
                        out_priority    <= stored_priority;
                    fan_idx         <= fan_idx + 1;
                end else begin
                    processing <= 0;
                end
                end else if (in_event_valid && in_event_ready) begin
                    current_src     <= in_neuron_id;
                    fan_count       <= fanout_count[in_neuron_id];
                    fan_idx         <= 0;
                    stored_ts       <= in_timestamp;
                    stored_priority <= in_priority;
                    processing      <= 1;
                end
            end
        end
    end

endmodule
