// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - strict-priority AER queue with backpressure.

`default_nettype none

module sc_aer_priority_queue #(
    parameter NEURON_ID_WIDTH      = 8,
    parameter DATA_WIDTH           = 16,
    parameter TIMESTAMP_WIDTH      = 16,
    parameter PRIO_WIDTH           = 2,
    parameter QUEUE_DEPTH          = 16,
    parameter MAX_LATENCY_CYCLES   = 16,
    parameter PTR_WIDTH            = (QUEUE_DEPTH <= 1) ? 1 : $clog2(QUEUE_DEPTH),
    parameter COUNT_WIDTH          = $clog2(QUEUE_DEPTH + 1),
    parameter LATENCY_COUNT_WIDTH  = $clog2(MAX_LATENCY_CYCLES + 2)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire [NEURON_ID_WIDTH-1:0]   in_target_id,
    input  wire signed [DATA_WIDTH-1:0] in_weight,
    input  wire [TIMESTAMP_WIDTH-1:0]   in_timestamp,
    input  wire [PRIO_WIDTH-1:0]        in_priority,

    output wire                         out_valid,
    input  wire                         out_ready,
    output wire [NEURON_ID_WIDTH-1:0]   out_target_id,
    output wire signed [DATA_WIDTH-1:0] out_weight,
    output wire [TIMESTAMP_WIDTH-1:0]   out_timestamp,
    output wire [PRIO_WIDTH-1:0]        out_priority,

    output wire                         full,
    output wire                         empty,
    output wire                         busy,
    output reg                          dropped_event,
    output reg                          critical_deadline_violation,
    output wire [COUNT_WIDTH-1:0]       occupancy
);

    reg [NEURON_ID_WIDTH-1:0]   queue_target_id [0:QUEUE_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] queue_weight    [0:QUEUE_DEPTH-1];
    reg [TIMESTAMP_WIDTH-1:0]   queue_timestamp [0:QUEUE_DEPTH-1];
    reg [PRIO_WIDTH-1:0]        queue_priority  [0:QUEUE_DEPTH-1];

    reg [COUNT_WIDTH-1:0] count;
    reg [PTR_WIDTH-1:0] best_idx;
    reg [PRIO_WIDTH-1:0]  best_priority;
    reg [LATENCY_COUNT_WIDTH-1:0] critical_wait_cycles;
    localparam [COUNT_WIDTH-1:0] QUEUE_DEPTH_LIMIT = QUEUE_DEPTH;
    localparam [COUNT_WIDTH-1:0] COUNT_ONE = 1;
    localparam [LATENCY_COUNT_WIDTH-1:0] MAX_LATENCY_LIMIT = MAX_LATENCY_CYCLES;
    wire [COUNT_WIDTH-1:0] count_minus_one;
    wire [PTR_WIDTH-1:0] append_index;
    wire [PTR_WIDTH-1:0] replace_index;

    integer scan_idx;
    integer shift_idx;

    assign full = (count == QUEUE_DEPTH_LIMIT);
    assign empty = (count == {COUNT_WIDTH{1'b0}});
    assign busy = !empty;
    assign out_valid = !empty;
    assign occupancy = count;
    assign in_ready = !full || (out_valid && out_ready);

    assign out_target_id = out_valid ? queue_target_id[best_idx] : {NEURON_ID_WIDTH{1'b0}};
    assign out_weight = out_valid ? queue_weight[best_idx] : {DATA_WIDTH{1'b0}};
    assign out_timestamp = out_valid ? queue_timestamp[best_idx] : {TIMESTAMP_WIDTH{1'b0}};
    assign out_priority = out_valid ? queue_priority[best_idx] : {PRIO_WIDTH{1'b0}};
    assign count_minus_one = count - COUNT_ONE;
    assign append_index = count[PTR_WIDTH-1:0];
    assign replace_index = count_minus_one[PTR_WIDTH-1:0];

    always @* begin
        best_idx = {PTR_WIDTH{1'b0}};
        best_priority = {PRIO_WIDTH{1'b1}};
        for (scan_idx = 0; scan_idx < QUEUE_DEPTH; scan_idx = scan_idx + 1) begin
            if (scan_idx[COUNT_WIDTH-1:0] < count) begin
                if (queue_priority[scan_idx] < best_priority) begin
                    best_idx = scan_idx[PTR_WIDTH-1:0];
                    best_priority = queue_priority[scan_idx];
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= {COUNT_WIDTH{1'b0}};
            dropped_event <= 1'b0;
            critical_deadline_violation <= 1'b0;
            critical_wait_cycles <= {LATENCY_COUNT_WIDTH{1'b0}};
            for (shift_idx = 0; shift_idx < QUEUE_DEPTH; shift_idx = shift_idx + 1) begin
                queue_target_id[shift_idx] <= {NEURON_ID_WIDTH{1'b0}};
                queue_weight[shift_idx] <= {DATA_WIDTH{1'b0}};
                queue_timestamp[shift_idx] <= {TIMESTAMP_WIDTH{1'b0}};
                queue_priority[shift_idx] <= {PRIO_WIDTH{1'b0}};
            end
        end else begin
            if (in_valid && !in_ready) begin
                dropped_event <= 1'b1;
            end

            if (out_valid && (out_priority == {PRIO_WIDTH{1'b0}}) && !out_ready) begin
                if (critical_wait_cycles >= MAX_LATENCY_LIMIT) begin
                    critical_deadline_violation <= 1'b1;
                end else begin
                    critical_wait_cycles <= critical_wait_cycles + {{(LATENCY_COUNT_WIDTH-1){1'b0}}, 1'b1};
                end
            end else if (!out_valid || out_ready || (out_priority != {PRIO_WIDTH{1'b0}})) begin
                critical_wait_cycles <= {LATENCY_COUNT_WIDTH{1'b0}};
            end

            if (out_valid && out_ready) begin
                for (shift_idx = 0; shift_idx < QUEUE_DEPTH - 1; shift_idx = shift_idx + 1) begin
                    if (
                        (shift_idx[PTR_WIDTH-1:0] >= best_idx)
                        && (shift_idx[COUNT_WIDTH-1:0] < count_minus_one)
                    ) begin
                        queue_target_id[shift_idx] <= queue_target_id[shift_idx + 1];
                        queue_weight[shift_idx] <= queue_weight[shift_idx + 1];
                        queue_timestamp[shift_idx] <= queue_timestamp[shift_idx + 1];
                        queue_priority[shift_idx] <= queue_priority[shift_idx + 1];
                    end
                end
            end

            if (in_valid && in_ready) begin
                if (out_valid && out_ready) begin
                    queue_target_id[replace_index] <= in_target_id;
                    queue_weight[replace_index] <= in_weight;
                    queue_timestamp[replace_index] <= in_timestamp;
                    queue_priority[replace_index] <= in_priority;
                end else begin
                    queue_target_id[append_index] <= in_target_id;
                    queue_weight[append_index] <= in_weight;
                    queue_timestamp[append_index] <= in_timestamp;
                    queue_priority[append_index] <= in_priority;
                end
            end

            case ({in_valid && in_ready, out_valid && out_ready})
                2'b10: count <= count + COUNT_ONE;
                2'b01: count <= count - COUNT_ONE;
                default: count <= count;
            endcase
        end
    end

endmodule

`default_nettype wire
