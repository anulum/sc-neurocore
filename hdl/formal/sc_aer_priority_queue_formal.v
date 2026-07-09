// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - bounded formal harness for the AER priority queue.

`default_nettype none

module sc_aer_priority_queue_formal (
    input wire clk
);
    reg rst_n = 1'b0;
    reg past_valid = 1'b0;
    reg [3:0] cycle = 4'd0;

    reg in_valid;
    wire in_ready;
    reg [3:0] in_target_id;
    reg signed [7:0] in_weight;
    reg [7:0] in_timestamp;
    reg [1:0] in_priority;
    wire out_valid;
    reg out_ready;
    wire [3:0] out_target_id;
    wire signed [7:0] out_weight;
    wire [7:0] out_timestamp;
    wire [1:0] out_priority;
    wire full;
    wire empty;
    wire busy;
    wire dropped_event;
    wire critical_deadline_violation;
    wire [1:0] occupancy;

    sc_aer_priority_queue #(
        .NEURON_ID_WIDTH(4),
        .DATA_WIDTH(8),
        .TIMESTAMP_WIDTH(8),
        .PRIO_WIDTH(2),
        .QUEUE_DEPTH(2),
        .MAX_LATENCY_CYCLES(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_target_id(in_target_id),
        .in_weight(in_weight),
        .in_timestamp(in_timestamp),
        .in_priority(in_priority),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_target_id(out_target_id),
        .out_weight(out_weight),
        .out_timestamp(out_timestamp),
        .out_priority(out_priority),
        .full(full),
        .empty(empty),
        .busy(busy),
        .dropped_event(dropped_event),
        .critical_deadline_violation(critical_deadline_violation),
        .occupancy(occupancy)
    );

    always @* begin
        in_valid = 1'b0;
        in_target_id = 4'd0;
        in_weight = 8'sd0;
        in_timestamp = {4'd0, cycle};
        in_priority = 2'd0;
        out_ready = 1'b0;

        if (rst_n) begin
            case (cycle)
                4'd1: begin
                    in_valid = 1'b1;
                    in_target_id = 4'd1;
                    in_weight = 8'sd1;
                    in_priority = 2'd3;
                end
                4'd2: begin
                    in_valid = 1'b1;
                    in_target_id = 4'd2;
                    in_weight = 8'sd2;
                    in_priority = 2'd0;
                end
                4'd5,
                4'd6: begin
                    out_ready = 1'b1;
                end
                4'd7: begin
                    in_valid = 1'b1;
                    in_target_id = 4'd3;
                    in_weight = 8'sd3;
                    in_priority = 2'd2;
                end
                4'd8: begin
                    in_valid = 1'b1;
                    in_target_id = 4'd4;
                    in_weight = 8'sd4;
                    in_priority = 2'd2;
                end
                4'd10: begin
                    in_valid = 1'b1;
                    in_target_id = 4'd5;
                    in_weight = 8'sd5;
                    in_priority = 2'd2;
                end
                default: begin
                    out_ready = 1'b0;
                end
            endcase
        end
    end

    always @(posedge clk) begin
        past_valid <= 1'b1;
        rst_n <= past_valid;

        if (!rst_n) begin
            cycle <= 4'd0;
        end else if (cycle != 4'd15) begin
            cycle <= cycle + 4'd1;
        end

        if (rst_n && past_valid) begin
            if ($past(dropped_event)) begin
                assert(dropped_event);
            end

            if ($past(critical_deadline_violation)) begin
                assert(critical_deadline_violation);
            end
        end
    end

    always @* begin
        if (rst_n) begin
            assert(occupancy <= 2'd2);
            assert(full == (occupancy == 2'd2));
            assert(empty == (occupancy == 2'd0));
            assert(busy == !empty);
            assert(out_valid == !empty);

            if (cycle == 4'd3) begin
                assert(out_valid);
                assert(out_priority == 2'd0);
                assert(out_target_id == 4'd2);
            end

            if (cycle == 4'd5) begin
                assert(critical_deadline_violation);
            end

            if (cycle == 4'd9) begin
                assert(out_valid);
                assert(out_priority == 2'd2);
                assert(out_target_id == 4'd3);
            end

            if (cycle == 4'd10) begin
                assert(full);
                assert(!in_ready);
            end

            if (cycle == 4'd11) begin
                assert(dropped_event);
            end

            cover(full && in_valid && !in_ready);
        end
    end
endmodule

`default_nettype wire
