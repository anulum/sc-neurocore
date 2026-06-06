`timescale 1ns/1ps
`default_nettype none

module sc_latency_monitor #(
    parameter integer MAX_CYCLES = 1,
    parameter integer COUNTER_WIDTH = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start_event,
    input  wire                         end_event,
    output reg                          violation,
    output reg                          active,
    output reg  [COUNTER_WIDTH-1:0]     age
);
    localparam [COUNTER_WIDTH-1:0] MAX_COUNT = MAX_CYCLES;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            violation <= 1'b0;
            active <= 1'b0;
            age <= {COUNTER_WIDTH{1'b0}};
        end else if (violation) begin
            violation <= 1'b1;
            active <= active;
            age <= age;
        end else if (active && end_event) begin
            active <= 1'b0;
            age <= {COUNTER_WIDTH{1'b0}};
        end else if (!active && start_event) begin
            if (end_event) begin
                active <= 1'b0;
                age <= {COUNTER_WIDTH{1'b0}};
            end else begin
                active <= 1'b1;
                age <= {COUNTER_WIDTH{1'b0}};
            end
        end else if (active) begin
            if (age >= MAX_COUNT) begin
                violation <= 1'b1;
                active <= active;
                age <= age;
            end else begin
                age <= age + {{(COUNTER_WIDTH-1){1'b0}}, 1'b1};
            end
        end else begin
            active <= 1'b0;
            age <= {COUNTER_WIDTH{1'b0}};
        end
    end
endmodule

module sc_deadline_monitor #(
    parameter integer DEADLINE_CYCLES = 1,
    parameter integer COUNTER_WIDTH = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         deadline_start,
    input  wire                         completion_event,
    output wire                         violation,
    output wire                         active,
    output wire [COUNTER_WIDTH-1:0]     age
);
    sc_latency_monitor #(
        .MAX_CYCLES(DEADLINE_CYCLES),
        .COUNTER_WIDTH(COUNTER_WIDTH)
    ) deadline_monitor (
        .clk(clk),
        .rst_n(rst_n),
        .start_event(deadline_start),
        .end_event(completion_event),
        .violation(violation),
        .active(active),
        .age(age)
    );
endmodule

module sc_bounded_liveness_monitor #(
    parameter integer WINDOW_CYCLES = 1,
    parameter integer COUNTER_WIDTH = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         request_event,
    input  wire                         witness_event,
    output wire                         violation,
    output wire                         active,
    output wire [COUNTER_WIDTH-1:0]     age
);
    sc_latency_monitor #(
        .MAX_CYCLES(WINDOW_CYCLES),
        .COUNTER_WIDTH(COUNTER_WIDTH)
    ) liveness_monitor (
        .clk(clk),
        .rst_n(rst_n),
        .start_event(request_event),
        .end_event(witness_event),
        .violation(violation),
        .active(active),
        .age(age)
    );
endmodule

`default_nettype wire
