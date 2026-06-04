`timescale 1ns/1ps
`default_nettype none
`include "timing_assertions.svh"

module example_dense_layer_core_latency (
    input wire clk,
    input wire rst_n
);
    localparam integer N_INPUTS = 1;
    localparam integer N_NEURONS = 1;
    localparam integer DATA_WIDTH = 16;

    reg start_pulse = 1'b0;
    reg [3:0] cycle = 4'd0;
    reg past_valid = 1'b0;

    wire signed [DATA_WIDTH-1:0] i_t;
    wire [N_NEURONS-1:0] spikes;
    wire step_valid;
    wire run_done;
    wire running;

    always @(posedge clk) begin
        past_valid <= 1'b1;
        if (!past_valid) begin
            assume (!rst_n);
        end else begin
            assume (rst_n);
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle <= 4'd0;
            start_pulse <= 1'b0;
        end else begin
            cycle <= cycle + 4'd1;
            start_pulse <= (cycle == 4'd1);
        end
    end

    sc_dense_layer_core #(
        .N_INPUTS(N_INPUTS),
        .N_NEURONS(N_NEURONS),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_pulse(start_pulse),
        .stream_len(32'd4),
        .x_input_fp(16'sd256),
        .weight_fp(16'sd128),
        .y_min_fp(-16'sd1024),
        .y_max_fp(16'sd1024),
        .cfg_leak(16'sd16),
        .cfg_gain(16'sd1),
        .I_t(i_t),
        .spikes(spikes),
        .step_valid(step_valid),
        .run_done(run_done),
        .running(running)
    );

    `SC_ASSERT_LATENCY_LE(dense_run_done, clk, rst_n, start_pulse, run_done, 6)
    `SC_ASSERT_BOUNDED_LIVENESS(dense_first_step, clk, rst_n, start_pulse, step_valid, 2)

    always @(posedge clk) begin
        if (rst_n) begin
            assert (!(run_done && running));
            assert (i_t <= 16'sd1024);
            assert (i_t >= -16'sd1024);
            cover (run_done);
            cover (|spikes || run_done);
        end
    end
endmodule

`default_nettype wire
