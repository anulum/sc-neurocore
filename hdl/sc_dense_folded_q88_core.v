// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - Folded Q8.8/Q16.16 dense layer core

`timescale 1ns / 1ps

module sc_dense_folded_q88_core #(
    parameter integer N_INPUTS = 64,
    parameter integer N_NEURONS = 32,
    parameter integer DATA_WIDTH = 16,
    parameter integer FRAC_BITS = 8,
    parameter integer ACC_WIDTH = 48,
    parameter integer PARALLEL_NEURONS = 5,
    parameter signed [ACC_WIDTH-1:0] THRESHOLD_Q = 48'sd256
)(
    input wire clk,
    input wire rst_n,
    input wire start_pulse,
    input wire [N_INPUTS*DATA_WIDTH-1:0] x_input_fp,
    input wire [N_INPUTS*N_NEURONS*DATA_WIDTH-1:0] weight_fp,
    input wire [DATA_WIDTH-1:0] cfg_leak,
    input wire [DATA_WIDTH-1:0] cfg_gain,
    output reg [N_NEURONS-1:0] spikes,
    output wire step_valid,
    output reg run_done,
    output reg running,
    output reg overflow,
    output reg [31:0] compute_cycle_count
);

localparam integer GROUP_COUNT = (N_NEURONS + PARALLEL_NEURONS - 1) / PARALLEL_NEURONS;
localparam signed [ACC_WIDTH-1:0] ACC_MAX = (48'sd1 <<< (ACC_WIDTH - 1)) - 48'sd1;
localparam signed [ACC_WIDTH-1:0] ACC_MIN = -(48'sd1 <<< (ACC_WIDTH - 1));

reg [31:0] group_index;
integer lane;
integer input_index;
integer neuron_index;
reg signed [DATA_WIDTH-1:0] x_value;
reg signed [DATA_WIDTH-1:0] w_value;
reg signed [DATA_WIDTH-1:0] gain_value;
reg signed [DATA_WIDTH-1:0] leak_value;
reg signed [ACC_WIDTH-1:0] accumulator;
reg signed [ACC_WIDTH-1:0] scaled;

assign step_valid = running;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        spikes <= {N_NEURONS{1'b0}};
        running <= 1'b0;
        run_done <= 1'b0;
        overflow <= 1'b0;
        compute_cycle_count <= 32'd0;
        group_index <= 32'd0;
    end else begin
        run_done <= 1'b0;
        if (start_pulse && !running) begin
            spikes <= {N_NEURONS{1'b0}};
            running <= 1'b1;
            overflow <= 1'b0;
            compute_cycle_count <= 32'd0;
            group_index <= 32'd0;
        end else if (running) begin
            gain_value = cfg_gain;
            leak_value = cfg_leak;
            for (lane = 0; lane < PARALLEL_NEURONS; lane = lane + 1) begin
                neuron_index = group_index * PARALLEL_NEURONS + lane;
                if (neuron_index < N_NEURONS) begin
                    accumulator = {ACC_WIDTH{1'b0}};
                    for (input_index = 0; input_index < N_INPUTS; input_index = input_index + 1) begin
                        x_value = x_input_fp[input_index*DATA_WIDTH +: DATA_WIDTH];
                        w_value = weight_fp[(neuron_index*N_INPUTS + input_index)*DATA_WIDTH +: DATA_WIDTH];
                        accumulator = accumulator + (($signed(x_value) * $signed(w_value)) >>> FRAC_BITS);
                    end
                    scaled = ((accumulator * $signed(gain_value)) >>> FRAC_BITS) - $signed(leak_value);
                    if (scaled > ACC_MAX) begin
                        scaled = ACC_MAX;
                        overflow <= 1'b1;
                    end else if (scaled < ACC_MIN) begin
                        scaled = ACC_MIN;
                        overflow <= 1'b1;
                    end
                    spikes[neuron_index] <= scaled >= THRESHOLD_Q;
                end
            end
            compute_cycle_count <= compute_cycle_count + 32'd1;
            if (group_index == GROUP_COUNT - 1) begin
                running <= 1'b0;
                run_done <= 1'b1;
                group_index <= 32'd0;
            end else begin
                group_index <= group_index + 32'd1;
            end
        end
    end
end

endmodule
