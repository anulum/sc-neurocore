// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - DCLS axonal delay line

`timescale 1ns / 1ps

module sc_dcls_axonal_delay #(
    parameter integer DEPTH = 31,
    parameter integer PTR_WIDTH = 5
)(
    input wire clk,
    input wire rst_n,
    input wire spike_in,
    input wire [PTR_WIDTH-1:0] read_offset,
    output wire spike_out
);
    reg [DEPTH-1:0] buffer_reg;
    reg [PTR_WIDTH-1:0] head;
    wire [PTR_WIDTH-1:0] read_idx;
    // DEPTH reduced into the pointer width on purpose: the circular buffer wraps modulo
    // 2**PTR_WIDTH, so DEPTH_VALUE carries DEPTH mod 2**PTR_WIDTH for the wrap-around add below.
    /* verilator lint_off WIDTHTRUNC */
    localparam [PTR_WIDTH-1:0] DEPTH_VALUE = DEPTH;
    /* verilator lint_on WIDTHTRUNC */

    assign read_idx = (head >= read_offset)
        ? (head - read_offset)
        : (head + DEPTH_VALUE - read_offset);
    assign spike_out = (read_idx == head) ? spike_in : buffer_reg[read_idx];

    integer init_idx;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head <= {PTR_WIDTH{1'b0}};
            for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                buffer_reg[init_idx] <= 1'b0;
            end
        end else begin
            buffer_reg[head] <= spike_in;
            if (head == DEPTH_VALUE - 1'b1) begin
                head <= {PTR_WIDTH{1'b0}};
            end else begin
                head <= head + 1'b1;
            end
        end
    end
endmodule
