// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Top-level wrapper:

// hdl/sc_neurocore_top.v
//
// Top-level wrapper:
//  - AXI-Lite for config & status
//  - SC dense layer core (7 neurons, 3 inputs)
//  - Firing-rate estimator bank
//
// You'll usually wrap this with Vivado's IP packaging.

`timescale 1ns / 1ps

module sc_neurocore_top #(
    parameter integer C_S_AXI_ADDR_WIDTH = 8,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer N_INPUTS             = 3,
    parameter integer N_NEURONS            = 7,
    parameter integer DATA_WIDTH           = 16
)(
    // AXI-Lite interface
    input wire                                     S_AXI_ACLK,
    input wire                                     S_AXI_ARESETN,
    input wire [C_S_AXI_ADDR_WIDTH-1:0]   S_AXI_AWADDR,
    input wire                                     S_AXI_AWVALID,
    output wire                                    S_AXI_AWREADY,
    input wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input wire [3:0]                               S_AXI_WSTRB,
    input wire                                     S_AXI_WVALID,
    output wire                                    S_AXI_WREADY,
    output wire [1:0]                              S_AXI_BRESP,
    output wire                                    S_AXI_BVALID,
    input wire                                     S_AXI_BREADY,
    input wire [C_S_AXI_ADDR_WIDTH-1:0]   S_AXI_ARADDR,
    input wire                                     S_AXI_ARVALID,
    output wire                                    S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0]  S_AXI_RDATA,
    output wire [1:0]                              S_AXI_RRESP,
    output wire                                    S_AXI_RVALID,
    input wire                                     S_AXI_RREADY
);

// ----------------------------------------------------------------
// Config wires from AXI block
// ----------------------------------------------------------------
wire          cfg_start_pulse;
wire [15:0]   cfg_x_input [0:N_INPUTS-1];
wire [15:0]   cfg_weight [0:N_INPUTS-1];
wire [15:0]   cfg_y_min;
wire [15:0]   cfg_y_max;
wire [15:0]   cfg_leak;
wire [15:0]   cfg_gain;
wire [31:0]   cfg_stream_len;
wire [31:0]   cfg_dt_ms;
wire [31:0]   cfg_scale_q16;

// ----------------------------------------------------------------
// Status / firing rates to AXI
// ----------------------------------------------------------------
wire          stat_busy;
wire          stat_done;
wire [31:0]   stat_rate_q16 [0:N_NEURONS-1];

// ----------------------------------------------------------------
// AXI-Lite config/register block
// ----------------------------------------------------------------
sc_axil_cfg #(
    .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
    .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH)
) u_axil (
    .S_AXI_ACLK     (S_AXI_ACLK),
    .S_AXI_ARESETN  (S_AXI_ARESETN),
    .S_AXI_AWADDR   (S_AXI_AWADDR),
    .S_AXI_AWVALID  (S_AXI_AWVALID),
    .S_AXI_AWREADY  (S_AXI_AWREADY),
    .S_AXI_WDATA    (S_AXI_WDATA),
    .S_AXI_WSTRB    (S_AXI_WSTRB),
    .S_AXI_WVALID   (S_AXI_WVALID),
    .S_AXI_WREADY   (S_AXI_WREADY),
    .S_AXI_BRESP    (S_AXI_BRESP),
    .S_AXI_BVALID   (S_AXI_BVALID),
    .S_AXI_BREADY   (S_AXI_BREADY),
    .S_AXI_ARADDR   (S_AXI_ARADDR),
    .S_AXI_ARVALID  (S_AXI_ARVALID),
    .S_AXI_ARREADY  (S_AXI_ARREADY),
    .S_AXI_RDATA    (S_AXI_RDATA),
    .S_AXI_RRESP    (S_AXI_RRESP),
    .S_AXI_RVALID   (S_AXI_RVALID),
    .S_AXI_RREADY   (S_AXI_RREADY),

    .cfg_start_pulse(cfg_start_pulse),
    .cfg_x_input    (cfg_x_input),
    .cfg_weight     (cfg_weight),
    .cfg_y_min      (cfg_y_min),
    .cfg_y_max      (cfg_y_max),
    .cfg_leak       (cfg_leak),
    .cfg_gain       (cfg_gain),
    .cfg_stream_len (cfg_stream_len),
    .cfg_dt_ms      (cfg_dt_ms),
    .cfg_scale_q16  (cfg_scale_q16),

    .stat_busy      (stat_busy),
    .stat_done      (stat_done),
    .stat_rate_q16  (stat_rate_q16)
);

// ----------------------------------------------------------------
// Pack unpacked arrays into flat buses
// ----------------------------------------------------------------
wire [N_INPUTS*DATA_WIDTH-1:0] flat_x_input;
wire [N_INPUTS*DATA_WIDTH-1:0] flat_weight;

genvar i;
generate
    for (i = 0; i < N_INPUTS; i = i + 1) begin : PACK_BUS
        assign flat_x_input[i*DATA_WIDTH +: DATA_WIDTH] = cfg_x_input[i];
        assign flat_weight[i*DATA_WIDTH +: DATA_WIDTH]  = cfg_weight[i];
    end
endgenerate


// ----------------------------------------------------------------
// SC dense layer core
// ----------------------------------------------------------------
wire [DATA_WIDTH-1:0] I_t;
wire [N_NEURONS-1:0] neuron_spikes_t;
wire                 running;
wire                 step_valid;
wire                 run_done;

// Busy/done outputs for AXI
assign stat_busy = running;
assign stat_done = run_done;

sc_dense_layer_core #(
    .N_INPUTS(N_INPUTS),
    .N_NEURONS(N_NEURONS),
    .DATA_WIDTH(DATA_WIDTH)
) u_core (
    .clk          (S_AXI_ACLK),
    .rst_n        (S_AXI_ARESETN),
    .start_pulse  (cfg_start_pulse),
    .stream_len   (cfg_stream_len),
    .x_input_fp   (flat_x_input),
    .weight_fp    (flat_weight),
    .y_min_fp     (cfg_y_min),
    .y_max_fp     (cfg_y_max),
    .cfg_leak     (cfg_leak),
    .cfg_gain     (cfg_gain),
    .I_t          (I_t),
    .spikes       (neuron_spikes_t),
    .running      (running),
    .step_valid   (step_valid),
    .run_done     (run_done)
);


// ----------------------------------------------------------------
// Firing-rate estimator bank
// ----------------------------------------------------------------
sc_firing_rate_bank #(
    .N_NEURONS(N_NEURONS),
    .CNT_WIDTH(16),
    .SCALE_WIDTH(32)
) u_ratebank (
    .clk         (S_AXI_ACLK),
    .rst_n       (S_AXI_ARESETN),
    .spikes      (neuron_spikes_t),
    .step_valid  (step_valid),
    .run_active  (running),
    .run_done    (run_done),
    .SCALE_Q16   (cfg_scale_q16),
    .rate_q16    (stat_rate_q16)
);

endmodule
