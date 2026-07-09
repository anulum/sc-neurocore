// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AXI-Lite slave for configuration and firing-rate readback

//hdl/sc_axil_cfg.v
//
// AXI-Lite slave for configuration and firing-rate readback.
// 32-bit data, 32-bit address, simple single-beat transactions.

`timescale 1ns / 1ps

module sc_axil_cfg #(
    parameter integer C_S_AXI_ADDR_WIDTH = 8, // 256-byte space
    parameter integer C_S_AXI_DATA_WIDTH = 32
)(
    input wire                                     S_AXI_ACLK,
    input wire                                     S_AXI_ARESETN,

    // Write address channel
    input wire [C_S_AXI_ADDR_WIDTH-1:0]   S_AXI_AWADDR,
    input wire                                     S_AXI_AWVALID,
    output reg                                     S_AXI_AWREADY,

    // Write data channel
    input wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input wire [3:0]                               S_AXI_WSTRB,
    input wire                                     S_AXI_WVALID,
    output reg                                     S_AXI_WREADY,

    // Write response channel
    output reg [1:0]                               S_AXI_BRESP,
    output reg                                     S_AXI_BVALID,
    input wire                                     S_AXI_BREADY,

    // Read address channel
    input wire [C_S_AXI_ADDR_WIDTH-1:0]   S_AXI_ARADDR,
    input wire                                     S_AXI_ARVALID,
    output reg                                     S_AXI_ARREADY,

    // Read data channel
    output reg [C_S_AXI_DATA_WIDTH-1:0]  S_AXI_RDATA,
    output reg [1:0]                               S_AXI_RRESP,
    output reg                                     S_AXI_RVALID,
    input wire                                     S_AXI_RREADY,

    // --------------------------------------------------
    // Register outputs (to SC core)
    // --------------------------------------------------
    output reg                                     cfg_start_pulse,
    output reg [15:0]                              cfg_x_input [0:2], // Q8.8
    output reg [15:0]                              cfg_weight [0:2],  // Q8.8
    output reg [15:0]                              cfg_y_min,         // Q8.8
    output reg [15:0]                              cfg_y_max,         // Q8.8
    output reg [15:0]                              cfg_leak,          // Q8.8 Leak rate
    output reg [15:0]                              cfg_gain,          // Q8.8 Input gain
    output reg [31:0]                              cfg_stream_len,
    output reg [31:0]                              cfg_dt_ms,
    output reg [31:0]                              cfg_scale_q16, // SCALE_Q16 for rate bank

    // --------------------------------------------------
    // Status inputs (from SC core)
    // --------------------------------------------------
    input wire                                     stat_busy,
    input wire                                     stat_done,
    input wire [31:0]                              stat_rate_q16 [0:6] // 7 neurons
);

// ----------------------------------------------------------------
// 3.2 Internal reg map + simple decode
// ----------------------------------------------------------------
// Address constants (byte offsets)
localparam ADDR_CTRL        = 8'h00;
localparam ADDR_STATUS      = 8'h04;

localparam ADDR_X0          = 8'h10;
localparam ADDR_X1          = 8'h14;
localparam ADDR_X2          = 8'h18;

localparam ADDR_W0          = 8'h20;
localparam ADDR_W1          = 8'h24;
localparam ADDR_W2          = 8'h28;

localparam ADDR_Y_MIN       = 8'h30;
localparam ADDR_Y_MAX       = 8'h34;

localparam ADDR_STREAM_LEN  = 8'h40;
localparam ADDR_DT_MS       = 8'h44;
localparam ADDR_SCALE_Q16   = 8'h48;

localparam ADDR_LEAK        = 8'h50;
localparam ADDR_GAIN        = 8'h54;

localparam ADDR_RATE0       = 8'h80;
localparam ADDR_RATE1       = 8'h84;
localparam ADDR_RATE2       = 8'h88;
localparam ADDR_RATE3       = 8'h8C;
localparam ADDR_RATE4       = 8'h90;
localparam ADDR_RATE5       = 8'h94;
localparam ADDR_RATE6       = 8'h98;


// Very compact write path (one beat, no bursts):
// Simple write FSM vars
reg aw_en;
reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;


// AXI ready/valid handshaking simplified
always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        S_AXI_AWREADY <= 1'b0;
        S_AXI_WREADY  <= 1'b0;
        S_AXI_BVALID  <= 1'b0;
        S_AXI_BRESP   <= 2'b00;
        aw_en         <= 1'b1;
        cfg_start_pulse <= 1'b0;
        // init cfg regs if you want
    end else begin
        // Write address ready
        if (~S_AXI_AWREADY && S_AXI_AWVALID && S_AXI_WVALID && aw_en) begin
            S_AXI_AWREADY <= 1'b1;
            axi_awaddr <= S_AXI_AWADDR;
        end else if (S_AXI_BVALID && S_AXI_BREADY) begin
            S_AXI_AWREADY <= 1'b0;
        end

        // Write data ready
        if (~S_AXI_WREADY && S_AXI_WVALID && S_AXI_AWVALID && aw_en) begin
            S_AXI_WREADY <= 1'b1;
        end else if (S_AXI_BVALID && S_AXI_BREADY) begin
            S_AXI_WREADY <= 1'b0;
        end

        // Write transaction complete
        if (S_AXI_AWREADY && S_AXI_AWVALID && S_AXI_WREADY && S_AXI_WVALID && ~S_AXI_BVALID) begin
            // Decode address and write to the selected config reg
            case (axi_awaddr[7:0])
                ADDR_CTRL: begin
                    // bit 0 -> start
                    if (S_AXI_WDATA[0])
                        cfg_start_pulse <= 1'b1; // single-cycle pulse
                end
                ADDR_X0: cfg_x_input[0] <= S_AXI_WDATA[15:0];
                ADDR_X1: cfg_x_input[1] <= S_AXI_WDATA[15:0];
                ADDR_X2: cfg_x_input[2] <= S_AXI_WDATA[15:0];

                ADDR_W0: cfg_weight[0] <= S_AXI_WDATA[15:0];
                ADDR_W1: cfg_weight[1] <= S_AXI_WDATA[15:0];
                ADDR_W2: cfg_weight[2] <= S_AXI_WDATA[15:0];

                ADDR_Y_MIN: cfg_y_min <= S_AXI_WDATA[15:0];
                ADDR_Y_MAX: cfg_y_max <= S_AXI_WDATA[15:0];

                ADDR_LEAK:  cfg_leak  <= S_AXI_WDATA[15:0];
                ADDR_GAIN:  cfg_gain  <= S_AXI_WDATA[15:0];

                ADDR_STREAM_LEN: cfg_stream_len <= S_AXI_WDATA;
                ADDR_DT_MS:      cfg_dt_ms      <= S_AXI_WDATA;
                ADDR_SCALE_Q16:  cfg_scale_q16  <= S_AXI_WDATA;
                default:;
            endcase

            S_AXI_BVALID <= 1'b1;
            S_AXI_BRESP  <= 2'b00;
        end else if (S_AXI_BVALID && S_AXI_BREADY) begin
            S_AXI_BVALID <= 1'b0;
            cfg_start_pulse <= 1'b0; // clear pulse
        end
    end
end


// Read path (just a combinational mux with basic handshaking):
// Read address handshake
always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        S_AXI_ARREADY <= 1'b0;
        axi_araddr    <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        S_AXI_RVALID  <= 1'b0;
        S_AXI_RRESP   <= 2'b00;
    end else begin
        if (~S_AXI_ARREADY && S_AXI_ARVALID) begin
            S_AXI_ARREADY <= 1'b1;
            axi_araddr    <= S_AXI_ARADDR;
        end else begin
            S_AXI_ARREADY <= 1'b0;
        end

        if (S_AXI_ARREADY && S_AXI_ARVALID && ~S_AXI_RVALID) begin
            // address latched, return data
            S_AXI_RVALID <= 1'b1;
            S_AXI_RRESP  <= 2'b00;
            case (axi_araddr[7:0])
                ADDR_STATUS: begin
                    S_AXI_RDATA[0] <= stat_busy;
                    S_AXI_RDATA[1] <= stat_done;
                    S_AXI_RDATA[31:2]<= 30'b0;
                end

                ADDR_X0: S_AXI_RDATA <= {16'b0, cfg_x_input[0]};
                ADDR_X1: S_AXI_RDATA <= {16'b0, cfg_x_input[1]};
                ADDR_X2: S_AXI_RDATA <= {16'b0, cfg_x_input[2]};

                ADDR_W0: S_AXI_RDATA <= {16'b0, cfg_weight[0]};
                ADDR_W1: S_AXI_RDATA <= {16'b0, cfg_weight[1]};
                ADDR_W2: S_AXI_RDATA <= {16'b0, cfg_weight[2]};

                ADDR_Y_MIN: S_AXI_RDATA <= {16'b0, cfg_y_min};
                ADDR_Y_MAX: S_AXI_RDATA <= {16'b0, cfg_y_max};

                ADDR_LEAK:  S_AXI_RDATA <= {16'b0, cfg_leak};
                ADDR_GAIN:  S_AXI_RDATA <= {16'b0, cfg_gain};

                ADDR_STREAM_LEN: S_AXI_RDATA <= cfg_stream_len;
                ADDR_DT_MS:      S_AXI_RDATA <= cfg_dt_ms;
                ADDR_SCALE_Q16:  S_AXI_RDATA <= cfg_scale_q16;

                ADDR_RATE0: S_AXI_RDATA <= stat_rate_q16[0];
                ADDR_RATE1: S_AXI_RDATA <= stat_rate_q16[1];
                ADDR_RATE2: S_AXI_RDATA <= stat_rate_q16[2];
                ADDR_RATE3: S_AXI_RDATA <= stat_rate_q16[3];
                ADDR_RATE4: S_AXI_RDATA <= stat_rate_q16[4];
                ADDR_RATE5: S_AXI_RDATA <= stat_rate_q16[5];
                ADDR_RATE6: S_AXI_RDATA <= stat_rate_q16[6];

                default: S_AXI_RDATA <= 32'h00000000;
            endcase
        end else if (S_AXI_RVALID && S_AXI_RREADY) begin
            S_AXI_RVALID <= 1'b0;
        end
    end
end

endmodule
