// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AXI4-Lite slave wrapper for sc_shd_top
// Target: Zynq XC7Z020 (PYNQ-Z2), PS ↔ PL interface
//
// Register map (byte addresses, 32-bit aligned):
//
//   Offset  Dir  Name              Description
//   ------  ---  ----              -----------
//   0x00    R/W  CTRL              [0]=start (W: pulse, self-clears)
//                                  [1]=running (R), [2]=done (R)
//   0x04    R/W  T_ORIG            [15:0] input length (max ~1000)
//   0x08    R/W  SCALE_L1          [31:0] Q16.16 scale, layer 1
//   0x0C    R/W  SCALE_L2          [31:0] Q16.16 scale, layer 2
//   0x10    R/W  SCALE_L3          [31:0] Q16.16 scale, layer 3
//   0x14    W    SPIKE_IN_0        spike_in[31:0]   (written each cycle while running)
//   0x18    W    SPIKE_IN_1        spike_in[63:32]
//   0x1C    W    SPIKE_IN_2        spike_in[95:64]
//   0x20    W    SPIKE_IN_3        spike_in[127:96]
//   0x24    W    SPIKE_IN_4        spike_in[139:128] (only [11:0] used)
//   0x28    W    SPIKE_COMMIT      Write any value → transfers spike_in to core + advances 1 cycle
//   0x40    R    OUT_V_0           output_v_sum[31:0]   (class 0)
//   0x44    R    OUT_V_1           output_v_sum[63:32]  (class 1)
//   ...     ...  ...               ...
//   0x8C    R    OUT_V_19          output_v_sum[639:608] (class 19)
//
// Usage from Python (PYNQ):
//   1. Write SCALE_L1/L2/L3 and T_ORIG
//   2. Write CTRL[0]=1 (start pulse)
//   3. While running: write SPIKE_IN_0..4, then SPIKE_COMMIT
//   4. Poll CTRL until done=1
//   5. Read OUT_V_0..19, argmax = predicted class

module sc_shd_axi_wrapper #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 8
)(
    // AXI4-Lite slave interface
    input  wire                                S_AXI_ACLK,
    input  wire                                S_AXI_ARESETN,
    // Write address
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       S_AXI_AWADDR,
    input  wire [2:0]                          S_AXI_AWPROT,
    input  wire                                S_AXI_AWVALID,
    output wire                                S_AXI_AWREADY,
    // Write data
    input  wire [C_S_AXI_DATA_WIDTH-1:0]       S_AXI_WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0]     S_AXI_WSTRB,
    input  wire                                S_AXI_WVALID,
    output wire                                S_AXI_WREADY,
    // Write response
    output wire [1:0]                          S_AXI_BRESP,
    output wire                                S_AXI_BVALID,
    input  wire                                S_AXI_BREADY,
    // Read address
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       S_AXI_ARADDR,
    input  wire [2:0]                          S_AXI_ARPROT,
    input  wire                                S_AXI_ARVALID,
    output wire                                S_AXI_ARREADY,
    // Read data
    output wire [C_S_AXI_DATA_WIDTH-1:0]       S_AXI_RDATA,
    output wire [1:0]                          S_AXI_RRESP,
    output wire                                S_AXI_RVALID,
    input  wire                                S_AXI_RREADY
);

    // ----------------------------------------------------------------
    // AXI4-Lite handshake registers
    // ----------------------------------------------------------------
    reg axi_awready, axi_wready, axi_bvalid, axi_arready, axi_rvalid;
    reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr, axi_araddr;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = 2'b00;  // OKAY
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = 2'b00;  // OKAY
    assign S_AXI_RVALID  = axi_rvalid;

    // ----------------------------------------------------------------
    // User registers
    // ----------------------------------------------------------------
    reg        start_pulse;
    reg [15:0] t_orig_reg;
    reg signed [31:0] scale_l1_reg, scale_l2_reg, scale_l3_reg;
    reg [31:0] spike_in_reg [0:4];  // 5 x 32-bit = 160 bits (140 used)
    reg        spike_commit;

    // sc_shd_top signals
    wire        core_running;
    wire        core_done;
    wire signed [20*32-1:0] core_output;

    // Spike input assembly
    wire [139:0] spike_in_assembled = {spike_in_reg[4][11:0],
                                        spike_in_reg[3],
                                        spike_in_reg[2],
                                        spike_in_reg[1],
                                        spike_in_reg[0]};

    // ----------------------------------------------------------------
    // sc_shd_top instance
    // ----------------------------------------------------------------
    sc_shd_top u_core (
        .clk              (S_AXI_ACLK),
        .rst_n            (S_AXI_ARESETN),
        .start            (start_pulse),
        .t_orig           (t_orig_reg),
        .spike_in         (spike_in_assembled),
        .scale_l1_q16_16  (scale_l1_reg),
        .scale_l2_q16_16  (scale_l2_reg),
        .scale_l3_q16_16  (scale_l3_reg),
        .running          (core_running),
        .done             (core_done),
        .output_v_sum_packed (core_output)
    );

    // ----------------------------------------------------------------
    // Write address handshake
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 1'b0;
            axi_awaddr  <= 0;
        end else if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID) begin
            axi_awready <= 1'b1;
            axi_awaddr  <= S_AXI_AWADDR;
        end else begin
            axi_awready <= 1'b0;
        end
    end

    // ----------------------------------------------------------------
    // Write data handshake
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN)
            axi_wready <= 1'b0;
        else if (~axi_wready && S_AXI_WVALID && S_AXI_AWVALID)
            axi_wready <= 1'b1;
        else
            axi_wready <= 1'b0;
    end

    // ----------------------------------------------------------------
    // Write logic — decode address, store to registers
    // ----------------------------------------------------------------
    wire wr_en = axi_awready && S_AXI_WVALID && axi_wready && S_AXI_AWVALID;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            start_pulse   <= 1'b0;
            t_orig_reg    <= 16'd250;
            scale_l1_reg  <= 32'sd0;
            scale_l2_reg  <= 32'sd0;
            scale_l3_reg  <= 32'sd0;
            spike_in_reg[0] <= 32'd0;
            spike_in_reg[1] <= 32'd0;
            spike_in_reg[2] <= 32'd0;
            spike_in_reg[3] <= 32'd0;
            spike_in_reg[4] <= 32'd0;
            spike_commit  <= 1'b0;
        end else begin
            // Self-clearing pulses
            start_pulse  <= 1'b0;
            spike_commit <= 1'b0;

            if (wr_en) begin
                case (axi_awaddr[7:2])  // Word address
                    6'h00: start_pulse     <= S_AXI_WDATA[0];      // CTRL
                    6'h01: t_orig_reg      <= S_AXI_WDATA[15:0];   // T_ORIG
                    6'h02: scale_l1_reg    <= S_AXI_WDATA;         // SCALE_L1
                    6'h03: scale_l2_reg    <= S_AXI_WDATA;         // SCALE_L2
                    6'h04: scale_l3_reg    <= S_AXI_WDATA;         // SCALE_L3
                    6'h05: spike_in_reg[0] <= S_AXI_WDATA;         // SPIKE_IN_0
                    6'h06: spike_in_reg[1] <= S_AXI_WDATA;         // SPIKE_IN_1
                    6'h07: spike_in_reg[2] <= S_AXI_WDATA;         // SPIKE_IN_2
                    6'h08: spike_in_reg[3] <= S_AXI_WDATA;         // SPIKE_IN_3
                    6'h09: spike_in_reg[4] <= S_AXI_WDATA;         // SPIKE_IN_4
                    6'h0A: spike_commit    <= 1'b1;                // SPIKE_COMMIT
                    default: ;
                endcase
            end
        end
    end

    // ----------------------------------------------------------------
    // Write response
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN)
            axi_bvalid <= 1'b0;
        else if (wr_en && ~axi_bvalid)
            axi_bvalid <= 1'b1;
        else if (S_AXI_BREADY && axi_bvalid)
            axi_bvalid <= 1'b0;
    end

    // ----------------------------------------------------------------
    // Read address handshake
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_arready <= 1'b0;
            axi_araddr  <= 0;
        end else if (~axi_arready && S_AXI_ARVALID) begin
            axi_arready <= 1'b1;
            axi_araddr  <= S_AXI_ARADDR;
        end else begin
            axi_arready <= 1'b0;
        end
    end

    // ----------------------------------------------------------------
    // Read data — decode address, return register value
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rvalid <= 1'b0;
            axi_rdata  <= 32'd0;
        end else if (axi_arready && S_AXI_ARVALID && ~axi_rvalid) begin
            axi_rvalid <= 1'b1;
            case (axi_araddr[7:2])
                // Control / status
                6'h00: axi_rdata <= {29'd0, core_done, core_running, 1'b0};
                6'h01: axi_rdata <= {16'd0, t_orig_reg};
                6'h02: axi_rdata <= scale_l1_reg;
                6'h03: axi_rdata <= scale_l2_reg;
                6'h04: axi_rdata <= scale_l3_reg;
                // Output voltages (20 classes)
                6'h10: axi_rdata <= core_output[1*32-1 -: 32];   // class 0
                6'h11: axi_rdata <= core_output[2*32-1 -: 32];   // class 1
                6'h12: axi_rdata <= core_output[3*32-1 -: 32];   // class 2
                6'h13: axi_rdata <= core_output[4*32-1 -: 32];   // class 3
                6'h14: axi_rdata <= core_output[5*32-1 -: 32];   // class 4
                6'h15: axi_rdata <= core_output[6*32-1 -: 32];   // class 5
                6'h16: axi_rdata <= core_output[7*32-1 -: 32];   // class 6
                6'h17: axi_rdata <= core_output[8*32-1 -: 32];   // class 7
                6'h18: axi_rdata <= core_output[9*32-1 -: 32];   // class 8
                6'h19: axi_rdata <= core_output[10*32-1 -: 32];  // class 9
                6'h1A: axi_rdata <= core_output[11*32-1 -: 32];  // class 10
                6'h1B: axi_rdata <= core_output[12*32-1 -: 32];  // class 11
                6'h1C: axi_rdata <= core_output[13*32-1 -: 32];  // class 12
                6'h1D: axi_rdata <= core_output[14*32-1 -: 32];  // class 13
                6'h1E: axi_rdata <= core_output[15*32-1 -: 32];  // class 14
                6'h1F: axi_rdata <= core_output[16*32-1 -: 32];  // class 15
                6'h20: axi_rdata <= core_output[17*32-1 -: 32];  // class 16
                6'h21: axi_rdata <= core_output[18*32-1 -: 32];  // class 17
                6'h22: axi_rdata <= core_output[19*32-1 -: 32];  // class 18
                6'h23: axi_rdata <= core_output[20*32-1 -: 32];  // class 19
                default: axi_rdata <= 32'hDEAD_BEEF;
            endcase
        end else if (axi_rvalid && S_AXI_RREADY) begin
            axi_rvalid <= 1'b0;
        end
    end

endmodule
