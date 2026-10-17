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
//   0x00    R/W  CTRL              [0]=start (W: reset and arm a sample)
//                                  [1]=busy (R), [2]=done (R, sticky until start)
//   0x04    R/W  T_ORIG            [15:0] input length (max ~1000)
//   0x08    R/W  SCALE_L1          [31:0] Q16.16 scale, layer 1
//   0x0C    R/W  SCALE_L2          [31:0] Q16.16 scale, layer 2
//   0x10    R/W  SCALE_L3          [31:0] Q16.16 scale, layer 3
//   0x14    W    SPIKE_IN_0        spike_in[31:0]   (written each cycle while running)
//   0x18    W    SPIKE_IN_1        spike_in[63:32]
//   0x1C    W    SPIKE_IN_2        spike_in[95:64]
//   0x20    W    SPIKE_IN_3        spike_in[127:96]
//   0x24    W    SPIKE_IN_4        spike_in[139:128] (only [11:0] used)
//   0x28    W    SPIKE_COMMIT      Snapshot all 140 bits and advance one input step
//   0x40    R    OUT_V_0           output_v_sum[31:0]   (class 0)
//   0x44    R    OUT_V_1           output_v_sum[63:32]  (class 1)
//   ...     ...  ...               ...
//   0x8C    R    OUT_V_19          output_v_sum[639:608] (class 19)
//
// Usage from Python (PYNQ):
//   1. Write SCALE_L1/L2/L3 and T_ORIG
//   2. Write CTRL[0]=1 (start pulse)
//   3. Write T_ORIG vectors: SPIKE_IN_0..4, then SPIKE_COMMIT for each
//   4. Poll CTRL until done=1
//   5. Read OUT_V_0..19, argmax = predicted class
//
// All state holds between input commits. After the last vector, the core
// consumes 30 zero-padding steps and three pipeline-drain clocks itself.
// Commits during/after that tail are ignored (legacy drivers send padding).
// Each accepted start resets the network, including delays and membranes.
// Configuration is snapshotted at start; byte strobes apply to staging words.
// Start while busy, invalid length (outside 1..65503), unaligned writes and
// writes to reserved/read-only addresses return SLVERR without mutation.
// All logic uses S_AXI_ACLK; no fabric-gated clock is generated.

`timescale 1ns / 1ps

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
    reg [1:0] axi_bresp;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr, axi_araddr;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
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
    reg [139:0] spike_vector;
    reg        start_pending, done_sticky;
    reg [15:0] input_remaining, core_t_orig;
    reg [5:0]  tail_steps;
    reg signed [31:0] core_scale_l1, core_scale_l2, core_scale_l3;

    // sc_shd_top signals
    wire        core_running;
    wire        core_done;
    wire signed [20*32-1:0] core_output;
    wire busy = start_pulse || start_pending || core_running;
    wire core_advance = spike_commit || (tail_steps != 0);

    function [31:0] write_bytes;
        input [31:0] previous, data;
        input [3:0] strobe;
        integer b;
        begin
            write_bytes = previous;
            for (b = 0; b < 4; b = b + 1)
                if (strobe[b]) write_bytes[8*b +: 8] = data[8*b +: 8];
        end
    endfunction
    wire [31:0] merged_t_orig = write_bytes({16'd0, t_orig_reg}, S_AXI_WDATA, S_AXI_WSTRB);
    wire write_error = (axi_awaddr[1:0] != 0) || (axi_awaddr[7:2] > 6'h0A) ||
        ((axi_awaddr[7:2] == 0) && S_AXI_WSTRB[0] && S_AXI_WDATA[0] &&
         (busy || (t_orig_reg == 0) || (t_orig_reg > 16'd65503)));

    // Spike input assembly
    wire [139:0] spike_in_assembled = {spike_in_reg[4][11:0],
                                        spike_in_reg[3],
                                        spike_in_reg[2],
                                        spike_in_reg[1],
                                        spike_in_reg[0]};

    // ----------------------------------------------------------------
    // sc_shd_top instance
    // ----------------------------------------------------------------
    sc_shd_top #(.USE_STEP_ENABLE(1)) u_core (
        .clk              (S_AXI_ACLK),
        .rst_n            (S_AXI_ARESETN && !start_pulse),
        .step_en          (core_advance),
        .start            (spike_commit && start_pending),
        .t_orig           (core_t_orig),
        .spike_in         ((tail_steps != 0) ? 140'd0 : spike_vector),
        .scale_l1_q16_16  (core_scale_l1),
        .scale_l2_q16_16  (core_scale_l2),
        .scale_l3_q16_16  (core_scale_l3),
        .running          (core_running),
        .done             (core_done),
        .output_v_sum_packed (core_output)
    );

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            start_pending <= 1'b0;
            done_sticky <= 1'b0;
            input_remaining <= 0;
            tail_steps <= 0;
            core_t_orig <= 0;
            core_scale_l1 <= 0;
            core_scale_l2 <= 0;
            core_scale_l3 <= 0;
        end else if (start_pulse) begin
            start_pending <= 1'b1;
            done_sticky <= 1'b0;
            input_remaining <= t_orig_reg;
            tail_steps <= 0;
            core_t_orig <= t_orig_reg;
            core_scale_l1 <= scale_l1_reg;
            core_scale_l2 <= scale_l2_reg;
            core_scale_l3 <= scale_l3_reg;
        end else begin
            if (core_done) done_sticky <= 1'b1;
            if (spike_commit) begin
                start_pending <= 1'b0;
                input_remaining <= input_remaining - 1'b1;
                if (input_remaining == 1) tail_steps <= 6'd33;
            end else if (tail_steps != 0) begin
                tail_steps <= tail_steps - 1'b1;
            end
        end
    end

    // ----------------------------------------------------------------
    // Write address handshake
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 1'b0;
            axi_awaddr  <= 0;
        end else if (~axi_awready && ~axi_bvalid && S_AXI_AWVALID && S_AXI_WVALID) begin
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
        else if (~axi_wready && ~axi_bvalid && S_AXI_WVALID && S_AXI_AWVALID)
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
            spike_vector  <= 140'd0;
        end else begin
            // Self-clearing pulses
            start_pulse  <= 1'b0;
            spike_commit <= 1'b0;

            if (wr_en && !write_error) begin
                case (axi_awaddr[7:2])  // Word address
                    6'h00: start_pulse <= S_AXI_WSTRB[0] && S_AXI_WDATA[0];
                    6'h01: t_orig_reg <= merged_t_orig[15:0];
                    6'h02: scale_l1_reg <= write_bytes(scale_l1_reg, S_AXI_WDATA, S_AXI_WSTRB);
                    6'h03: scale_l2_reg <= write_bytes(scale_l2_reg, S_AXI_WDATA, S_AXI_WSTRB);
                    6'h04: scale_l3_reg <= write_bytes(scale_l3_reg, S_AXI_WDATA, S_AXI_WSTRB);
                    6'h05: spike_in_reg[0] <= write_bytes(spike_in_reg[0], S_AXI_WDATA, S_AXI_WSTRB);
                    6'h06: spike_in_reg[1] <= write_bytes(spike_in_reg[1], S_AXI_WDATA, S_AXI_WSTRB);
                    6'h07: spike_in_reg[2] <= write_bytes(spike_in_reg[2], S_AXI_WDATA, S_AXI_WSTRB);
                    6'h08: spike_in_reg[3] <= write_bytes(spike_in_reg[3], S_AXI_WDATA, S_AXI_WSTRB);
                    6'h09: spike_in_reg[4] <= write_bytes(spike_in_reg[4], S_AXI_WDATA, S_AXI_WSTRB);
                    6'h0A: if (busy && input_remaining != 0 && tail_steps == 0 && |S_AXI_WSTRB) begin
                        spike_vector <= spike_in_assembled;
                        spike_commit <= 1'b1;
                    end
                    default: ;
                endcase
            end
        end
    end

    // ----------------------------------------------------------------
    // Write response
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_bvalid <= 1'b0;
            axi_bresp <= 2'b00;
        end else if (wr_en && ~axi_bvalid) begin
            axi_bvalid <= 1'b1;
            axi_bresp <= write_error ? 2'b10 : 2'b00;
        end else if (S_AXI_BREADY && axi_bvalid)
            axi_bvalid <= 1'b0;
    end

    // ----------------------------------------------------------------
    // Read address handshake
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_arready <= 1'b0;
            axi_araddr  <= 0;
        end else if (~axi_arready && ~axi_rvalid && S_AXI_ARVALID) begin
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
                6'h00: axi_rdata <= {29'd0, done_sticky, busy, 1'b0};
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
