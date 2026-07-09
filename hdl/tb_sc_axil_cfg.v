// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for SC AXIL CFG

// Testbench for sc_axil_cfg

`timescale 1ns / 1ps

module tb_sc_axil_cfg;

    localparam C_S_AXI_ADDR_WIDTH = 8;
    localparam C_S_AXI_DATA_WIDTH = 32;
    localparam CLK_PERIOD         = 10;

    // AXI-Lite signals
    reg                               clk;
    reg                               resetn;
    reg  [C_S_AXI_ADDR_WIDTH-1:0]     awaddr;
    reg                               awvalid;
    wire                              awready;
    reg  [C_S_AXI_DATA_WIDTH-1:0]     wdata;
    reg  [3:0]                        wstrb;
    reg                               wvalid;
    wire                              wready;
    wire [1:0]                        bresp;
    wire                              bvalid;
    reg                               bready;
    reg  [C_S_AXI_ADDR_WIDTH-1:0]     araddr;
    reg                               arvalid;
    wire                              arready;
    wire [C_S_AXI_DATA_WIDTH-1:0]     rdata;
    wire [1:0]                        rresp;
    wire                              rvalid;
    reg                               rready;

    // Config outputs
    wire                              cfg_start_pulse;
    wire [15:0]                       cfg_x_input [0:2];
    wire [15:0]                       cfg_weight  [0:2];
    wire [15:0]                       cfg_y_min;
    wire [15:0]                       cfg_y_max;
    wire [15:0]                       cfg_leak;
    wire [15:0]                       cfg_gain;
    wire [31:0]                       cfg_stream_len;
    wire [31:0]                       cfg_dt_ms;
    wire [31:0]                       cfg_scale_q16;

    // Status inputs
    reg                               stat_busy;
    reg                               stat_done;
    reg  [31:0]                       stat_rate_q16 [0:6];

    integer pass_count;
    integer fail_count;

    // Read result capture
    reg [C_S_AXI_DATA_WIDTH-1:0] rd_data;

    sc_axil_cfg #(
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH)
    ) uut (
        .S_AXI_ACLK     (clk),
        .S_AXI_ARESETN   (resetn),
        .S_AXI_AWADDR   (awaddr),
        .S_AXI_AWVALID  (awvalid),
        .S_AXI_AWREADY  (awready),
        .S_AXI_WDATA    (wdata),
        .S_AXI_WSTRB    (wstrb),
        .S_AXI_WVALID   (wvalid),
        .S_AXI_WREADY   (wready),
        .S_AXI_BRESP    (bresp),
        .S_AXI_BVALID   (bvalid),
        .S_AXI_BREADY   (bready),
        .S_AXI_ARADDR   (araddr),
        .S_AXI_ARVALID  (arvalid),
        .S_AXI_ARREADY  (arready),
        .S_AXI_RDATA    (rdata),
        .S_AXI_RRESP    (rresp),
        .S_AXI_RVALID   (rvalid),
        .S_AXI_RREADY   (rready),
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

    always #(CLK_PERIOD/2) clk = ~clk;

    // ----------------------------------------------------------------
    // AXI-Lite write task
    // ----------------------------------------------------------------
    task axi_write;
        input [C_S_AXI_ADDR_WIDTH-1:0] addr;
        input [C_S_AXI_DATA_WIDTH-1:0] data;
        begin
            @(posedge clk); #1;
            awaddr  = addr;
            awvalid = 1'b1;
            wdata   = data;
            wstrb   = 4'hF;
            wvalid  = 1'b1;
            bready  = 1'b1;

            // Wait for both AWREADY and WREADY
            wait (awready && wready);
            @(posedge clk); #1;
            awvalid = 1'b0;
            wvalid  = 1'b0;

            // Wait for BVALID
            wait (bvalid);
            @(posedge clk); #1;
            bready = 1'b0;
        end
    endtask

    // ----------------------------------------------------------------
    // AXI-Lite read task
    // ----------------------------------------------------------------
    task axi_read;
        input  [C_S_AXI_ADDR_WIDTH-1:0] addr;
        output [C_S_AXI_DATA_WIDTH-1:0] data;
        begin
            @(posedge clk); #1;
            araddr  = addr;
            arvalid = 1'b1;
            rready  = 1'b1;

            // Wait for ARREADY
            wait (arready);
            @(posedge clk); #1;
            arvalid = 1'b0;

            // Wait for RVALID
            wait (rvalid);
            data = rdata;
            @(posedge clk); #1;
            rready = 1'b0;
        end
    endtask

    integer i;

    initial begin
        clk       = 0;
        resetn    = 0;
        awaddr    = 0;
        awvalid   = 0;
        wdata     = 0;
        wstrb     = 0;
        wvalid    = 0;
        bready    = 0;
        araddr    = 0;
        arvalid   = 0;
        rready    = 0;
        stat_busy = 0;
        stat_done = 0;
        pass_count = 0;
        fail_count = 0;
        for (i = 0; i < 7; i = i + 1)
            stat_rate_q16[i] = 32'd0;

        // Reset
        repeat (4) @(posedge clk);
        resetn = 1;
        repeat (2) @(posedge clk);

        // --- Test A: write ADDR_X0 (0x10), read back ---
        axi_write(8'h10, 32'h0000_0080); // Q8.8 = 0.5
        axi_read(8'h10, rd_data);

        if (rd_data[15:0] == 16'h0080) begin
            $display("[PASS] A: X0 write/read = 0x%04h", rd_data[15:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] A: X0 read = 0x%08h, expected 0x00000080", rd_data);
            fail_count = fail_count + 1;
        end

        // --- Test B: write Y_MIN (0x30), Y_MAX (0x34), read back ---
        axi_write(8'h30, 32'h0000_FF00); // y_min = -1.0
        axi_write(8'h34, 32'h0000_0100); // y_max = +1.0
        axi_read(8'h30, rd_data);
        if (rd_data[15:0] == 16'hFF00) begin
            $display("[PASS] B1: Y_MIN = 0x%04h", rd_data[15:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] B1: Y_MIN = 0x%04h, expected 0xFF00", rd_data[15:0]);
            fail_count = fail_count + 1;
        end
        axi_read(8'h34, rd_data);
        if (rd_data[15:0] == 16'h0100) begin
            $display("[PASS] B2: Y_MAX = 0x%04h", rd_data[15:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] B2: Y_MAX = 0x%04h, expected 0x0100", rd_data[15:0]);
            fail_count = fail_count + 1;
        end

        // --- Test C: write CTRL bit 0 -> cfg_start_pulse asserts 1 cycle ---
        // Monitor cfg_start_pulse on next write
        fork
            begin
                axi_write(8'h00, 32'h0000_0001);
            end
            begin
                // Wait for cfg_start_pulse to rise
                @(posedge cfg_start_pulse);
                $display("[PASS] C: cfg_start_pulse asserted");
                pass_count = pass_count + 1;
            end
        join

        // Verify it cleared after BVALID/BREADY handshake
        repeat (3) @(posedge clk);
        if (cfg_start_pulse === 1'b0) begin
            $display("[PASS] C2: cfg_start_pulse cleared");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] C2: cfg_start_pulse still high");
            fail_count = fail_count + 1;
        end

        // --- Test D: read STATUS register ---
        stat_busy = 1'b1;
        stat_done = 1'b0;
        axi_read(8'h04, rd_data);
        if (rd_data[0] === 1'b1 && rd_data[1] === 1'b0) begin
            $display("[PASS] D1: STATUS busy=1, done=0");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] D1: STATUS=0x%08h, expected bit0=1 bit1=0", rd_data);
            fail_count = fail_count + 1;
        end

        stat_busy = 1'b0;
        stat_done = 1'b1;
        axi_read(8'h04, rd_data);
        if (rd_data[0] === 1'b0 && rd_data[1] === 1'b1) begin
            $display("[PASS] D2: STATUS busy=0, done=1");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] D2: STATUS=0x%08h, expected bit0=0 bit1=1", rd_data);
            fail_count = fail_count + 1;
        end

        // --- Test E: read RATE registers ---
        stat_rate_q16[0] = 32'hDEAD_BEEF;
        stat_rate_q16[3] = 32'h1234_5678;
        axi_read(8'h80, rd_data);
        if (rd_data == 32'hDEAD_BEEF) begin
            $display("[PASS] E1: RATE0 = 0x%08h", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] E1: RATE0 = 0x%08h, expected 0xDEADBEEF", rd_data);
            fail_count = fail_count + 1;
        end
        axi_read(8'h8C, rd_data);
        if (rd_data == 32'h1234_5678) begin
            $display("[PASS] E2: RATE3 = 0x%08h", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] E2: RATE3 = 0x%08h, expected 0x12345678", rd_data);
            fail_count = fail_count + 1;
        end

        // --- Test F: write STREAM_LEN, SCALE_Q16, read back ---
        axi_write(8'h40, 32'd1024);
        axi_write(8'h48, 32'h0001_0000);
        axi_read(8'h40, rd_data);
        if (rd_data == 32'd1024) begin
            $display("[PASS] F1: STREAM_LEN = %0d", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] F1: STREAM_LEN = %0d, expected 1024", rd_data);
            fail_count = fail_count + 1;
        end
        axi_read(8'h48, rd_data);
        if (rd_data == 32'h0001_0000) begin
            $display("[PASS] F2: SCALE_Q16 = 0x%08h", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] F2: SCALE_Q16 = 0x%08h, expected 0x00010000", rd_data);
            fail_count = fail_count + 1;
        end

        $display("---------------------------------------");
        $display("tb_sc_axil_cfg: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("[PASS] All tests passed");
        else
            $display("[FAIL] %0d test(s) failed", fail_count);
        $finish;
    end

    // Safety timeout
    initial begin
        #100000;
        $display("[FAIL] Global timeout");
        $finish;
    end

endmodule
