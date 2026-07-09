// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Testbench for SC Neurocore Top

// Testbench for sc_neurocore_top (full system)

`timescale 1ns / 1ps

module tb_sc_neurocore_top;

    localparam C_S_AXI_ADDR_WIDTH = 8;
    localparam C_S_AXI_DATA_WIDTH = 32;
    localparam N_INPUTS           = 3;
    localparam N_NEURONS          = 7;
    localparam DATA_WIDTH         = 16;
    localparam CLK_PERIOD         = 10;
    localparam STREAM_LEN         = 128;

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

    integer pass_count;
    integer fail_count;
    reg [C_S_AXI_DATA_WIDTH-1:0] rd_data;

    sc_neurocore_top #(
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .N_INPUTS          (N_INPUTS),
        .N_NEURONS         (N_NEURONS),
        .DATA_WIDTH        (DATA_WIDTH)
    ) uut (
        .S_AXI_ACLK    (clk),
        .S_AXI_ARESETN  (resetn),
        .S_AXI_AWADDR  (awaddr),
        .S_AXI_AWVALID (awvalid),
        .S_AXI_AWREADY (awready),
        .S_AXI_WDATA   (wdata),
        .S_AXI_WSTRB   (wstrb),
        .S_AXI_WVALID  (wvalid),
        .S_AXI_WREADY  (wready),
        .S_AXI_BRESP   (bresp),
        .S_AXI_BVALID  (bvalid),
        .S_AXI_BREADY  (bready),
        .S_AXI_ARADDR  (araddr),
        .S_AXI_ARVALID (arvalid),
        .S_AXI_ARREADY (arready),
        .S_AXI_RDATA   (rdata),
        .S_AXI_RRESP   (rresp),
        .S_AXI_RVALID  (rvalid),
        .S_AXI_RREADY  (rready)
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
            wait (awready && wready);
            @(posedge clk); #1;
            awvalid = 1'b0;
            wvalid  = 1'b0;
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
            wait (arready);
            @(posedge clk); #1;
            arvalid = 1'b0;
            wait (rvalid);
            data = rdata;
            @(posedge clk); #1;
            rready = 1'b0;
        end
    endtask

    // ----------------------------------------------------------------
    // Poll STATUS register until done bit (bit 1) is set
    // ----------------------------------------------------------------
    task poll_done;
        output integer cycles;
        reg [31:0] status;
        begin
            cycles = 0;
            status = 32'd0;
            while (status[1] == 1'b0 && cycles < STREAM_LEN + 200) begin
                axi_read(8'h04, status);
                cycles = cycles + 1;
            end
        end
    endtask

    integer poll_cycles;
    integer i;
    integer any_nonzero;
    reg [31:0] rate_val;

    initial begin
        clk        = 0;
        resetn     = 0;
        awaddr     = 0;
        awvalid    = 0;
        wdata      = 0;
        wstrb      = 0;
        wvalid     = 0;
        bready     = 0;
        araddr     = 0;
        arvalid    = 0;
        rready     = 0;
        pass_count = 0;
        fail_count = 0;

        // Reset
        repeat (4) @(posedge clk);
        resetn = 1;
        repeat (2) @(posedge clk);

        // --- Step A: configure registers via AXI ---
        // X inputs: 0.75 Q8.8 = 0x00C0
        axi_write(8'h10, 32'h0000_00C0); // X0
        axi_write(8'h14, 32'h0000_00C0); // X1
        axi_write(8'h18, 32'h0000_00C0); // X2

        // Weights: 1.0 Q8.8 = 0x0100
        axi_write(8'h20, 32'h0000_0100); // W0
        axi_write(8'h24, 32'h0000_0100); // W1
        axi_write(8'h28, 32'h0000_0100); // W2

        // Y range: y_min=-1.0 (0xFF00), y_max=+2.0 (0x0200)
        axi_write(8'h30, 32'h0000_FF00); // Y_MIN
        axi_write(8'h34, 32'h0000_0200); // Y_MAX

        // Stream length
        axi_write(8'h40, STREAM_LEN);

        // Leak = 0.9 ~ 0x00E6, Gain = 1.5 = 0x0180
        axi_write(8'h50, 32'h0000_00E6); // LEAK
        axi_write(8'h54, 32'h0000_0180); // GAIN

        // SCALE_Q16 = 65536/128 = 512 (Q16.16 reciprocal of stream_len)
        axi_write(8'h48, 32'd512);

        // Verify one config register read-back
        axi_read(8'h10, rd_data);
        if (rd_data[15:0] == 16'h00C0) begin
            $display("[PASS] A: config readback X0 = 0x%04h", rd_data[15:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] A: config readback X0 = 0x%04h, expected 0x00C0", rd_data[15:0]);
            fail_count = fail_count + 1;
        end

        // --- Step B: verify STATUS.busy=0 before start ---
        axi_read(8'h04, rd_data);
        if (rd_data[0] === 1'b0) begin
            $display("[PASS] B: STATUS.busy=0 before start");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] B: STATUS.busy=%b before start", rd_data[0]);
            fail_count = fail_count + 1;
        end

        // --- Step C: write CTRL to start ---
        axi_write(8'h00, 32'h0000_0001);

        // Brief delay then check busy
        repeat (3) @(posedge clk);
        axi_read(8'h04, rd_data);
        if (rd_data[0] === 1'b1) begin
            $display("[PASS] C: STATUS.busy=1 after start");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] C: STATUS.busy=%b, expected 1", rd_data[0]);
            fail_count = fail_count + 1;
        end

        // --- Step D: poll STATUS until done ---
        poll_done(poll_cycles);
        axi_read(8'h04, rd_data);
        if (rd_data[1] === 1'b1) begin
            $display("[PASS] D: STATUS.done=1 after %0d polls", poll_cycles);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] D: STATUS.done not set after %0d polls", poll_cycles);
            fail_count = fail_count + 1;
        end

        // --- Step E: read rate registers, check at least one non-zero ---
        any_nonzero = 0;
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            axi_read(8'h80 + i * 4, rate_val);
            $display("  RATE[%0d] = 0x%08h (%0d)", i, rate_val, rate_val);
            if (rate_val != 32'd0)
                any_nonzero = 1;
        end

        if (any_nonzero) begin
            $display("[PASS] E: at least one neuron fired (rate != 0)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] E: all rates are zero");
            fail_count = fail_count + 1;
        end

        // --- Step F: verify busy cleared after done ---
        axi_read(8'h04, rd_data);
        if (rd_data[0] === 1'b0) begin
            $display("[PASS] F: STATUS.busy=0 after completion");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] F: STATUS.busy still set after done");
            fail_count = fail_count + 1;
        end

        $display("---------------------------------------");
        $display("tb_sc_neurocore_top: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("[PASS] All tests passed");
        else
            $display("[FAIL] %0d test(s) failed", fail_count);
        $finish;
    end

    // Safety timeout
    initial begin
        #500000;
        $display("[FAIL] Global timeout");
        $finish;
    end

endmodule
