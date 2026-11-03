// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Reset, polling and repeated measurement through the AXI-Lite ports

`timescale 1ns / 1ps

module tb_sc_neurocore_top_lifecycle;
    localparam integer STREAM_LEN = 64;
    localparam integer CLK_PERIOD = 10;
    reg clk = 0;
    reg resetn = 0;
    reg [7:0] awaddr = 0;
    reg awvalid = 0;
    wire awready;
    reg [31:0] wdata = 0;
    reg [3:0] wstrb = 0;
    reg wvalid = 0;
    wire wready;
    wire [1:0] bresp;
    wire bvalid;
    reg bready = 0;
    reg [7:0] araddr = 0;
    reg arvalid = 0;
    wire arready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire rvalid;
    reg rready = 0;
    integer checks = 0;
    integer i;

    sc_neurocore_top uut (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(resetn),
        .S_AXI_AWADDR(awaddr), .S_AXI_AWVALID(awvalid), .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata), .S_AXI_WSTRB(wstrb), .S_AXI_WVALID(wvalid),
        .S_AXI_WREADY(wready), .S_AXI_BRESP(bresp), .S_AXI_BVALID(bvalid),
        .S_AXI_BREADY(bready), .S_AXI_ARADDR(araddr), .S_AXI_ARVALID(arvalid),
        .S_AXI_ARREADY(arready), .S_AXI_RDATA(rdata), .S_AXI_RRESP(rresp),
        .S_AXI_RVALID(rvalid), .S_AXI_RREADY(rready)
    );

    always #(CLK_PERIOD/2) clk = ~clk;

    task axi_write;
        input [7:0] addr;
        input [31:0] data;
        begin
            @(posedge clk); #1;
            awaddr = addr;
            awvalid = 1;
            wdata = data;
            wstrb = 4'hF;
            wvalid = 1;
            bready = 1;
            wait (awready && wready);
            @(posedge clk); #1;
            awvalid = 0;
            wvalid = 0;
            wait (bvalid);
            if (bresp !== 2'b00)
                $fatal(1, "[FAIL] AXI write response at %h: %b", addr, bresp);
            @(posedge clk); #1;
            bready = 0;
        end
    endtask

    task check_register;
        input [7:0] addr;
        input [31:0] expected;
        reg [31:0] observed;
        begin
            @(posedge clk); #1;
            araddr = addr;
            arvalid = 1;
            rready = 1;
            wait (arready);
            @(posedge clk); #1;
            arvalid = 0;
            wait (rvalid);
            observed = rdata;
            if (rresp !== 2'b00 || observed !== expected)
                $fatal(1, "[FAIL] AXI register %h: got %h response %b, expected %h",
                       addr, observed, rresp, expected);
            checks = checks + 1;
            @(posedge clk); #1;
            rready = 0;
        end
    endtask

    task check_reset_defaults;
        integer channel;
        begin
            check_register(8'h04, 32'd0);
            for (channel = 0; channel < 3; channel = channel + 1) begin
                check_register(8'h10 + {channel[5:0], 2'b00}, 32'd0);
                check_register(8'h20 + {channel[5:0], 2'b00}, 32'd0);
            end
            check_register(8'h30, 32'd0);
            check_register(8'h34, 32'd0);
            check_register(8'h50, 32'd0);
            check_register(8'h54, 32'd0);
            check_register(8'h40, 32'd1);
            check_register(8'h44, 32'h0001_0000);
            check_register(8'h48, 32'h0001_0000);
        end
    endtask

    initial begin
        repeat (4) @(negedge clk);
        resetn = 1;
        repeat (2) @(negedge clk);
        check_reset_defaults;

        for (i = 0; i < 3; i = i + 1) begin
            axi_write(8'h10 + {i[5:0], 2'b00}, 32'h0000_C000);
            axi_write(8'h20 + {i[5:0], 2'b00}, 32'h0000_FFFF);
        end
        // Equal current endpoints make I=1 exactly, independently of the
        // stochastic bits. With gain=1 and leak=0 each measured clock spikes.
        axi_write(8'h30, 32'h0000_0100);
        axi_write(8'h34, 32'h0000_0100);
        axi_write(8'h40, STREAM_LEN);
        axi_write(8'h48, 32'd65536 / STREAM_LEN);
        axi_write(8'h50, 32'd0);
        axi_write(8'h54, 32'h0000_0100);
        repeat (4) @(negedge clk);

        axi_write(8'h00, 32'd1);
        check_register(8'h04, 32'd1);
        axi_write(8'h00, 32'd1); // A busy start must not open another window.
        check_register(8'h04, 32'd1);
        repeat (STREAM_LEN + 16) @(negedge clk);
        check_register(8'h04, 32'd2);
        for (i = 0; i < 7; i = i + 1)
            check_register(8'h80 + {i[5:0], 2'b00}, 32'h0001_0000);

        repeat (16) @(negedge clk);
        check_register(8'h04, 32'd2);
        for (i = 0; i < 7; i = i + 1)
            check_register(8'h80 + {i[5:0], 2'b00}, 32'h0001_0000);
        axi_write(8'h00, 32'd0);
        check_register(8'h04, 32'd2);

        // A second measured window sees zero drive and overwrites all old rates.
        axi_write(8'h54, 32'd0);
        repeat (4) @(negedge clk);
        axi_write(8'h00, 32'd1);
        check_register(8'h04, 32'd1);
        repeat (STREAM_LEN + 16) @(negedge clk);
        check_register(8'h04, 32'd2);
        for (i = 0; i < 7; i = i + 1)
            check_register(8'h80 + {i[5:0], 2'b00}, 32'd0);

        @(negedge clk);
        resetn = 0;
        repeat (4) @(negedge clk);
        resetn = 1;
        repeat (2) @(negedge clk);
        check_reset_defaults;
        for (i = 0; i < 7; i = i + 1)
            check_register(8'h80 + {i[5:0], 2'b00}, 32'd0);

        $display("AXI LIFECYCLE PASS: %0d register observations", checks);
        $finish;
    end

    initial begin
        repeat (4 * STREAM_LEN + 2048) @(posedge clk);
        $fatal(1, "[FAIL] AXI lifecycle watchdog expired");
    end
endmodule
