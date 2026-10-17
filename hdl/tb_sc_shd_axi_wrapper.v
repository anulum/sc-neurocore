// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SHD AXI input-commit and inference regression
`timescale 1ns / 1ps

module tb_sc_shd_axi_wrapper;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst_n = 0;
    reg [7:0] awaddr = 0, araddr = 0;
    reg awvalid = 0, wvalid = 0, bready = 0, arvalid = 0, rready = 0;
    reg [31:0] wdata = 0;
    reg [3:0] wstrb = 0;
    wire awready, wready, bvalid, arready, rvalid;
    wire [1:0] bresp, rresp;
    wire [31:0] rdata;
    reg [31:0] expected [0:59];
    reg [31:0] held_output [0:19];
    reg [31:0] value;
    integer sample, t, word_index, byte_index, cycles = 0;
    reg [139:0] vector;
    reg [159:0] packed_vector;
    reg [31:0] word_value;

    sc_shd_axi_wrapper dut (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(awaddr), .S_AXI_AWPROT(3'd0),
        .S_AXI_AWVALID(awvalid), .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata), .S_AXI_WSTRB(wstrb),
        .S_AXI_WVALID(wvalid), .S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp), .S_AXI_BVALID(bvalid), .S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr), .S_AXI_ARPROT(3'd0),
        .S_AXI_ARVALID(arvalid), .S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata), .S_AXI_RRESP(rresp),
        .S_AXI_RVALID(rvalid), .S_AXI_RREADY(rready)
    );

    always @(posedge clk) cycles <= cycles + 1;
    initial begin
        #2000000;
        $fatal(1, "AXI inference did not complete within the test deadline");
    end

    // AW and W arrive in either order; hold the response under backpressure.
    task write_reg;
        input [31:0] address;
        input [31:0] data;
        input [3:0] strobe;
        input integer skew, stall;
        input [1:0] response;
        integer k;
        begin
            @(negedge clk);
            awaddr = address[7:0];
            wdata = data;
            wstrb = strobe;
            awvalid = skew >= 0;
            wvalid = skew <= 0;
            bready = 0;
            repeat (skew < 0 ? -skew : skew) @(negedge clk);
            awvalid = 1;
            wvalid = 1;
            @(posedge clk);
            while (!(awready && wready)) @(posedge clk);
            @(negedge clk);
            awvalid = 0;
            wvalid = 0;
            wait (bvalid);
            for (k = 0; k < stall; k = k + 1) begin
                @(negedge clk);
                if (!bvalid || bresp !== response)
                    $fatal(1, "write response lost under backpressure at %h", address);
            end
            if (bresp !== response)
                $fatal(1, "unexpected write response at %h: %b", address, bresp);
            @(negedge clk);
            bready = 1;
            @(posedge clk);
            @(negedge clk);
            bready = 0;
        end
    endtask

    task read_reg;
        input [31:0] address;
        input integer stall;
        output [31:0] data;
        integer k;
        begin
            @(negedge clk);
            araddr = address[7:0];
            arvalid = 1;
            rready = 0;
            @(posedge clk);
            while (!arready) @(posedge clk);
            @(negedge clk);
            arvalid = 0;
            wait (rvalid);
            data = rdata;
            for (k = 0; k < stall; k = k + 1) begin
                @(negedge clk);
                if (!rvalid || rdata !== data || rresp !== 0)
                    $fatal(1, "read response lost under backpressure at %h", address);
            end
            if (rresp !== 0) $fatal(1, "unexpected read response");
            @(negedge clk);
            rready = 1;
            @(posedge clk);
            @(negedge clk);
            rready = 0;
        end
    endtask

    // Present another request while a response is outstanding. The slave
    // must stall it, then accept it exactly once after the first response.
    task queued_transactions;
        integer k;
        begin
            @(negedge clk);
            awaddr = 4; wdata = 6; wstrb = 4'hf;
            awvalid = 1; wvalid = 1; bready = 0;
            @(posedge clk);
            while (!(awready && wready)) @(posedge clk);
            @(negedge clk);
            awvalid = 0; wvalid = 0;
            wait (bvalid);
            @(negedge clk);
            awvalid = 1; wvalid = 1; wdata = 7;
            for (k = 0; k < 4; k = k + 1) begin
                @(negedge clk);
                if (awready || wready || !bvalid || bresp !== 0)
                    $fatal(1, "new write accepted before outstanding response");
            end
            bready = 1;
            @(posedge clk);
            @(negedge clk);
            bready = 0;
            @(posedge clk);
            while (!(awready && wready)) @(posedge clk);
            @(negedge clk);
            awvalid = 0; wvalid = 0;
            wait (bvalid);
            if (bresp !== 0) $fatal(1, "queued write failed");
            @(negedge clk); bready = 1;
            @(posedge clk);
            @(negedge clk); bready = 0;

            araddr = 4; arvalid = 1; rready = 0;
            @(posedge clk);
            while (!arready) @(posedge clk);
            @(negedge clk); arvalid = 0;
            wait (rvalid);
            if (rdata !== 7) $fatal(1, "queued write was lost");
            @(negedge clk); arvalid = 1; araddr = 32;
            for (k = 0; k < 4; k = k + 1) begin
                @(negedge clk);
                if (arready || !rvalid || rdata !== 7 || rresp !== 0)
                    $fatal(1, "new read replaced outstanding response");
            end
            rready = 1;
            @(posedge clk);
            @(negedge clk); rready = 0;
            @(posedge clk);
            while (!arready) @(posedge clk);
            @(negedge clk); arvalid = 0;
            wait (rvalid);
            if (rdata !== 32'hdeadbeef || rresp !== 0)
                $fatal(1, "queued read returned the wrong address");
            @(negedge clk); rready = 1;
            @(posedge clk);
            @(negedge clk); rready = 0;
        end
    endtask

    function [139:0] input_vector;
        input integer which_sample, time_bin;
        integer c;
        begin
            for (c = 0; c < 140; c = c + 1)
                input_vector[c] = (which_sample != 2) &&
                    (((c*7 + time_bin*11 + which_sample*13) % 17) <
                     (which_sample == 0 ? 6 : 11));
        end
    endfunction

    task reset_bus;
        begin
            @(negedge clk);
            rst_n = 0;
            repeat (3) @(negedge clk);
            rst_n = 1;
            repeat (3) @(negedge clk);
            read_reg(0, 2, value);
            if (value !== 0) $fatal(1, "reset did not clear busy/done");
        end
    endtask

    task arm;
        input integer length;
        begin
            write_reg(4, length, 4'hf, 2, 1, 0);
            write_reg(8, 4096, 4'hf, -2, 2, 0);
            write_reg(12, 4096, 4'hf, 1, 0, 0);
            write_reg(16, 2048, 4'hf, 0, 3, 0);
            read_reg(4, 1, value);
            if (value !== length) $fatal(1, "input length readback differs");
            read_reg(8, 2, value);
            if (value !== 4096) $fatal(1, "layer 1 scale readback differs");
            read_reg(12, 1, value);
            if (value !== 4096) $fatal(1, "layer 2 scale readback differs");
            read_reg(16, 0, value);
            if (value !== 2048) $fatal(1, "layer 3 scale readback differs");
            write_reg(0, 1, 4'h1, -1, 2, 0);
            repeat (80) @(negedge clk);
            read_reg(0, 3, value);
            if (value !== 2) $fatal(1, "armed core advanced without a commit");
        end
    endtask

    task snapshot_outputs;
        integer j;
        begin
            for (j = 0; j < 20; j = j + 1)
                read_reg(32'h40 + 4*j, j % 3, held_output[j]);
        end
    endtask

    task check_held_outputs;
        integer j;
        begin
            for (j = 0; j < 20; j = j + 1) begin
                read_reg(32'h40 + 4*j, j % 2, value);
                if (value !== held_output[j])
                    $fatal(1, "network advanced while staging an uncommitted input");
            end
        end
    endtask

    task check_result;
        input integer which_sample;
        integer j;
        begin
            repeat (80) @(negedge clk);
            read_reg(0, 5, value);
            if (value !== 4) $fatal(1, "completion is not sticky or core remains busy");
            for (j = 0; j < 20; j = j + 1) begin
                read_reg(32'h40 + 4*j, j % 4, value);
                if (value !== expected[which_sample*20+j])
                    $fatal(1, "sample %0d class %0d: got %h, Python reference %h",
                           which_sample, j, value, expected[which_sample*20+j]);
            end
        end
    endtask

    initial begin
        $readmemh("expected.hex", expected);
        reset_bus();
        write_reg(32'h28, 1, 4'hf, 0, 4, 0); // Idle commits are harmless.
        read_reg(0, 0, value);
        if (value !== 0) $fatal(1, "idle commit started inference");
        // Start must wait for a committed first vector, even after many clocks.
        arm(6);
        write_reg(32'h14, 32'hffffffff, 4'hf, 0, 0, 0);
        write_reg(32'h28, 1, 4'hf, 0, 0, 0);
        reset_bus();
        write_reg(4, 0, 4'hf, 0, 0, 0);
        write_reg(0, 1, 4'hf, 0, 0, 2'b10);
        write_reg(4, 65504, 4'hf, 0, 0, 0);
        write_reg(0, 1, 4'hf, 0, 0, 2'b10);
        write_reg(32'h40, 1, 4'hf, 0, 0, 2'b10);
        write_reg(32'h15, 1, 4'hf, 0, 0, 2'b10);
        read_reg(32'hfc, 1, value);
        if (value !== 32'hdeadbeef) $fatal(1, "reserved read changed");
        queued_transactions();
        arm(65503);
        reset_bus();

        for (sample = 0; sample < 3; sample = sample + 1) begin
            arm(sample == 0 ? 6 : (sample == 1 ? 1 : 3));
            // A zero-strobe start has no effect; a busy start is refused.
            write_reg(0, 1, 0, 0, 1, 0);
            write_reg(0, 1, 4'hf, 0, 1, 2'b10);
            // Configuration writes are staged for the next sample.
            write_reg(4, 100, 4'hf, 0, 1, 0);
            write_reg(8, 0, 4'hf, 0, 1, 0);
            write_reg(12, 0, 4'hf, 0, 1, 0);
            write_reg(16, 0, 4'hf, 0, 1, 0);
            for (t = 0; t < (sample == 0 ? 6 : (sample == 1 ? 1 : 3)); t = t + 1) begin
                vector = input_vector(sample, t);
                packed_vector = {20'd0, vector};
                snapshot_outputs();
                for (word_index = 0; word_index < 5; word_index = word_index + 1) begin
                    word_value = packed_vector[32*word_index +: 32];
                    write_reg(32'h14 + 4*word_index, 32'hffffffff, 4'hf, 1, 0, 0);
                    for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                        write_reg(32'h14 + 4*word_index, word_value,
                                  4'b0001 << byte_index, byte_index-2, byte_index, 0);
                    write_reg(32'h14 + 4*word_index, ~word_value, 0, -1, 2, 0);
                end
                write_reg(32'h28, 0, 0, 0, 2, 0);
                check_held_outputs();
                write_reg(32'h28, 0, 4'hf, (t % 3)-1, t % 5, 0);
            end
            check_result(sample);
            for (t = 0; t < 30; t = t + 1)
                write_reg(32'h28, 0, 4'hf, 0, 0, 0);
            check_result(sample);
        end
        $display("PASS: SHD AXI three Python-golden samples; stalls, byte writes, reset, tail and sticky done; %0d bus clocks", cycles);
        $finish;
    end
endmodule
