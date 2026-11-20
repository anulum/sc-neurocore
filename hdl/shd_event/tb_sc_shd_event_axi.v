// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AXI4-Lite cosimulation and negative-protocol testbench for sc_shd_event_axi
//
// Zero preprocessor conditionals. Supports three parameterised run modes:
//   1. Default (BINARY_SPIKE_MODE=0, RUN_NEGATIVE_TESTS=0):
//      Integrity-checked load with auto_increment=1, CRC-32 check against expected_crc32.hex,
//      seal, and bit-exact inference streaming counts via FRAME_COUNT_0..17 + SPIKE_COMMIT.
//   2. Binary expansion (BINARY_SPIKE_MODE=1, RUN_NEGATIVE_TESTS=0):
//      Integrity-checked load with explicit LOAD_TARGET (auto_increment=0), seal, and binary
//      inference streaming 140 spike bits via SPIKE_IN_0..4 + SPIKE_COMMIT.
//   3. Negative & protocol suite (RUN_NEGATIVE_TESTS=1):
//      Exercises unsealed start, empty seal, CRC mismatch, post-seal loader mutations,
//      out-of-range LOAD_TARGET and auto-increment past d2[127] refusals (refused_range),
//      t_orig 0 and 1025 refusals, reserved 0x008..0x010 and 0x0B0/0x0B4 SLVERR refusals,
//      write-only SPIKE_IN_*/SPIKE_COMMIT/FRAME_COUNT_* zero readback, read-only refusals,
//      unaligned writes, nonzero FRAME_COUNT_17[31:16], commit while !frame_ready,
//      sticky commit_refused and refused_busy flags, hardware reset seal clear, and
//      AXI4-Lite channel skew/backpressure. Optionally pokes reserved 0x008..0x010 before
//      inference when POKE_RESERVED_SCALES=1.

`timescale 1ns / 1ps

module tb_sc_shd_event_axi;
    parameter integer FRAC_BITS          = 0;
    parameter integer THETA1             = 1;
    parameter integer THETA2             = 1;
    parameter integer V1_WIDTH           = 22;
    parameter integer V2_WIDTH           = 17;
    parameter integer WSHAPE             = 6;
    parameter integer CAP1               = 140;
    parameter integer CAP2               = 128;
    parameter integer QSHAPE             = 12;
    parameter integer P                  = 8;
    parameter integer MAX_WORDS          = 4000000;
    parameter integer BINARY_SPIKE_MODE    = 0;
    parameter integer RUN_NEGATIVE_TESTS   = 0;
    parameter integer POKE_RESERVED_SCALES = 0;

    localparam [8:0] ADDR_CTRL            = 9'h000;
    localparam [8:0] ADDR_T_ORIG          = 9'h004;
    localparam [8:0] ADDR_SCALE_L1        = 9'h008;
    localparam [8:0] ADDR_SCALE_L2        = 9'h00C;
    localparam [8:0] ADDR_SCALE_L3        = 9'h010;
    localparam [8:0] ADDR_SPIKE_IN_BASE   = 9'h014;
    localparam [8:0] ADDR_SPIKE_COMMIT    = 9'h028;
    localparam [8:0] ADDR_OUT_V_BASE      = 9'h040;
    localparam [8:0] ADDR_LOAD_CTRL       = 9'h090;
    localparam [8:0] ADDR_LOAD_TARGET     = 9'h094;
    localparam [8:0] ADDR_LOAD_WDATA_LO   = 9'h098;
    localparam [8:0] ADDR_LOAD_WDATA_HI   = 9'h09C;
    localparam [8:0] ADDR_LOAD_COUNT      = 9'h0A0;
    localparam [8:0] ADDR_LOAD_CRC32      = 9'h0A4;
    localparam [8:0] ADDR_EXPECTED_CRC32  = 9'h0A8;
    localparam [8:0] ADDR_LOAD_STATUS     = 9'h0AC;
    localparam [8:0] ADDR_RSVD_B0         = 9'h0B0;
    localparam [8:0] ADDR_RSVD_B4         = 9'h0B4;
    localparam [8:0] ADDR_FRAME_CNT_BASE  = 9'h0B8;
    localparam [8:0] ADDR_FRAME_CNT_17    = 9'h0FC;

    localparam [1:0] RESP_OKAY   = 2'b00;
    localparam [1:0] RESP_SLVERR = 2'b10;

    reg         clk     = 1'b0;
    reg         rst_n   = 1'b0;
    reg  [8:0]  awaddr  = 9'd0;
    reg         awvalid = 1'b0;
    reg  [31:0] wdata   = 32'd0;
    reg  [3:0]  wstrb   = 4'd0;
    reg         wvalid  = 1'b0;
    reg         bready  = 1'b0;
    reg  [8:0]  araddr  = 9'd0;
    reg         arvalid = 1'b0;
    reg         rready  = 1'b0;

    wire        awready;
    wire        wready;
    wire [1:0]  bresp;
    wire        bvalid;
    wire        arready;
    wire [31:0] rdata;
    wire [1:0]  rresp;
    wire        rvalid;

    sc_shd_event_axi #(
        .C_S_AXI_DATA_WIDTH(32),
        .C_S_AXI_ADDR_WIDTH(9),
        .P                 (P),
        .FRAC_BITS         (FRAC_BITS),
        .THETA1            (THETA1),
        .THETA2            (THETA2),
        .V1_WIDTH          (V1_WIDTH),
        .V2_WIDTH          (V2_WIDTH),
        .CAP1              (CAP1),
        .CAP2              (CAP2),
        .WSHAPE            (WSHAPE),
        .QSHAPE            (QSHAPE)
    ) dut (
        .S_AXI_ACLK   (clk),
        .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR (awaddr),
        .S_AXI_AWPROT (3'd0),
        .S_AXI_AWVALID(awvalid),
        .S_AXI_AWREADY(awready),
        .S_AXI_WDATA  (wdata),
        .S_AXI_WSTRB  (wstrb),
        .S_AXI_WVALID (wvalid),
        .S_AXI_WREADY (wready),
        .S_AXI_BRESP  (bresp),
        .S_AXI_BVALID (bvalid),
        .S_AXI_BREADY (bready),
        .S_AXI_ARADDR (araddr),
        .S_AXI_ARPROT (3'd0),
        .S_AXI_ARVALID(arvalid),
        .S_AXI_ARREADY(arready),
        .S_AXI_RDATA  (rdata),
        .S_AXI_RRESP  (rresp),
        .S_AXI_RVALID (rvalid),
        .S_AXI_RREADY (rready)
    );

    always #5.0 clk = ~clk;

    reg [559:0]  stim     [0:MAX_WORDS-1];
    reg [1023:0] q1       [0:139];
    reg [1023:0] q2       [0:127];
    reg [159:0]  q3       [0:127];
    reg [4:0]    d1       [0:139];
    reg [4:0]    d2       [0:127];
    reg [31:0]   exp_crc  [0:0];

    integer fd, n, s, f, k, ptr, r, gq, poll_guard;
    reg [31:0] rd_val;
    reg [63:0] w64;
    reg [559:0] cur_frame;
    reg [159:0] bin_packed;

    task axi_write_ext(
        input [8:0]   addr,
        input [31:0]  data,
        input [3:0]   strobe,
        input integer skew,
        input integer stall,
        input [1:0]   expected_resp
    );
        integer m;
        begin
            @(negedge clk);
            awaddr  = addr;
            wdata   = data;
            wstrb   = strobe;
            awvalid = (skew >= 0);
            wvalid  = (skew <= 0);
            bready  = 1'b0;
            repeat (skew < 0 ? -skew : skew) @(negedge clk);
            awvalid = 1'b1;
            wvalid  = 1'b1;
            @(posedge clk);
            while (!(awready && wready)) @(posedge clk);
            @(negedge clk);
            awvalid = 1'b0;
            wvalid  = 1'b0;
            while (!bvalid) @(posedge clk);
            for (m = 0; m < stall; m = m + 1) begin
                @(negedge clk);
                if (!bvalid || (bresp !== expected_resp))
                    $fatal(1, "AXI write response unstable under stall at 0x%03x", addr);
            end
            if (bresp !== expected_resp)
                $fatal(1, "AXI write 0x%03x: got bresp=%b expected=%b", addr, bresp, expected_resp);
            @(negedge clk);
            bready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            bready = 1'b0;
        end
    endtask

    task axi_write_ok(input [8:0] addr, input [31:0] data);
        begin
            axi_write_ext(addr, data, 4'hF, 0, 0, RESP_OKAY);
        end
    endtask

    task axi_write_err(input [8:0] addr, input [31:0] data);
        begin
            axi_write_ext(addr, data, 4'hF, 0, 0, RESP_SLVERR);
        end
    endtask

    task axi_read_ext(
        input  [8:0]   addr,
        input  integer stall,
        output [31:0]  data_out
    );
        integer m;
        begin
            @(negedge clk);
            araddr  = addr;
            arvalid = 1'b1;
            rready  = 1'b0;
            @(posedge clk);
            while (!arready) @(posedge clk);
            @(negedge clk);
            arvalid = 1'b0;
            while (!rvalid) @(posedge clk);
            data_out = rdata;
            if (rresp !== RESP_OKAY)
                $fatal(1, "AXI read 0x%03x returned non-OKAY rresp=%b", addr, rresp);
            for (m = 0; m < stall; m = m + 1) begin
                @(negedge clk);
                if (!rvalid || (rdata !== data_out) || (rresp !== RESP_OKAY))
                    $fatal(1, "AXI read data unstable under stall at 0x%03x", addr);
            end
            @(negedge clk);
            rready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            rready = 1'b0;
        end
    endtask

    task axi_read(input [8:0] addr, output [31:0] data_out);
        begin
            axi_read_ext(addr, 0, data_out);
        end
    endtask

    task push_word_explicit(
        input [2:0]  sel,
        input [7:0]  addr,
        input [6:0]  grp,
        input [63:0] word
    );
        begin
            axi_write_ok(ADDR_LOAD_TARGET, {14'd0, addr, grp, sel});
            axi_write_ok(ADDR_LOAD_WDATA_LO, word[31:0]);
            axi_write_ok(ADDR_LOAD_WDATA_HI, word[63:32]);
        end
    endtask

    task push_word_auto(input [63:0] word);
        begin
            axi_write_ok(ADDR_LOAD_WDATA_LO, word[31:0]);
            axi_write_ok(ADDR_LOAD_WDATA_HI, word[63:32]);
        end
    endtask

    task load_and_seal_network(input use_auto_inc);
        integer expected_words;
        reg [1023:0] shifted;
        begin
            expected_words = 140 * (128 / P) + 128 * (128 / P) +
                             128 * ((20 + P - 1) / P) + 140 + 128;
            if (use_auto_inc) begin
                axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0003);
                axi_write_ok(ADDR_LOAD_TARGET, 32'd0);
            end else begin
                axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0001);
            end

            for (r = 0; r < 140; r = r + 1) begin
                for (gq = 0; gq < 128 / P; gq = gq + 1) begin
                    shifted = q1[r] >> (gq * 8 * P);
                    w64 = shifted[63:0];
                    if (use_auto_inc) push_word_auto(w64);
                    else push_word_explicit(3'd0, r[7:0], gq[6:0], w64);
                end
            end
            for (r = 0; r < 128; r = r + 1) begin
                for (gq = 0; gq < 128 / P; gq = gq + 1) begin
                    shifted = q2[r] >> (gq * 8 * P);
                    w64 = shifted[63:0];
                    if (use_auto_inc) push_word_auto(w64);
                    else push_word_explicit(3'd1, r[7:0], gq[6:0], w64);
                end
            end
            for (r = 0; r < 128; r = r + 1) begin
                for (gq = 0; gq < (20 + P - 1) / P; gq = gq + 1) begin
                    shifted = {864'd0, q3[r]} >> (gq * 8 * P);
                    w64 = shifted[63:0];
                    if (use_auto_inc) push_word_auto(w64);
                    else push_word_explicit(3'd2, r[7:0], gq[6:0], w64);
                end
            end
            for (r = 0; r < 140; r = r + 1) begin
                w64 = {59'd0, d1[r]};
                if (use_auto_inc) push_word_auto(w64);
                else push_word_explicit(3'd3, r[7:0], 7'd0, w64);
            end
            for (r = 0; r < 128; r = r + 1) begin
                w64 = {59'd0, d2[r]};
                if (use_auto_inc) push_word_auto(w64);
                else push_word_explicit(3'd4, r[7:0], 7'd0, w64);
            end

            axi_read(ADDR_LOAD_COUNT, rd_val);
            if (rd_val !== expected_words[31:0])
                $fatal(1, "LOAD_COUNT mismatch: got %0d expected %0d", rd_val, expected_words);

            axi_read(ADDR_LOAD_CRC32, rd_val);
            if (rd_val !== exp_crc[0])
                $fatal(1, "LOAD_CRC32 mismatch: got 0x%08x expected 0x%08x", rd_val, exp_crc[0]);

            axi_write_ok(ADDR_EXPECTED_CRC32, exp_crc[0]);
            axi_write_ok(ADDR_LOAD_CTRL, use_auto_inc ? 32'h0000_0006 : 32'h0000_0004);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'h0000_0001)
                $fatal(1, "Expected sealed LOAD_STATUS=1, got 0x%08x", rd_val);
        end
    endtask

    task wait_frame_ready;
        begin
            poll_guard = 0;
            axi_read(ADDR_CTRL, rd_val);
            while (!rd_val[3]) begin
                poll_guard = poll_guard + 1;
                if (poll_guard > 200000)
                    $fatal(1, "Timed out polling CTRL.frame_ready (CTRL=0x%08x)", rd_val);
                axi_read(ADDR_CTRL, rd_val);
            end
        end
    endtask

    task wait_done;
        begin
            poll_guard = 0;
            axi_read(ADDR_CTRL, rd_val);
            while (!rd_val[2]) begin
                poll_guard = poll_guard + 1;
                if (poll_guard > 500000)
                    $fatal(1, "Timed out polling CTRL.done (CTRL=0x%08x)", rd_val);
                axi_read(ADDR_CTRL, rd_val);
            end
        end
    endtask

    task run_negative_suite;
        reg [31:0] one_word_crc;
        reg [4:0]  saved_d2_0;
        begin
            // 1. Reset defaults
            axi_read(ADDR_T_ORIG, rd_val);
            if (rd_val !== 32'd250) $fatal(1, "Reset T_ORIG != 250");
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "Reset LOAD_STATUS != 0");
            axi_read(ADDR_LOAD_COUNT, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "Reset LOAD_COUNT != 0");
            axi_read(ADDR_LOAD_CRC32, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "Reset LOAD_CRC32 != 0");

            // 2. Start while unsealed -> SLVERR and not busy
            axi_write_err(ADDR_CTRL, 32'd1);
            axi_read(ADDR_CTRL, rd_val);
            if (rd_val[1] !== 1'b0) $fatal(1, "Core started while unsealed");

            // 3. Seal on empty loader (count == 0) -> crc_mismatch=1, sealed=0
            axi_write_ok(ADDR_EXPECTED_CRC32, 32'd0);
            axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0004);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'h0000_0002) $fatal(1, "Empty seal status != 2: 0x%08x", rd_val);

            // 4. Commit 1 word, wrong CRC -> crc_mismatch=1, sealed=0; matching CRC -> sealed=1
            axi_write_ext(ADDR_LOAD_CTRL, 32'h0000_0001, 4'hF, 2, 2, RESP_OKAY);
            axi_write_ext(ADDR_LOAD_TARGET, 32'd0, 4'hF, -2, 1, RESP_OKAY);
            axi_write_ok(ADDR_LOAD_WDATA_LO, 32'h89AB_CDEF);
            axi_write_ok(ADDR_LOAD_WDATA_HI, 32'h0123_4567);
            axi_read_ext(ADDR_LOAD_CRC32, 2, one_word_crc);

            axi_write_ok(ADDR_EXPECTED_CRC32, one_word_crc ^ 32'h0000_0001);
            axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0004);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'h0000_0002) $fatal(1, "Bad-CRC seal status != 2: 0x%08x", rd_val);
            axi_write_err(ADDR_CTRL, 32'd1);

            axi_write_ok(ADDR_EXPECTED_CRC32, one_word_crc);
            axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0004);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'h0000_0001) $fatal(1, "Valid seal status != 1: 0x%08x", rd_val);

            // 5. Every idle loader write after seal clears sealed:
            axi_write_ok(ADDR_LOAD_TARGET, 32'd4);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val[0] !== 1'b0) $fatal(1, "LOAD_TARGET write did not clear sealed");
            axi_write_err(ADDR_CTRL, 32'd1);

            axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0004);
            axi_write_ok(ADDR_LOAD_WDATA_LO, 32'd9);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val[0] !== 1'b0) $fatal(1, "LOAD_WDATA_LO write did not clear sealed");

            axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0004);
            axi_write_ok(ADDR_EXPECTED_CRC32, one_word_crc);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val[0] !== 1'b0) $fatal(1, "EXPECTED_CRC32 write did not clear sealed");

            axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0004);
            axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0002);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val[0] !== 1'b0) $fatal(1, "LOAD_CTRL non-seal write did not clear sealed");

            axi_write_ok(ADDR_LOAD_CTRL, 32'h0000_0004);
            axi_write_ok(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val[0] !== 1'b0) $fatal(1, "LOAD_WDATA_HI write did not clear sealed");

            // 6. Load & seal full network, verify N2 out-of-range target refusals, then reload
            load_and_seal_network(1);
            saved_d2_0 = dut.u_core.d2[0];

            // 6a. Auto-increment past d2[127] (4,941st word): SLVERR, no d2[0] alias, refused_range=1
            axi_write_ok(ADDR_LOAD_WDATA_LO, {27'd0, ~saved_d2_0});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_read(ADDR_LOAD_COUNT, rd_val);
            if (rd_val !== 32'd4940)
                $fatal(1, "Auto-inc past d2[127] mutated LOAD_COUNT: %0d", rd_val);
            axi_read(ADDR_LOAD_CRC32, rd_val);
            if (rd_val !== exp_crc[0])
                $fatal(1, "Auto-inc past d2[127] mutated LOAD_CRC32: 0x%08x", rd_val);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'h0000_0008)
                $fatal(1, "Expected refused_range=1 & sealed=0 (0x8), got 0x%08x", rd_val);
            if (dut.u_core.d2[0] !== saved_d2_0)
                $fatal(1, "Auto-inc past d2[127] aliased into d2[0]");

            // 6b. Manual sel=5, out-of-range grp, and out-of-range addr across selectors 0..4
            axi_write_ok(ADDR_LOAD_TARGET, {8'd0, 7'd0, 3'd5});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_write_ok(ADDR_LOAD_TARGET, {8'd0, 7'd16, 3'd0});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_write_ok(ADDR_LOAD_TARGET, {8'd0, 7'd3, 3'd2});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_write_ok(ADDR_LOAD_TARGET, {8'd0, 7'd1, 3'd3});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_write_ok(ADDR_LOAD_TARGET, {8'd0, 7'd1, 3'd4});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_write_ok(ADDR_LOAD_TARGET, {8'd140, 7'd0, 3'd0});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_write_ok(ADDR_LOAD_TARGET, {8'd128, 7'd0, 3'd1});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_write_ok(ADDR_LOAD_TARGET, {8'd128, 7'd0, 3'd2});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_write_ok(ADDR_LOAD_TARGET, {8'd140, 7'd0, 3'd3});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_write_ok(ADDR_LOAD_TARGET, {8'd128, 7'd0, 3'd4});
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd0);
            axi_read(ADDR_LOAD_COUNT, rd_val);
            if (rd_val !== 32'd4940)
                $fatal(1, "Out-of-range manual target mutated LOAD_COUNT: %0d", rd_val);
            axi_read(ADDR_LOAD_CRC32, rd_val);
            if (rd_val !== exp_crc[0])
                $fatal(1, "Out-of-range manual target mutated LOAD_CRC32: 0x%08x", rd_val);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'h0000_0008)
                $fatal(1, "refused_range lost after manual out-of-range checks: 0x%08x", rd_val);

            // Reload and seal cleanly (verifies LOAD_CTRL.clear clears refused_range)
            load_and_seal_network(1);

            axi_write_ok(ADDR_T_ORIG, 32'd0);
            axi_write_err(ADDR_CTRL, 32'd1);
            axi_read(ADDR_CTRL, rd_val);
            if (rd_val[1] !== 1'b0 || rd_val[4] !== 1'b0)
                $fatal(1, "T_ORIG=0 start was not cleanly refused");

            axi_write_ok(ADDR_T_ORIG, 32'd1025);
            axi_write_err(ADDR_CTRL, 32'd1);
            axi_read(ADDR_CTRL, rd_val);
            if (rd_val[1] !== 1'b0 || rd_val[4] !== 1'b0)
                $fatal(1, "T_ORIG=1025 start was not cleanly refused");

            // 7. Reserved 0x008/0x00C/0x010, 0x0B0/0x0B4, >=0x100, read-only, unaligned, FRAME_COUNT_17[31:16]
            axi_write_err(ADDR_SCALE_L1, 32'h0000_0751);
            axi_write_err(ADDR_SCALE_L2, 32'h0000_07B0);
            axi_write_err(ADDR_SCALE_L3, 32'h0000_011B);
            axi_write_err(ADDR_SCALE_L1, 32'd0);
            axi_write_err(ADDR_SCALE_L2, 32'd0);
            axi_write_err(ADDR_SCALE_L3, 32'd0);
            axi_read(ADDR_SCALE_L1, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "0x008 read != 0");
            axi_read(ADDR_SCALE_L2, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "0x00C read != 0");
            axi_read(ADDR_SCALE_L3, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "0x010 read != 0");

            axi_read(ADDR_RSVD_B0, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "0x0B0 read != 0");
            axi_read(ADDR_RSVD_B4, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "0x0B4 read != 0");
            axi_write_err(ADDR_RSVD_B0, 32'hDEAD_BEEF);
            axi_write_err(ADDR_RSVD_B4, 32'hDEAD_BEEF);
            axi_write_err(9'h02C, 32'd1);
            axi_write_err(9'h100, 32'd1);
            axi_read(9'h100, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "0x100 read != 0");
            axi_write_err(ADDR_OUT_V_BASE, 32'd1);
            axi_write_err(ADDR_LOAD_COUNT, 32'd1);
            axi_write_err(ADDR_LOAD_CRC32, 32'd1);
            axi_write_err(ADDR_LOAD_STATUS, 32'd1);

            axi_write_ok(ADDR_T_ORIG, 32'd2);
            axi_write_err(9'h005, 32'hFFFF_FFFF);
            axi_read(ADDR_T_ORIG, rd_val);
            if (rd_val !== 32'd2) $fatal(1, "Unaligned write mutated T_ORIG");

            axi_write_ok(ADDR_SPIKE_IN_BASE, 32'h0000_0005);
            axi_read(ADDR_SPIKE_IN_BASE, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "Write-only SPIKE_IN_0 read != 0");
            axi_write_ok(ADDR_SPIKE_IN_BASE, 32'd0);
            axi_read(ADDR_SPIKE_COMMIT, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "Write-only SPIKE_COMMIT read != 0");

            axi_write_ok(ADDR_FRAME_CNT_17, 32'h0000_4321);
            axi_read(ADDR_FRAME_CNT_17, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "Write-only FRAME_COUNT_17 read != 0");
            axi_write_err(ADDR_FRAME_CNT_17, 32'h0001_9999);
            if (dut.frame_buf[559:544] !== 16'h4321)
                $fatal(1, "Refused FRAME_COUNT_17 write mutated frame_buf");

            // 8. Commit while idle (!frame_ready): SLVERR, commit_refused=1, frame_buf preserved
            axi_write_err(ADDR_SPIKE_COMMIT, 32'd1);
            axi_read(ADDR_CTRL, rd_val);
            if (rd_val[6] !== 1'b1) $fatal(1, "commit_refused not set on idle commit");
            if (dut.frame_buf[559:544] !== 16'h4321)
                $fatal(1, "Idle refused commit cleared frame_buf");

            // 9. Start T_ORIG=2: clears commit_refused; test loader busy refusal while running
            axi_write_ok(ADDR_CTRL, 32'd1);
            axi_read(ADDR_CTRL, rd_val);
            if (rd_val[6] !== 1'b0 || rd_val[1] !== 1'b1)
                $fatal(1, "Accepted start did not clear commit_refused or set busy");

            axi_write_err(ADDR_CTRL, 32'd1);
            axi_write_err(ADDR_LOAD_TARGET, 32'd7);
            axi_write_err(ADDR_LOAD_WDATA_LO, 32'd7);
            axi_write_err(ADDR_LOAD_WDATA_HI, 32'd7);
            axi_write_err(ADDR_EXPECTED_CRC32, 32'd7);
            axi_write_err(ADDR_LOAD_CTRL, 32'd1);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'h0000_0005)
                $fatal(1, "Expected refused_busy=1 & sealed=1 (0x5), got 0x%08x", rd_val);

            // Commit frame 0, then immediately attempt back-to-back commit while !frame_ready
            wait_frame_ready();
            axi_write_ok(ADDR_SPIKE_COMMIT, 32'd1);
            axi_write_err(ADDR_SPIKE_COMMIT, 32'd1);
            axi_read(ADDR_CTRL, rd_val);
            if (rd_val[6] !== 1'b1)
                $fatal(1, "Mid-step refused commit did not set commit_refused");

            wait_frame_ready();
            axi_write_ok(ADDR_SPIKE_COMMIT, 32'd1);
            wait_done();

            // refused_busy remains sticky after done until LOAD_CTRL.clear
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'h0000_0005)
                $fatal(1, "refused_busy did not remain sticky after done: 0x%08x", rd_val);

            // 10. Hardware reset clears seal and loader counters
            @(negedge clk);
            rst_n = 1'b0;
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            @(posedge clk);
            axi_read(ADDR_LOAD_STATUS, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "ARESETN did not clear LOAD_STATUS");
            axi_write_err(ADDR_CTRL, 32'd1);
        end
    endtask

    initial begin
        $readmemh("q1.hex", q1);
        $readmemh("q2.hex", q2);
        $readmemh("q3.hex", q3);
        $readmemh("d1.hex", d1);
        $readmemh("d2.hex", d2);
        $readmemh("expected_crc32.hex", exp_crc);
        fd = $fopen("outputs.txt", "w");

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        if (RUN_NEGATIVE_TESTS != 0) begin
            run_negative_suite();
            $fwrite(fd, "PASS_NEGATIVE_SUITE\n");
            $fflush(fd);
            $fclose(fd);
            $finish;
        end

        load_and_seal_network(BINARY_SPIKE_MODE == 0);

        if (POKE_RESERVED_SCALES != 0) begin
            axi_write_err(ADDR_SCALE_L1, 32'h0000_0751);
            axi_write_err(ADDR_SCALE_L2, 32'h0000_07B0);
            axi_write_err(ADDR_SCALE_L3, 32'h0000_011B);
            axi_read(ADDR_SCALE_L1, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "Reserved 0x008 read != 0");
            axi_read(ADDR_SCALE_L2, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "Reserved 0x00C read != 0");
            axi_read(ADDR_SCALE_L3, rd_val);
            if (rd_val !== 32'd0) $fatal(1, "Reserved 0x010 read != 0");
        end

        $readmemh("stimulus.hex", stim);
        n   = stim[0][31:0];
        ptr = 1;
        for (s = 0; s < n; s = s + 1) begin
            axi_write_ok(ADDR_T_ORIG, {16'd0, stim[ptr][15:0]});
            if ((stim[ptr][15:0] == 16'd0) || (stim[ptr][15:0] > 16'd1024)) begin
                ptr = ptr + 1;
                axi_write_err(ADDR_CTRL, 32'd1);
                axi_read(ADDR_CTRL, rd_val);
                if (rd_val[1] !== 1'b0)
                    $fatal(1, "Out-of-range sample started core");
                $fwrite(fd, "REFUSED 1\n");
                $fflush(fd);
            end else begin
                r   = {16'd0, stim[ptr][15:0]};
                ptr = ptr + 1;
                axi_write_ok(ADDR_CTRL, 32'd1);
                for (f = 0; f < r; f = f + 1) begin
                    cur_frame = stim[ptr];
                    ptr       = ptr + 1;
                    wait_frame_ready();
                    if (BINARY_SPIKE_MODE != 0) begin
                        bin_packed = 160'd0;
                        for (k = 0; k < 140; k = k + 1)
                            bin_packed[k] = cur_frame[4*k];
                        for (k = 0; k < 5; k = k + 1) begin
                            if (bin_packed[32*k +: 32] != 32'd0)
                                axi_write_ok(ADDR_SPIKE_IN_BASE + (k[8:0] * 9'd4),
                                             bin_packed[32*k +: 32]);
                        end
                    end else begin
                        for (k = 0; k < 18; k = k + 1) begin
                            if (k < 17) begin
                                if (cur_frame[32*k +: 32] != 32'd0)
                                    axi_write_ok(ADDR_FRAME_CNT_BASE + (k[8:0] * 9'd4),
                                                 cur_frame[32*k +: 32]);
                            end else begin
                                if (cur_frame[559:544] != 16'd0)
                                    axi_write_ok(ADDR_FRAME_CNT_17,
                                                 {16'd0, cur_frame[559:544]});
                            end
                        end
                    end
                    axi_write_ok(ADDR_SPIKE_COMMIT, 32'd1);
                end
                wait_done();
                for (k = 0; k < 20; k = k + 1) begin
                    axi_read(ADDR_OUT_V_BASE + (k[8:0] * 9'd4), rd_val);
                    $fwrite(fd, "%0d ", $signed(rd_val));
                end
                $fwrite(fd, "\n");
                $fflush(fd);
            end
        end

        $fclose(fd);
        $finish;
    end
endmodule
