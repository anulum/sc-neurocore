// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal verification harness for sc_shd_event_axi

`default_nettype none

module sc_shd_event_axi_formal (
    input wire        clk,
    input wire        rst_n,
    input wire [8:0]  S_AXI_AWADDR,
    input wire [2:0]  S_AXI_AWPROT,
    input wire        S_AXI_AWVALID,
    input wire [31:0] S_AXI_WDATA,
    input wire [3:0]  S_AXI_WSTRB,
    input wire        S_AXI_WVALID,
    input wire        S_AXI_BREADY,
    input wire [8:0]  S_AXI_ARADDR,
    input wire [2:0]  S_AXI_ARPROT,
    input wire        S_AXI_ARVALID,
    input wire        S_AXI_RREADY
);
    wire        S_AXI_AWREADY;
    wire        S_AXI_WREADY;
    wire [1:0]  S_AXI_BRESP;
    wire        S_AXI_BVALID;
    wire        S_AXI_ARREADY;
    wire [31:0] S_AXI_RDATA;
    wire [1:0]  S_AXI_RRESP;
    wire        S_AXI_RVALID;

    wire [3:0]   f_core_state;
    wire         f_core_busy;
    wire         f_core_error;
    wire         f_core_frame_ready;
    wire         f_sealed;
    wire         f_crc_mismatch;
    wire         f_refused_busy;
    wire         f_refused_range;
    wire         f_commit_refused;
    wire         f_start_r;
    wire         f_frame_valid_r;
    wire         f_load_we_r;
    wire [2:0]   f_load_commit_sel;
    wire [6:0]   f_load_commit_grp;
    wire [7:0]   f_load_commit_addr;
    wire [15:0]  f_core_t_orig;
    wire [31:0]  f_load_count;
    wire [31:0]  f_load_crc32;
    wire [31:0]  f_expected_crc32;
    wire [31:0]  f_crc_state;
    wire [559:0] f_frame_buf;
    wire [8:0]   f_axi_awaddr;
    wire         f_wr_en;
    wire         f_wr_aligned;
    wire         f_is_ctrl_w;
    wire         f_start_req;
    wire         f_start_admissible;
    wire         f_loader_busy_err;
    wire         f_loader_range_err;
    wire         f_commit_err;

    // Exposed internal wires are added to sc_shd_event_axi as output ports by
    // Yosys `expose` prior to elaborating sc_shd_event_axi_formal, and verified
    // driver-complete by `check -assert`.
    sc_shd_event_axi uut (
        .S_AXI_ACLK      (clk),
        .S_AXI_ARESETN   (rst_n),
        .S_AXI_AWADDR    (S_AXI_AWADDR),
        .S_AXI_AWPROT    (S_AXI_AWPROT),
        .S_AXI_AWVALID   (S_AXI_AWVALID),
        .S_AXI_AWREADY   (S_AXI_AWREADY),
        .S_AXI_WDATA     (S_AXI_WDATA),
        .S_AXI_WSTRB     (S_AXI_WSTRB),
        .S_AXI_WVALID    (S_AXI_WVALID),
        .S_AXI_WREADY    (S_AXI_WREADY),
        .S_AXI_BRESP     (S_AXI_BRESP),
        .S_AXI_BVALID    (S_AXI_BVALID),
        .S_AXI_BREADY    (S_AXI_BREADY),
        .S_AXI_ARADDR    (S_AXI_ARADDR),
        .S_AXI_ARPROT    (S_AXI_ARPROT),
        .S_AXI_ARVALID   (S_AXI_ARVALID),
        .S_AXI_ARREADY   (S_AXI_ARREADY),
        .S_AXI_RDATA     (S_AXI_RDATA),
        .S_AXI_RRESP     (S_AXI_RRESP),
        .S_AXI_RVALID    (S_AXI_RVALID),
        .S_AXI_RREADY    (S_AXI_RREADY),
        .sealed          (f_sealed),
        .crc_mismatch    (f_crc_mismatch),
        .refused_busy    (f_refused_busy),
        .refused_range   (f_refused_range),
        .commit_refused  (f_commit_refused),
        .start_r         (f_start_r),
        .frame_valid_r   (f_frame_valid_r),
        .load_we_r       (f_load_we_r),
        .load_commit_sel (f_load_commit_sel),
        .load_commit_grp (f_load_commit_grp),
        .load_commit_addr(f_load_commit_addr),
        .core_t_orig     (f_core_t_orig),
        .load_count      (f_load_count),
        .load_crc32      (f_load_crc32),
        .expected_crc32  (f_expected_crc32),
        .crc_state       (f_crc_state),
        .frame_buf       (f_frame_buf),
        .core_busy       (f_core_busy),
        .core_error      (f_core_error),
        .core_frame_ready(f_core_frame_ready),
        .axi_awaddr      (f_axi_awaddr),
        .wr_en           (f_wr_en),
        .wr_aligned      (f_wr_aligned),
        .is_ctrl_w       (f_is_ctrl_w),
        .start_req       (f_start_req),
        .start_admissible(f_start_admissible),
        .loader_busy_err (f_loader_busy_err),
        .loader_range_err(f_loader_range_err),
        .commit_err      (f_commit_err),
        .\u_core.state   (f_core_state)
    );

`ifdef FORMAL
    reg past_valid = 1'b0;
    always @(posedge clk) begin
        if (!past_valid)
            assume(!rst_n);
        past_valid <= 1'b1;
    end

    // Standard AXI4-Lite master stability assumptions while VALID && !READY:
    always @(posedge clk) begin
        if (past_valid && $past(rst_n) && rst_n) begin
            if ($past(S_AXI_AWVALID && !S_AXI_AWREADY)) begin
                assume(S_AXI_AWVALID);
                assume($stable(S_AXI_AWADDR));
                assume($stable(S_AXI_AWPROT));
            end
            if ($past(S_AXI_WVALID && !S_AXI_WREADY)) begin
                assume(S_AXI_WVALID);
                assume($stable(S_AXI_WDATA));
                assume($stable(S_AXI_WSTRB));
            end
            if ($past(S_AXI_ARVALID && !S_AXI_ARREADY)) begin
                assume(S_AXI_ARVALID);
                assume($stable(S_AXI_ARADDR));
                assume($stable(S_AXI_ARPROT));
            end
        end
    end

    // Core FSM reachable-state strengthening invariants for unbounded induction:
    always @(posedge clk) begin
        if (rst_n) begin
            assert(f_core_state <= 4'd11);
            assert((f_core_state == 4'd0) == !f_core_busy);
            assert(!f_core_error);
            if (S_AXI_AWREADY || S_AXI_WREADY)
                assert(!S_AXI_BVALID && !f_start_r && !f_frame_valid_r && !f_load_we_r);
        end
    end

    // 1. Reset clears seal, pulses, sticky flags, and AXI handshakes.
    always @(posedge clk) begin
        if (past_valid && !$past(rst_n)) begin
            assert(!f_sealed);
            assert(f_load_count == 32'd0);
            assert(f_load_crc32 == 32'd0);
            assert(!f_start_r);
            assert(!f_frame_valid_r);
            assert(!f_load_we_r);
            assert(!f_commit_refused);
            assert(!f_refused_busy);
            assert(!f_refused_range);
            assert(!S_AXI_AWREADY);
            assert(!S_AXI_WREADY);
            assert(!S_AXI_BVALID);
            assert(!S_AXI_ARREADY);
            assert(!S_AXI_RVALID);
        end
    end

    // 2. Seal invariant: sealed implies load_count > 0 and load_crc32 == expected_crc32.
    always @(posedge clk) begin
        if (rst_n && f_sealed) begin
            assert(f_load_count > 32'd0);
            assert(f_load_crc32 == f_expected_crc32);
        end
    end

    // 3. Start admission invariants:
    //    - never started unsealed;
    //    - never started with t_orig outside [1, 1024];
    //    - never started while core_busy or alongside load_we.
    always @(posedge clk) begin
        if (rst_n && f_start_r) begin
            assert(f_sealed);
            assert(f_load_count > 32'd0);
            assert(f_load_crc32 == f_expected_crc32);
            assert(f_core_t_orig >= 16'd1);
            assert(f_core_t_orig <= 16'd1024);
            assert(!f_core_busy);
            assert(!f_load_we_r);
        end
    end

    // 4. Stepping protocol invariant: frame_valid is pulsed only while frame_ready.
    always @(posedge clk) begin
        if (rst_n && f_frame_valid_r) begin
            assert(f_core_frame_ready);
            assert(!f_start_r);
        end
    end

    // 5. Loader write-enable invariant: load_we pulses only while core is idle and
    //    always carries an in-range committed target (sc_shd_event_p.v:56-62, 97-102, 167-178).
    always @(posedge clk) begin
        if (rst_n && f_load_we_r) begin
            assert(!f_core_busy);
            assert(!f_start_r);
            assert(!f_sealed);
            assert(
                ((f_load_commit_sel == 3'd0) && (f_load_commit_grp < 7'd16) && (f_load_commit_addr < 8'd140)) ||
                ((f_load_commit_sel == 3'd1) && (f_load_commit_grp < 7'd16) && (f_load_commit_addr < 8'd128)) ||
                ((f_load_commit_sel == 3'd2) && (f_load_commit_grp < 7'd3)  && (f_load_commit_addr < 8'd128)) ||
                ((f_load_commit_sel == 3'd3) && (f_load_commit_grp == 7'd0) && (f_load_commit_addr < 8'd140)) ||
                ((f_load_commit_sel == 3'd4) && (f_load_commit_grp == 7'd0) && (f_load_commit_addr < 8'd128))
            );
        end
    end

    // 6. Single-cycle pulse invariants for start, frame_valid, and load_we,
    //    plus frame_buf zeroing immediately after frame_valid.
    always @(posedge clk) begin
        if (past_valid && $past(rst_n) && rst_n) begin
            if ($past(f_start_r))
                assert(!f_start_r);
            if ($past(f_frame_valid_r)) begin
                assert(!f_frame_valid_r);
                assert(f_frame_buf == 560'd0);
            end
            if ($past(f_load_we_r))
                assert(!f_load_we_r);
        end
    end

    // 7. Negative-path response invariants:
    //    - completed write to reserved scale offsets 0x08, 0x0C, 0x10 returns SLVERR;
    //    - unsealed or out-of-range CTRL.start returns SLVERR without starting;
    //    - loader write while busy returns SLVERR, sets refused_busy, and keeps state;
    //    - out-of-range LOAD_WDATA_HI commit returns SLVERR, sets refused_range,
    //      clears sealed, suppresses load_we, and preserves count and CRC;
    //    - SPIKE_COMMIT while !status_frame_ready returns SLVERR, sets commit_refused,
    //      and leaves frame_buf unchanged.
    always @(posedge clk) begin
        if (past_valid && $past(rst_n) && rst_n && $past(f_wr_en)) begin
            if ($past((f_axi_awaddr[8:2] == 7'h02) || (f_axi_awaddr[8:2] == 7'h03) || (f_axi_awaddr[8:2] == 7'h04))) begin
                assert(S_AXI_BVALID && (S_AXI_BRESP == 2'b10));
            end
            if ($past(f_wr_aligned && f_is_ctrl_w && f_start_req && !f_start_admissible)) begin
                assert(S_AXI_BVALID && (S_AXI_BRESP == 2'b10));
                assert(!f_start_r);
            end
            if ($past(f_wr_aligned && f_loader_busy_err)) begin
                assert(S_AXI_BVALID && (S_AXI_BRESP == 2'b10));
                assert(f_refused_busy);
                assert(!f_load_we_r);
                assert(f_sealed == $past(f_sealed));
                assert(f_load_count == $past(f_load_count));
                assert(f_crc_state == $past(f_crc_state));
            end
            if ($past(f_wr_aligned && f_loader_range_err)) begin
                assert(S_AXI_BVALID && (S_AXI_BRESP == 2'b10));
                assert(f_refused_range);
                assert(!f_load_we_r);
                assert(!f_sealed);
                assert(f_load_count == $past(f_load_count));
                assert(f_crc_state == $past(f_crc_state));
            end
            if ($past(f_wr_aligned && f_commit_err)) begin
                assert(S_AXI_BVALID && (S_AXI_BRESP == 2'b10));
                assert(f_commit_refused);
                assert(!f_frame_valid_r);
                if (!$past(f_frame_valid_r))
                    assert(f_frame_buf == $past(f_frame_buf));
            end
        end
    end

    // 8. Cover witnesses:
    //    - full integrity-checked load -> seal -> start -> frame_ready -> commit;
    //    - CRC mismatch refusal on seal;
    //    - refused loader write while busy;
    //    - refused out-of-range LOAD_WDATA_HI commit;
    //    - refused SPIKE_COMMIT while !frame_ready.
    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            cover(f_sealed && f_frame_valid_r && f_core_frame_ready);
            cover(f_crc_mismatch && !f_sealed);
            cover(f_refused_busy && S_AXI_BVALID && (S_AXI_BRESP == 2'b10));
            cover(f_refused_range && S_AXI_BVALID && (S_AXI_BRESP == 2'b10));
            cover(f_commit_refused && S_AXI_BVALID && (S_AXI_BRESP == 2'b10));
        end
    end
`endif
endmodule
`default_nettype wire
