// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AXI4-Lite slave wrapper and integrity-checked loader for sc_shd_event_p
//
// Register map (byte addresses, 32-bit aligned, C_S_AXI_ADDR_WIDTH = 9):
//
//   Offset       Dir  Name              Description
//   -----------  ---  ----------------  ----------------------------------------------------
//   0x000        R/W  CTRL              W: [0]=start (requires idle, sealed, 1 <= T_ORIG <= 1024)
//                                       R: [1]=busy, [2]=done (sticky until start),
//                                          [3]=frame_ready, [4]=error, [5]=overflow,
//                                          [6]=commit_refused (sticky until start)
//   0x004        R/W  T_ORIG            [15:0] input frame count (reset 250; valid 1..1024)
//   0x008..0x010 -    RESERVED          Reserved on the event bitstream (the event core's
//                                       layer scaling is fixed at synthesis by its parameters);
//                                       reads 0; writes refused with SLVERR
//   0x014..0x024 W    SPIKE_IN_0..4     Write-only (reads 0); 140 binary spike bits expanded
//                                       to {3'b000, b} nibbles
//   0x028        W    SPIKE_COMMIT      Write-only (reads 0); commit 560-bit frame_buf when
//                                       frame_ready=1; clears frame_buf upon capture
//   0x02C..0x03C -    RESERVED          Reads 0; writes refused with SLVERR
//   0x040..0x08C R    OUT_V_0..19       Signed 32-bit class accumulators (20 words)
//   0x090        R/W  LOAD_CTRL         W: [0]=clear, [1]=auto_increment, [2]=seal; R: [1]
//   0x094        R/W  LOAD_TARGET       [2:0]=load_sel, [9:3]=load_grp, [17:10]=load_addr
//   0x098        R/W  LOAD_WDATA_LO     Low 32 bits of 64-bit group word
//   0x09C        R/W  LOAD_WDATA_HI     High 32 bits; write commits 64-bit word when idle and
//                                       LOAD_TARGET is in range, else refused with SLVERR
//   0x0A0        R    LOAD_COUNT        Words committed since clear
//   0x0A4        R    LOAD_CRC32        Reflected IEEE CRC-32 (0xEDB88320, LE bytes) over words
//   0x0A8        R/W  EXPECTED_CRC32    Expected CRC-32 written by host before seal
//   0x0AC        R    LOAD_STATUS       [0]=sealed, [1]=crc_mismatch, [2]=refused_busy,
//                                       [3]=refused_range (sticky until LOAD_CTRL.clear)
//   0x0B0..0x0B4 -    RESERVED          Reads 0; writes refused with SLVERR
//   0x0B8..0x0FC W    FRAME_COUNT_0..17 Write-only (reads 0); 140 4-bit channel counts
//                                       (FRAME_COUNT_17[31:16] == 0 enforced via SLVERR)
//
// Integrity-checked load note:
//   CRC-32 detects accidental transfer corruption; it does not authenticate the network.
//   Out-of-range LOAD_TARGET commits are refused with SLVERR before reaching sc_shd_event_p.
//   Core rst_n is driven solely by S_AXI_ARESETN so per-sample start preserves loaded RAM
//   and delay tables, while S_AXI_ARESETN clears sealed and requires reloading after reset.

`timescale 1ns / 1ps

module sc_shd_event_axi #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 9,
    parameter integer P                  = 8,
    parameter integer FRAC_BITS          = 0,
    parameter integer THETA1             = 1,
    parameter integer THETA2             = 1,
    parameter integer V1_WIDTH           = 22,
    parameter integer V2_WIDTH           = 17,
    parameter integer CAP1               = 62,
    parameter integer CAP2               = 39,
    parameter integer WSHAPE             = 6,
    parameter integer QSHAPE             = 12
)(
    input  wire                              S_AXI_ACLK,
    input  wire                              S_AXI_ARESETN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_AWADDR,
    input  wire [2:0]                        S_AXI_AWPROT,
    input  wire                              S_AXI_AWVALID,
    output wire                              S_AXI_AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0]   S_AXI_WSTRB,
    input  wire                              S_AXI_WVALID,
    output wire                              S_AXI_WREADY,
    output wire [1:0]                        S_AXI_BRESP,
    output wire                              S_AXI_BVALID,
    input  wire                              S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_ARADDR,
    input  wire [2:0]                        S_AXI_ARPROT,
    input  wire                              S_AXI_ARVALID,
    output wire                              S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_RDATA,
    output wire [1:0]                        S_AXI_RRESP,
    output wire                              S_AXI_RVALID,
    input  wire                              S_AXI_RREADY
);
    localparam integer NI         = 140;
    localparam integer NH         = 128;
    localparam integer NO         = 20;
    localparam integer FRAME_BITS = NI * 4;
    localparam integer G_GROUPS   = NH / P;
    localparam integer PC_PASSES  = (NO + P - 1) / P;
    localparam [15:0]  T_MAX      = 16'd1024;
    localparam [1:0]   AXI_OKAY   = 2'b00;
    localparam [1:0]   AXI_SLVERR = 2'b10;

    // ----------------------------------------------------------------
    // AXI4-Lite handshake registers
    // ----------------------------------------------------------------
    reg                              axi_awready;
    reg                              axi_wready;
    reg                              axi_bvalid;
    reg [1:0]                        axi_bresp;
    reg                              axi_arready;
    reg                              axi_rvalid;
    reg [C_S_AXI_DATA_WIDTH-1:0]     axi_rdata;
    reg [C_S_AXI_ADDR_WIDTH-1:0]     axi_awaddr;
    reg [C_S_AXI_ADDR_WIDTH-1:0]     axi_araddr;

    wire unused_prot = ^S_AXI_AWPROT ^ ^S_AXI_ARPROT;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = AXI_OKAY;
    assign S_AXI_RVALID  = axi_rvalid;

    // ----------------------------------------------------------------
    // Helper functions: byte strobe merge, reflected CRC-32, target bounds
    // ----------------------------------------------------------------
    function [31:0] write_bytes(
        input [31:0] previous,
        input [31:0] data,
        input [3:0]  strobe
    );
        integer b;
        begin
            write_bytes = previous;
            for (b = 0; b < 4; b = b + 1)
                if (strobe[b]) write_bytes[8*b +: 8] = data[8*b +: 8];
        end
    endfunction

    // Reflected IEEE 802.3 CRC-32 (polynomial 0xEDB88320) over 64 bits
    // in little-endian byte order (matching Python struct.pack("<Q", word64)).
    function [31:0] crc32_u64_le(
        input [31:0] state_in,
        input [63:0] word_le
    );
        reg [31:0] c;
        integer    i;
        begin
            c = state_in;
            for (i = 0; i < 64; i = i + 1) begin
                if (c[0] ^ word_le[i])
                    c = (c >> 1) ^ 32'hEDB8_8320;
                else
                    c = (c >> 1);
            end
            crc32_u64_le = c;
        end
    endfunction

    // Per-selector bounds matching sc_shd_event_p.v:56-62, 97-102, 167-178.
    function is_valid_load_target(input [17:0] tgt);
        reg [2:0] sel;
        reg [6:0] grp;
        reg [7:0] addr;
        begin
            sel  = tgt[2:0];
            grp  = tgt[9:3];
            addr = tgt[17:10];
            case (sel)
                3'd0: is_valid_load_target = ({25'd0, grp} < G_GROUPS)  && (addr < 8'd140);
                3'd1: is_valid_load_target = ({25'd0, grp} < G_GROUPS)  && (addr < 8'd128);
                3'd2: is_valid_load_target = ({25'd0, grp} < PC_PASSES) && (addr < 8'd128);
                3'd3: is_valid_load_target = (grp == 7'd0) && (addr < 8'd140);
                3'd4: is_valid_load_target = (grp == 7'd0) && (addr < 8'd128);
                default: is_valid_load_target = 1'b0;
            endcase
        end
    endfunction

    function [17:0] next_load_target(input [17:0] cur);
        reg [2:0] sel;
        reg [6:0] grp;
        reg [7:0] addr;
        begin
            sel  = cur[2:0];
            grp  = cur[9:3];
            addr = cur[17:10];
            case (sel)
                3'd0: begin
                    if ({25'd0, grp} + 32'd1 < G_GROUPS)
                        grp = grp + 7'd1;
                    else begin
                        grp = 7'd0;
                        if (addr + 8'd1 < 8'd140)
                            addr = addr + 8'd1;
                        else begin
                            sel  = 3'd1;
                            addr = 8'd0;
                        end
                    end
                end
                3'd1: begin
                    if ({25'd0, grp} + 32'd1 < G_GROUPS)
                        grp = grp + 7'd1;
                    else begin
                        grp = 7'd0;
                        if (addr + 8'd1 < 8'd128)
                            addr = addr + 8'd1;
                        else begin
                            sel  = 3'd2;
                            addr = 8'd0;
                        end
                    end
                end
                3'd2: begin
                    if ({25'd0, grp} + 32'd1 < PC_PASSES)
                        grp = grp + 7'd1;
                    else begin
                        grp = 7'd0;
                        if (addr + 8'd1 < 8'd128)
                            addr = addr + 8'd1;
                        else begin
                            sel  = 3'd3;
                            addr = 8'd0;
                        end
                    end
                end
                3'd3: begin
                    grp = 7'd0;
                    if (addr + 8'd1 < 8'd140)
                        addr = addr + 8'd1;
                    else begin
                        sel  = 3'd4;
                        addr = 8'd0;
                    end
                end
                3'd4: begin
                    grp = 7'd0;
                    if (addr + 8'd1 < 8'd128)
                        addr = addr + 8'd1;
                    else begin
                        sel  = 3'd5;
                        addr = 8'd0;
                    end
                end
                default: begin
                    // Saturate at invalid selector once end of d2[127] is passed.
                end
            endcase
            next_load_target = {addr, grp, sel};
        end
    endfunction

    // ----------------------------------------------------------------
    // Control, frame-staging, and integrity-checked loader state
    // ----------------------------------------------------------------
    reg                  start_r;
    reg                  frame_valid_r;
    reg [15:0]           t_orig_reg;
    reg [15:0]           core_t_orig;
    reg [FRAME_BITS-1:0] frame_buf;
    reg                  done_sticky;
    reg                  core_done_d;
    reg                  commit_refused;

    reg                  auto_increment;
    reg [17:0]           load_target_reg;
    reg [31:0]           load_wdata_lo_reg;
    reg [31:0]           load_wdata_hi_reg;
    reg                  load_we_r;
    reg [2:0]            load_commit_sel;
    reg [6:0]            load_commit_grp;
    reg [7:0]            load_commit_addr;
    reg [63:0]           load_commit_data;
    reg [31:0]           load_count;
    reg [31:0]           crc_state;
    reg [31:0]           expected_crc32;
    reg                  sealed;
    reg                  crc_mismatch;
    reg                  refused_busy;
    reg                  refused_range;

    wire [31:0] load_crc32 = ~crc_state;

    // ----------------------------------------------------------------
    // Event-driven core instance (rst_n is S_AXI_ARESETN alone)
    // ----------------------------------------------------------------
    wire                  core_frame_ready;
    wire                  core_busy;
    wire                  core_done;
    wire                  core_error;
    wire                  core_overflow;
    wire [NO*32-1:0]      core_out_packed;

    wire [1023:0] load_bus_wide = {960'd0, load_commit_data};

    sc_shd_event_p #(
        .P        (P),
        .FRAC_BITS(FRAC_BITS),
        .THETA1   (THETA1),
        .THETA2   (THETA2),
        .V1_WIDTH (V1_WIDTH),
        .V2_WIDTH (V2_WIDTH),
        .CAP1     (CAP1),
        .CAP2     (CAP2),
        .WSHAPE   (WSHAPE),
        .QSHAPE   (QSHAPE)
    ) u_core (
        .clk        (S_AXI_ACLK),
        .rst_n      (S_AXI_ARESETN),
        .load_we    (load_we_r),
        .load_sel   (load_commit_sel),
        .load_addr  (load_commit_addr),
        .load_grp   (load_commit_grp),
        .load_data  (load_bus_wide[8*P-1:0]),
        .start      (start_r),
        .t_orig     (core_t_orig),
        .frame      (frame_buf),
        .frame_valid(frame_valid_r),
        .frame_ready(core_frame_ready),
        .busy       (core_busy),
        .done       (core_done),
        .error      (core_error),
        .overflow   (core_overflow),
        .out_packed (core_out_packed)
    );

    wire wrapper_busy       = start_r || core_busy;
    wire status_frame_ready = core_frame_ready && !frame_valid_r && !start_r;

    // ----------------------------------------------------------------
    // AXI4-Lite write address & data channel handshakes
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 1'b0;
            axi_wready  <= 1'b0;
            axi_awaddr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else if (!axi_awready && !axi_bvalid && S_AXI_AWVALID && S_AXI_WVALID) begin
            axi_awready <= 1'b1;
            axi_wready  <= 1'b1;
            axi_awaddr  <= S_AXI_AWADDR;
        end else begin
            axi_awready <= 1'b0;
            axi_wready  <= 1'b0;
        end
    end

    wire       wr_en            = axi_awready && axi_wready && S_AXI_AWVALID && S_AXI_WVALID;
    wire       wr_aligned       = (axi_awaddr[1:0] == 2'b00) && !axi_awaddr[8];
    wire [5:0] wr_word          = axi_awaddr[7:2];

    wire is_ctrl_w              = (wr_word == 6'h00);
    wire is_t_orig_w            = (wr_word == 6'h01);
    wire is_spike_in_w          = (wr_word >= 6'h05) && (wr_word <= 6'h09);
    wire is_spike_commit_w      = (wr_word == 6'h0A);
    wire is_wdata_hi_w          = (wr_word == 6'h27);
    wire is_loader_rw_w         = (wr_word == 6'h24) || (wr_word == 6'h25) ||
                                  (wr_word == 6'h26) || is_wdata_hi_w ||
                                  (wr_word == 6'h2A);
    wire is_frame_cnt_w         = (wr_word >= 6'h2E) && (wr_word <= 6'h3F);
    wire is_mapped_wr           = is_ctrl_w || is_t_orig_w || is_spike_in_w ||
                                  is_spike_commit_w || is_loader_rw_w || is_frame_cnt_w;

    wire start_req              = S_AXI_WSTRB[0] && S_AXI_WDATA[0];
    wire start_admissible       = !wrapper_busy && !load_we_r && sealed &&
                                  (t_orig_reg >= 16'd1) && (t_orig_reg <= T_MAX);
    wire ctrl_err               = is_ctrl_w && start_req && !start_admissible;
    wire commit_err             = is_spike_commit_w && !status_frame_ready;
    wire loader_busy_err        = is_loader_rw_w && (wrapper_busy || load_we_r);
    wire loader_range_err       = is_wdata_hi_w && !loader_busy_err &&
                                  !is_valid_load_target(load_target_reg);
    wire frame_cnt17_err        = (wr_word == 6'h3F) &&
                                  ((S_AXI_WSTRB[2] && (S_AXI_WDATA[23:16] != 8'd0)) ||
                                   (S_AXI_WSTRB[3] && (S_AXI_WDATA[31:24] != 8'd0)));

    wire write_error            = !wr_aligned || !is_mapped_wr || ctrl_err ||
                                  commit_err || loader_busy_err || loader_range_err ||
                                  frame_cnt17_err;

    wire [31:0] merged_t_orig   = write_bytes({16'd0, t_orig_reg}, S_AXI_WDATA, S_AXI_WSTRB);
    wire [31:0] merged_target   = write_bytes({14'd0, load_target_reg}, S_AXI_WDATA, S_AXI_WSTRB);
    wire [31:0] merged_wdata_hi = write_bytes(load_wdata_hi_reg, S_AXI_WDATA, S_AXI_WSTRB);

    // ----------------------------------------------------------------
    // Synchronous state updates for control, frame staging, and loader
    // ----------------------------------------------------------------
    integer sb, sk, sch, fc_idx;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            start_r           <= 1'b0;
            frame_valid_r     <= 1'b0;
            t_orig_reg        <= 16'd250;
            core_t_orig       <= 16'd0;
            frame_buf         <= {FRAME_BITS{1'b0}};
            done_sticky       <= 1'b0;
            core_done_d       <= 1'b0;
            commit_refused    <= 1'b0;
            auto_increment    <= 1'b0;
            load_target_reg   <= 18'd0;
            load_wdata_lo_reg <= 32'd0;
            load_wdata_hi_reg <= 32'd0;
            load_we_r         <= 1'b0;
            load_commit_sel   <= 3'd0;
            load_commit_grp   <= 7'd0;
            load_commit_addr  <= 8'd0;
            load_commit_data  <= 64'd0;
            load_count        <= 32'd0;
            crc_state         <= 32'hFFFF_FFFF;
            expected_crc32    <= 32'd0;
            sealed            <= 1'b0;
            crc_mismatch      <= 1'b0;
            refused_busy      <= 1'b0;
            refused_range     <= 1'b0;
        end else begin
            start_r       <= 1'b0;
            frame_valid_r <= 1'b0;
            load_we_r     <= 1'b0;
            core_done_d   <= core_done;

            if (core_done && !core_done_d)
                done_sticky <= 1'b1;

            if (frame_valid_r)
                frame_buf <= {FRAME_BITS{1'b0}};

            if (wr_en && wr_aligned) begin
                if (commit_err)
                    commit_refused <= 1'b1;
                if (loader_busy_err)
                    refused_busy <= 1'b1;
                if (loader_range_err) begin
                    refused_range <= 1'b1;
                    sealed        <= 1'b0;
                end
            end

            if (wr_en && !write_error) begin
                case (wr_word)
                    6'h00: begin
                        if (start_req) begin
                            start_r        <= 1'b1;
                            core_t_orig    <= t_orig_reg;
                            done_sticky    <= 1'b0;
                            commit_refused <= 1'b0;
                        end
                    end
                    6'h01: t_orig_reg <= merged_t_orig[15:0];
                    6'h05, 6'h06, 6'h07, 6'h08, 6'h09: begin
                        for (sb = 0; sb < 4; sb = sb + 1) begin
                            if (S_AXI_WSTRB[sb]) begin
                                for (sk = 0; sk < 8; sk = sk + 1) begin
                                    sch = ({26'd0, wr_word - 6'h05} * 32) + (sb * 8) + sk;
                                    if (sch < NI)
                                        frame_buf[4*sch +: 4] <= {3'b000, S_AXI_WDATA[8*sb + sk]};
                                end
                            end
                        end
                    end
                    6'h0A: begin
                        frame_valid_r <= 1'b1;
                    end
                    6'h24: begin
                        if (S_AXI_WSTRB[0])
                            auto_increment <= S_AXI_WDATA[1];
                        if (S_AXI_WSTRB[0] && S_AXI_WDATA[0]) begin
                            load_count    <= 32'd0;
                            crc_state     <= 32'hFFFF_FFFF;
                            sealed        <= 1'b0;
                            crc_mismatch  <= 1'b0;
                            refused_busy  <= 1'b0;
                            refused_range <= 1'b0;
                        end else if (S_AXI_WSTRB[0] && S_AXI_WDATA[2]) begin
                            if ((load_count != 32'd0) && (load_crc32 == expected_crc32)) begin
                                sealed       <= 1'b1;
                                crc_mismatch <= 1'b0;
                            end else begin
                                sealed       <= 1'b0;
                                crc_mismatch <= 1'b1;
                            end
                        end else begin
                            sealed <= 1'b0;
                        end
                    end
                    6'h25: begin
                        load_target_reg <= merged_target[17:0];
                        sealed          <= 1'b0;
                    end
                    6'h26: begin
                        load_wdata_lo_reg <= write_bytes(load_wdata_lo_reg, S_AXI_WDATA, S_AXI_WSTRB);
                        sealed            <= 1'b0;
                    end
                    6'h27: begin
                        load_wdata_hi_reg <= merged_wdata_hi;
                        load_commit_sel   <= load_target_reg[2:0];
                        load_commit_grp   <= load_target_reg[9:3];
                        load_commit_addr  <= load_target_reg[17:10];
                        load_commit_data  <= {merged_wdata_hi, load_wdata_lo_reg};
                        load_we_r         <= 1'b1;
                        if (load_count != 32'hFFFF_FFFF)
                            load_count <= load_count + 32'd1;
                        crc_state         <= crc32_u64_le(crc_state, {merged_wdata_hi, load_wdata_lo_reg});
                        sealed            <= 1'b0;
                        if (auto_increment)
                            load_target_reg <= next_load_target(load_target_reg);
                    end
                    6'h2A: begin
                        expected_crc32 <= write_bytes(expected_crc32, S_AXI_WDATA, S_AXI_WSTRB);
                        sealed         <= 1'b0;
                    end
                    default: begin
                        if (is_frame_cnt_w) begin
                            fc_idx = {26'd0, wr_word - 6'h2E};
                            for (sb = 0; sb < 4; sb = sb + 1) begin
                                if (S_AXI_WSTRB[sb] && ((fc_idx < 17) || (sb < 2)))
                                    frame_buf[32*fc_idx + 8*sb +: 8] <= S_AXI_WDATA[8*sb +: 8];
                            end
                        end
                    end
                endcase
            end
        end
    end

    // ----------------------------------------------------------------
    // AXI4-Lite write response channel
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_bvalid <= 1'b0;
            axi_bresp  <= AXI_OKAY;
        end else if (wr_en && !axi_bvalid) begin
            axi_bvalid <= 1'b1;
            axi_bresp  <= write_error ? AXI_SLVERR : AXI_OKAY;
        end else if (S_AXI_BREADY && axi_bvalid) begin
            axi_bvalid <= 1'b0;
            axi_bresp  <= AXI_OKAY;
        end
    end

    // ----------------------------------------------------------------
    // AXI4-Lite read address & data channels
    // ----------------------------------------------------------------
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_arready <= 1'b0;
            axi_araddr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else if (!axi_arready && !axi_rvalid && S_AXI_ARVALID) begin
            axi_arready <= 1'b1;
            axi_araddr  <= S_AXI_ARADDR;
        end else begin
            axi_arready <= 1'b0;
        end
    end

    wire       rd_en   = axi_arready && S_AXI_ARVALID && !axi_rvalid;
    wire [5:0] rd_word = axi_araddr[7:2];

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rvalid <= 1'b0;
            axi_rdata  <= 32'd0;
        end else if (rd_en) begin
            axi_rvalid <= 1'b1;
            if (axi_araddr[8]) begin
                axi_rdata <= 32'd0;
            end else begin
                case (rd_word)
                    6'h00: axi_rdata <= {25'd0, commit_refused, core_overflow, core_error,
                                         status_frame_ready, done_sticky, wrapper_busy, 1'b0};
                    6'h01: axi_rdata <= {16'd0, t_orig_reg};
                    6'h24: axi_rdata <= {30'd0, auto_increment, 1'b0};
                    6'h25: axi_rdata <= {14'd0, load_target_reg};
                    6'h26: axi_rdata <= load_wdata_lo_reg;
                    6'h27: axi_rdata <= load_wdata_hi_reg;
                    6'h28: axi_rdata <= load_count;
                    6'h29: axi_rdata <= load_crc32;
                    6'h2A: axi_rdata <= expected_crc32;
                    6'h2B: axi_rdata <= {28'd0, refused_range, refused_busy, crc_mismatch, sealed};
                    default: begin
                        if ((rd_word >= 6'h10) && (rd_word <= 6'h23))
                            axi_rdata <= core_out_packed[32*(rd_word - 6'h10) +: 32];
                        else
                            axi_rdata <= 32'd0;
                    end
                endcase
            end
        end else if (axi_rvalid && S_AXI_RREADY) begin
            axi_rvalid <= 1'b0;
        end
    end

    wire unused_ok = unused_prot;
endmodule
