// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Formal Verification for SC AXIL CFG

`default_nettype none

module sc_axil_cfg_formal (
    input wire        clk,
    input wire        rst_n,

    input wire [7:0]  S_AXI_AWADDR,
    input wire        S_AXI_AWVALID,
    input wire [31:0] S_AXI_WDATA,
    input wire [3:0]  S_AXI_WSTRB,
    input wire        S_AXI_WVALID,
    input wire        S_AXI_BREADY,

    input wire [7:0]  S_AXI_ARADDR,
    input wire        S_AXI_ARVALID,
    input wire        S_AXI_RREADY,

    input wire        stat_busy,
    input wire        stat_done,
    input wire [31:0] stat_rate_q16_0,
    input wire [31:0] stat_rate_q16_1,
    input wire [31:0] stat_rate_q16_2,
    input wire [31:0] stat_rate_q16_3,
    input wire [31:0] stat_rate_q16_4,
    input wire [31:0] stat_rate_q16_5,
    input wire [31:0] stat_rate_q16_6
);

    wire        S_AXI_AWREADY;
    wire        S_AXI_WREADY;
    wire [1:0]  S_AXI_BRESP;
    wire        S_AXI_BVALID;
    wire        S_AXI_ARREADY;
    wire [31:0] S_AXI_RDATA;
    wire [1:0]  S_AXI_RRESP;
    wire        S_AXI_RVALID;

    wire        cfg_start_pulse;
    wire [15:0] cfg_x_input [0:2];
    wire [15:0] cfg_weight  [0:2];
    wire [15:0] cfg_y_min;
    wire [15:0] cfg_y_max;
    wire [15:0] cfg_leak;
    wire [15:0] cfg_gain;
    wire [31:0] cfg_stream_len;
    wire [31:0] cfg_dt_ms;
    wire [31:0] cfg_scale_q16;

    wire [31:0] stat_rate_q16 [0:6];
    assign stat_rate_q16[0] = stat_rate_q16_0;
    assign stat_rate_q16[1] = stat_rate_q16_1;
    assign stat_rate_q16[2] = stat_rate_q16_2;
    assign stat_rate_q16[3] = stat_rate_q16_3;
    assign stat_rate_q16[4] = stat_rate_q16_4;
    assign stat_rate_q16[5] = stat_rate_q16_5;
    assign stat_rate_q16[6] = stat_rate_q16_6;

    sc_axil_cfg #(
        .C_S_AXI_ADDR_WIDTH(8),
        .C_S_AXI_DATA_WIDTH(32)
    ) uut (
        .S_AXI_ACLK(clk),
        .S_AXI_ARESETN(rst_n),

        .S_AXI_AWADDR(S_AXI_AWADDR),
        .S_AXI_AWVALID(S_AXI_AWVALID),
        .S_AXI_AWREADY(S_AXI_AWREADY),

        .S_AXI_WDATA(S_AXI_WDATA),
        .S_AXI_WSTRB(S_AXI_WSTRB),
        .S_AXI_WVALID(S_AXI_WVALID),
        .S_AXI_WREADY(S_AXI_WREADY),

        .S_AXI_BRESP(S_AXI_BRESP),
        .S_AXI_BVALID(S_AXI_BVALID),
        .S_AXI_BREADY(S_AXI_BREADY),

        .S_AXI_ARADDR(S_AXI_ARADDR),
        .S_AXI_ARVALID(S_AXI_ARVALID),
        .S_AXI_ARREADY(S_AXI_ARREADY),

        .S_AXI_RDATA(S_AXI_RDATA),
        .S_AXI_RRESP(S_AXI_RRESP),
        .S_AXI_RVALID(S_AXI_RVALID),
        .S_AXI_RREADY(S_AXI_RREADY),

        .cfg_start_pulse(cfg_start_pulse),
        .cfg_x_input(cfg_x_input),
        .cfg_weight(cfg_weight),
        .cfg_y_min(cfg_y_min),
        .cfg_y_max(cfg_y_max),
        .cfg_leak(cfg_leak),
        .cfg_gain(cfg_gain),
        .cfg_stream_len(cfg_stream_len),
        .cfg_dt_ms(cfg_dt_ms),
        .cfg_scale_q16(cfg_scale_q16),

        .stat_busy(stat_busy),
        .stat_done(stat_done),
        .stat_rate_q16(stat_rate_q16)
    );

`ifdef FORMAL
    reg past_valid = 0;
    always @(posedge clk)
        past_valid <= 1;

    // 1. AWREADY and WREADY assert only during a valid transaction
    //    Both require their respective VALID signals + aw_en
    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            if (S_AXI_AWREADY)
                assert($past(S_AXI_AWVALID) && $past(S_AXI_WVALID));
            if (S_AXI_WREADY)
                assert($past(S_AXI_WVALID) && $past(S_AXI_AWVALID));
        end
    end

    // 2. BVALID de-asserts after BREADY handshake
    always @(posedge clk) begin
        if (past_valid && rst_n && $past(S_AXI_BVALID) && $past(S_AXI_BREADY))
            assert(!S_AXI_BVALID);
    end

    // 3. RVALID de-asserts after RREADY handshake
    always @(posedge clk) begin
        if (past_valid && rst_n && $past(S_AXI_RVALID) && $past(S_AXI_RREADY))
            assert(!S_AXI_RVALID);
    end

    // 4. cfg_start_pulse is single-cycle: if it was high, it clears next cycle
    //    (clears on the BVALID+BREADY beat or stays low)
    always @(posedge clk) begin
        if (past_valid && rst_n && $past(cfg_start_pulse) && $past(rst_n))
            assert(!cfg_start_pulse);
    end

    // 5. After reset, handshake outputs are de-asserted
    always @(posedge clk) begin
        if (past_valid && !rst_n) begin
            assert(!S_AXI_AWREADY);
            assert(!S_AXI_WREADY);
            assert(!S_AXI_BVALID);
            assert(!S_AXI_RVALID);
            assert(!S_AXI_ARREADY);
            assert(!cfg_start_pulse);
        end
    end

    // 6. Cover: write transaction completes
    always @(posedge clk) begin
        if (past_valid && rst_n)
            cover(S_AXI_BVALID);
    end

    // 7. Cover: read transaction completes
    always @(posedge clk) begin
        if (past_valid && rst_n)
            cover(S_AXI_RVALID);
    end
`endif

endmodule
