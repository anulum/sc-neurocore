// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Clock domain crossing primitives for multi-rate systems (Tier 3.4)

// --- 2-FF synchronizer for single-bit CDC ---
module sc_sync_2ff (
    input  wire clk_dst,
    input  wire rst_dst,
    input  wire data_in,
    output wire data_out
);
    reg [1:0] sync_ff;

    always @(posedge clk_dst or posedge rst_dst) begin
        if (rst_dst)
            sync_ff <= 2'b0;
        else
            sync_ff <= {sync_ff[0], data_in};
    end

    assign data_out = sync_ff[1];
endmodule


// --- Gray-code counter for FIFO pointer CDC ---
module sc_gray_counter #(
    parameter WIDTH = 4
)(
    input  wire             clk,
    input  wire             rst,
    input  wire             enable,
    output wire [WIDTH-1:0] gray_out,
    output wire [WIDTH-1:0] binary_out
);
    reg [WIDTH-1:0] binary;

    always @(posedge clk or posedge rst) begin
        if (rst)
            binary <= {WIDTH{1'b0}};
        else if (enable)
            binary <= binary + 1;
    end

    assign binary_out = binary;
    assign gray_out   = binary ^ (binary >> 1);
endmodule


// --- Async FIFO for multi-clock domain data transfer ---
module sc_async_fifo #(
    parameter DATA_WIDTH = 64,
    parameter DEPTH_LOG2 = 4  // FIFO depth = 2^DEPTH_LOG2
)(
    // Write side
    input  wire                    wr_clk,
    input  wire                    wr_rst,
    input  wire [DATA_WIDTH-1:0]   wr_data,
    input  wire                    wr_en,
    output wire                    wr_full,

    // Read side
    input  wire                    rd_clk,
    input  wire                    rd_rst,
    output wire [DATA_WIDTH-1:0]   rd_data,
    input  wire                    rd_en,
    output wire                    rd_empty
);

    localparam DEPTH = 1 << DEPTH_LOG2;

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Write pointer (gray-coded for CDC)
    wire [DEPTH_LOG2:0] wr_ptr_gray, wr_ptr_bin;
    sc_gray_counter #(.WIDTH(DEPTH_LOG2+1)) wr_cnt (
        .clk(wr_clk), .rst(wr_rst),
        .enable(wr_en && !wr_full),
        .gray_out(wr_ptr_gray), .binary_out(wr_ptr_bin)
    );

    // Read pointer (gray-coded for CDC)
    wire [DEPTH_LOG2:0] rd_ptr_gray, rd_ptr_bin;
    sc_gray_counter #(.WIDTH(DEPTH_LOG2+1)) rd_cnt (
        .clk(rd_clk), .rst(rd_rst),
        .enable(rd_en && !rd_empty),
        .gray_out(rd_ptr_gray), .binary_out(rd_ptr_bin)
    );

    // Synchronize write pointer to read domain
    reg [DEPTH_LOG2:0] wr_ptr_gray_rd [0:1];
    always @(posedge rd_clk or posedge rd_rst) begin
        if (rd_rst) begin
            wr_ptr_gray_rd[0] <= 0;
            wr_ptr_gray_rd[1] <= 0;
        end else begin
            wr_ptr_gray_rd[0] <= wr_ptr_gray;
            wr_ptr_gray_rd[1] <= wr_ptr_gray_rd[0];
        end
    end

    // Synchronize read pointer to write domain
    reg [DEPTH_LOG2:0] rd_ptr_gray_wr [0:1];
    always @(posedge wr_clk or posedge wr_rst) begin
        if (wr_rst) begin
            rd_ptr_gray_wr[0] <= 0;
            rd_ptr_gray_wr[1] <= 0;
        end else begin
            rd_ptr_gray_wr[0] <= rd_ptr_gray;
            rd_ptr_gray_wr[1] <= rd_ptr_gray_wr[0];
        end
    end

    // Full/empty flags
    assign wr_full  = (wr_ptr_gray == {~rd_ptr_gray_wr[1][DEPTH_LOG2:DEPTH_LOG2-1],
                                         rd_ptr_gray_wr[1][DEPTH_LOG2-2:0]});
    assign rd_empty = (rd_ptr_gray == wr_ptr_gray_rd[1]);

    // Memory write
    always @(posedge wr_clk) begin
        if (wr_en && !wr_full)
            mem[wr_ptr_bin[DEPTH_LOG2-1:0]] <= wr_data;
    end

    // Memory read
    assign rd_data = mem[rd_ptr_bin[DEPTH_LOG2-1:0]];

endmodule
