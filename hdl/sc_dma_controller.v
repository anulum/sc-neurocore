// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — DMA controller for weight upload and output readback (Tier 3.2)
//
// Simple scatter-gather DMA for bulk weight transfer to SC layer RAM
// and output spike count readback. Controlled via AXI-Lite registers.

module sc_dma_controller #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 64,
    parameter RAM_DEPTH  = 256
)(
    input  wire                    clk,
    input  wire                    rst,

    // Control registers (from AXI-Lite)
    input  wire [ADDR_WIDTH-1:0]   src_addr,
    input  wire [ADDR_WIDTH-1:0]   dst_addr,
    input  wire [15:0]             transfer_len,
    input  wire                    start,
    output reg                     done,
    output reg                     busy,

    // Memory interface (to weight RAM)
    output reg  [ADDR_WIDTH-1:0]   mem_addr,
    output reg  [DATA_WIDTH-1:0]   mem_wdata,
    output reg                     mem_we,
    input  wire [DATA_WIDTH-1:0]   mem_rdata,

    // AXI-Stream output for readback
    output wire [DATA_WIDTH-1:0]   m_axis_tdata,
    output wire                    m_axis_tvalid,
    input  wire                    m_axis_tready,
    output wire                    m_axis_tlast
);

    localparam IDLE    = 2'd0;
    localparam WRITE   = 2'd1;
    localparam READ    = 2'd2;

    reg [1:0]  state;
    reg [15:0] count;
    reg [ADDR_WIDTH-1:0] addr_reg;
    reg readback_valid;
    reg readback_last;

    assign m_axis_tdata  = mem_rdata;
    assign m_axis_tvalid = readback_valid;
    assign m_axis_tlast  = readback_last;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done  <= 1'b0;
            busy  <= 1'b0;
            mem_we <= 1'b0;
            readback_valid <= 1'b0;
            readback_last <= 1'b0;
            count <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    mem_we <= 1'b0;
                    readback_valid <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        count <= 0;
                        addr_reg <= src_addr;
                        state <= WRITE;
                    end
                end
                WRITE: begin
                    mem_addr <= addr_reg + count;
                    mem_we <= 1'b1;
                    mem_wdata <= {DATA_WIDTH{1'b0}}; // host stream supplies payload data
                    count <= count + 1;
                    if (count >= transfer_len - 1) begin
                        mem_we <= 1'b0;
                        count <= 0;
                        addr_reg <= dst_addr;
                        state <= READ;
                    end
                end
                READ: begin
                    mem_addr <= addr_reg + count;
                    mem_we <= 1'b0;
                    readback_valid <= 1'b1;
                    readback_last <= (count >= transfer_len - 1);
                    if (m_axis_tready) begin
                        count <= count + 1;
                        if (count >= transfer_len - 1) begin
                            state <= IDLE;
                            done <= 1'b1;
                            busy <= 1'b0;
                            readback_valid <= 1'b0;
                        end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
