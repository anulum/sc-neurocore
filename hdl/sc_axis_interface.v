// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AXI-Stream interface for bulk bitstream transfer (Tier 3.1)
//
// Bridges between AXI-Stream bus and the SC dense layer core.
// TDATA carries packed bitstream words (64-bit), TLAST marks frame end.

module sc_axis_interface #(
    parameter DATA_WIDTH = 64,
    parameter N_INPUTS   = 4,
    parameter N_NEURONS  = 8
)(
    input  wire                    clk,
    input  wire                    rst,

    // AXI-Stream slave (input bitstreams)
    input  wire [DATA_WIDTH-1:0]   s_axis_tdata,
    input  wire                    s_axis_tvalid,
    output wire                    s_axis_tready,
    input  wire                    s_axis_tlast,

    // AXI-Stream master (output spike counts)
    output wire [DATA_WIDTH-1:0]   m_axis_tdata,
    output wire                    m_axis_tvalid,
    input  wire                    m_axis_tready,
    output wire                    m_axis_tlast,

    // Internal SC layer interface
    output wire [DATA_WIDTH-1:0]   layer_input_data,
    output wire                    layer_input_valid,
    input  wire [DATA_WIDTH-1:0]   layer_output_data,
    input  wire                    layer_output_valid
);

    // Input path: accept AXI-Stream, buffer for SC layer
    reg [DATA_WIDTH-1:0] input_buf;
    reg input_buf_valid;
    reg input_last;

    assign s_axis_tready = !input_buf_valid || layer_input_valid;
    assign layer_input_data = input_buf;
    assign layer_input_valid = input_buf_valid;

    always @(posedge clk) begin
        if (rst) begin
            input_buf_valid <= 1'b0;
            input_last <= 1'b0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            input_buf <= s_axis_tdata;
            input_buf_valid <= 1'b1;
            input_last <= s_axis_tlast;
        end else begin
            input_buf_valid <= 1'b0;
        end
    end

    // Output path: forward SC layer output as AXI-Stream. The dense-layer
    // interface is one-result-per-input, so the accepted input TLAST bit is
    // delayed until the corresponding output word is emitted.
    reg [DATA_WIDTH-1:0] output_buf;
    reg output_buf_valid;
    reg output_last;

    assign m_axis_tdata = output_buf;
    assign m_axis_tvalid = output_buf_valid;
    assign m_axis_tlast = output_last;

    always @(posedge clk) begin
        if (rst) begin
            output_buf_valid <= 1'b0;
            output_last <= 1'b0;
        end else if (layer_output_valid) begin
            output_buf <= layer_output_data;
            output_buf_valid <= 1'b1;
            output_last <= input_last;
        end else if (m_axis_tready) begin
            output_buf_valid <= 1'b0;
        end
    end

endmodule
